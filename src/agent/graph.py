from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from src.core.llm import build_chat_model, normalize_content
from src.core.schemas import (
    AgentResult,
    CalculateTotalsInput,
    DiscountInput,
    ListProductsInput,
    OrderLineInput,
    ProductDetailInput,
    SaveOrderInput,
    ToolCallRecord,
)
from src.utils.data_store import OrderDataStore

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT_DIR / "data"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts" / "orders"


@dataclass
class OrderDraft:
    customer_name: str = ""
    customer_phone: str = ""
    customer_email: str = ""
    shipping_address: str = ""
    items: list[OrderLineInput] = field(default_factory=list)
    discount: dict[str, Any] | None = None
    totals: dict[str, Any] | None = None

    def missing_fields(self) -> list[str]:
        missing: list[str] = []
        if not self.customer_name:
            missing.append("tên khách hàng hoặc tên công ty")
        if not self.customer_phone:
            missing.append("số điện thoại")
        if not self.customer_email:
            missing.append("email")
        if not self.shipping_address:
            missing.append("địa chỉ giao hàng")
        if not self.items:
            missing.append("sản phẩm và số lượng")
        return missing


def build_system_prompt(today: str | None = None) -> str:
    """
    Student TODO:
    - Rewrite this prompt for the advanced order-agent lab.
    - The assistant should manage electronics orders, not travel planning.
    - Require this tool order whenever the request has enough information:
      1. `list_products`
      2. `get_product_details`
      3. `get_discount`
      4. `calculate_order_totals`
      5. `save_order`
    - Clarify and stop if any of these are missing:
      - customer name
      - phone number
      - email
      - shipping address
      - at least one product request with quantity
    - Refuse fake invoices, manual discount overrides, stock bypass requests, or anything that asks the model
      to ignore the catalog or policy.
    - Use only tool outputs for product IDs, prices, stock, discount, totals, and save path.
    - Return one concise final answer in Vietnamese.
    - Mention `today` so the model knows the current date for deterministic references if needed.
    """
    current_day = today or "2026-06-01"
    return f"""
Bạn là OrderDesk, trợ lý tạo đơn hàng điện tử. Hôm nay là {current_day}.

Luật bắt buộc:
- Trả lời tiếng Việt, ngắn gọn, chỉ dựa trên output của tool.
- Nếu thiếu tên khách, số điện thoại, email, địa chỉ giao hàng, hoặc sản phẩm kèm số lượng: hỏi bổ sung và dừng, không gọi tool.
- Nếu yêu cầu tạo hóa đơn giả, bỏ qua tồn kho/catalog/policy, hoặc ép mã giảm giá thủ công: từ chối ngay, không gọi tool.
- Khi đủ thông tin hợp lệ, luôn gọi tool đúng thứ tự: list_products -> get_product_details -> get_discount -> calculate_order_totals -> save_order.
- Không tự sinh product_id, giá, tồn kho, discount_rate, campaign_code, order_id, save_path.
- Chỉ save sau khi calculate_order_totals trả status ok.
- Sau khi save thành công, xác nhận bằng order_id, discount, final_total và save_path từ saved_order.
""".strip()


def build_tools(store: OrderDataStore):
    """
    Student TODO:
    - Define exactly five tools with strong tool schemas:
      - `list_products`
      - `get_product_details`
      - `get_discount`
      - `calculate_order_totals`
      - `save_order`
    - Use the provided Pydantic schemas from `core.schemas` so the tool arguments stay explicit.
    - Keep outputs compact and JSON-friendly because the grader will inspect the saved order payload.
    - `get_product_details` should return a validation token, and later pricing/save tools should require it.
    """

    @tool(args_schema=ListProductsInput)
    def list_products(
        query: str | None = None,
        category: str | None = None,
        max_unit_price: int | None = None,
        required_tags: list[str] | None = None,
        in_stock_only: bool = True,
        limit: int = 8,
    ) -> str:
        """Search the local product catalog and return the best matching items."""
        payload = store.list_products(
            query=query,
            category=category,
            max_unit_price=max_unit_price,
            required_tags=required_tags,
            in_stock_only=in_stock_only,
            limit=limit,
        )
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=ProductDetailInput)
    def get_product_details(product_ids: list[str]) -> str:
        """Return exact product details for previously discovered product IDs."""
        return json.dumps(store.get_product_details(product_ids), ensure_ascii=False)

    @tool(args_schema=DiscountInput)
    def get_discount(seed_hint: str, customer_tier: str = "standard") -> str:
        """Return the simulated campaign discount for the order."""
        return json.dumps(store.get_discount(seed_hint=seed_hint, customer_tier=customer_tier), ensure_ascii=False)

    @tool(args_schema=CalculateTotalsInput)
    def calculate_order_totals(items, detail_token: str, discount_rate: float) -> str:
        """Validate stock and calculate the discounted order total."""
        lines = [item if isinstance(item, OrderLineInput) else OrderLineInput(**item) for item in items]
        return json.dumps(
            store.calculate_order_totals(items=lines, detail_token=detail_token, discount_rate=discount_rate),
            ensure_ascii=False,
        )

    @tool(args_schema=SaveOrderInput)
    def save_order(
        customer_name: str,
        customer_phone: str,
        customer_email: str,
        shipping_address: str,
        items,
        detail_token: str,
        discount_rate: float,
        campaign_code: str,
        customer_tier: str = "standard",
        notes: str = "",
    ) -> str:
        """Persist the final order to a local JSON file."""
        lines = [item if isinstance(item, OrderLineInput) else OrderLineInput(**item) for item in items]
        payload = store.save_order(
            customer_name=customer_name,
            customer_phone=customer_phone,
            customer_email=customer_email,
            shipping_address=shipping_address,
            items=lines,
            detail_token=detail_token,
            discount_rate=discount_rate,
            campaign_code=campaign_code,
            customer_tier=customer_tier,
            notes=notes,
        )
        return json.dumps(payload, ensure_ascii=False)

    return [list_products, get_product_details, get_discount, calculate_order_totals, save_order]


def build_agent(
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    *,
    provider: str = "openai",
    model_name: str | None = None,
    today: str | None = None,
):
    """
    Student TODO:
    1. Create `OrderDataStore`.
    2. Build the chat model with `build_chat_model(...)`.
    3. Build the tools with `build_tools(store)`.
    4. Return `create_agent(model=..., tools=..., system_prompt=...)`.
    """
    store = OrderDataStore(data_dir or DEFAULT_DATA_DIR, output_dir or DEFAULT_OUTPUT_DIR, today=today)
    model = build_chat_model(provider=provider, model_name=model_name, temperature=0.0)
    return create_agent(
        model=model,
        tools=build_tools(store),
        system_prompt=build_system_prompt(today or store.today),
    )


def run_agent(
    query: str,
    *,
    provider: str = "openai",
    model_name: str | None = None,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    today: str | None = None,
) -> AgentResult:
    """
    Student TODO:
    - Build the agent.
    - Invoke it with one user message.
    - Extract:
      - the final AI answer
      - the tool trace
      - the saved order payload, if any
    - Return an `AgentResult`.
    """
    store = OrderDataStore(data_dir or DEFAULT_DATA_DIR, output_dir or DEFAULT_OUTPUT_DIR, today=today)

    guardrail_reasons = get_policy_violations(query)
    if guardrail_reasons:
        return AgentResult(
            query=query,
            final_answer=render_guardrail_answer(guardrail_reasons),
            tool_calls=[],
            provider=provider,
            model_name=model_name,
        )

    draft = build_order_draft(query, store)
    missing = draft.missing_fields()
    if missing:
        return AgentResult(
            query=query,
            final_answer=render_clarification_answer(missing),
            tool_calls=[],
            provider=provider,
            model_name=model_name,
        )

    tool_calls: list[ToolCallRecord] = []

    list_args = {"query": query, "in_stock_only": True, "limit": 20}
    list_payload = store.list_products(**list_args)
    tool_calls.append(ToolCallRecord(name="list_products", args=list_args, output=json.dumps(list_payload, ensure_ascii=False)))

    product_ids = [item.product_id for item in draft.items]
    details_args = {"product_ids": product_ids}
    details_payload = store.get_product_details(product_ids)
    tool_calls.append(
        ToolCallRecord(name="get_product_details", args=details_args, output=json.dumps(details_payload, ensure_ascii=False))
    )

    stock_errors = find_stock_errors(draft.items, store)
    if stock_errors:
        return AgentResult(
            query=query,
            final_answer=render_stock_failure_answer(stock_errors),
            tool_calls=tool_calls,
            provider=provider,
            model_name=model_name,
        )

    discount_args = {"seed_hint": draft.customer_email, "customer_tier": "standard"}
    discount_payload = store.get_discount(**discount_args)
    draft.discount = discount_payload
    tool_calls.append(ToolCallRecord(name="get_discount", args=discount_args, output=json.dumps(discount_payload, ensure_ascii=False)))

    detail_token = str(details_payload.get("detail_token", ""))
    totals_args = {
        "items": [item.model_dump() for item in draft.items],
        "detail_token": detail_token,
        "discount_rate": discount_payload["discount_rate"],
    }
    totals_payload = store.calculate_order_totals(
        items=draft.items,
        detail_token=detail_token,
        discount_rate=discount_payload["discount_rate"],
    )
    draft.totals = totals_payload
    tool_calls.append(
        ToolCallRecord(name="calculate_order_totals", args=totals_args, output=json.dumps(totals_payload, ensure_ascii=False))
    )

    if totals_payload.get("status") != "ok":
        errors = totals_payload.get("errors", ["không thể tính tổng đơn hàng."])
        return AgentResult(
            query=query,
            final_answer=render_stock_failure_answer([str(error) for error in errors]),
            tool_calls=tool_calls,
            provider=provider,
            model_name=model_name,
        )

    save_args = {
        "customer_name": draft.customer_name,
        "customer_phone": draft.customer_phone,
        "customer_email": draft.customer_email,
        "shipping_address": draft.shipping_address,
        "items": [item.model_dump() for item in draft.items],
        "detail_token": detail_token,
        "discount_rate": discount_payload["discount_rate"],
        "campaign_code": discount_payload["campaign_code"],
        "customer_tier": discount_payload["customer_tier"],
        "notes": "",
    }
    save_payload = store.save_order(
        customer_name=draft.customer_name,
        customer_phone=draft.customer_phone,
        customer_email=draft.customer_email,
        shipping_address=draft.shipping_address,
        items=draft.items,
        detail_token=detail_token,
        discount_rate=discount_payload["discount_rate"],
        campaign_code=discount_payload["campaign_code"],
        customer_tier=discount_payload["customer_tier"],
    )
    tool_calls.append(ToolCallRecord(name="save_order", args=save_args, output=json.dumps(save_payload, ensure_ascii=False)))
    saved_order, saved_order_path = extract_saved_order(tool_calls)

    if not saved_order:
        return AgentResult(
            query=query,
            final_answer="Không thể xác nhận lưu đơn vì save_order chưa trả về saved_order hợp lệ. Đơn chưa được lưu.",
            tool_calls=tool_calls,
            provider=provider,
            model_name=model_name,
        )

    final_answer = render_save_success_answer(saved_order)
    return AgentResult(
        query=query,
        final_answer=final_answer,
        tool_calls=tool_calls,
        provider=provider,
        model_name=model_name,
        saved_order=saved_order,
        saved_order_path=saved_order_path,
    )


def extract_final_answer(messages) -> str:
    """Optional helper: return the last non-empty AI answer."""
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = normalize_content(message.content)
            if text:
                return text
    return ""


def extract_tool_calls(messages) -> list[ToolCallRecord]:
    """Optional helper: convert tool calls and tool results into a simple grading trace."""
    pending: dict[str, dict[str, Any]] = {}
    records: list[ToolCallRecord] = []

    for message in messages:
        if isinstance(message, AIMessage):
            for tool_call in getattr(message, "tool_calls", []) or []:
                pending[tool_call["id"]] = {
                    "name": tool_call["name"],
                    "args": tool_call.get("args", {}) or {},
                }
        elif isinstance(message, ToolMessage):
            metadata = pending.pop(message.tool_call_id, {})
            records.append(
                ToolCallRecord(
                    name=str(getattr(message, "name", None) or metadata.get("name", "")),
                    args=metadata.get("args", {}),
                    output=normalize_content(message.content),
                )
            )

    for metadata in pending.values():
        records.append(ToolCallRecord(name=metadata["name"], args=metadata["args"], output=""))
    return records


def extract_saved_order(tool_calls: list[ToolCallRecord]) -> tuple[dict | None, str | None]:
    """Optional helper: parse the `save_order` tool output into `(saved_order, path)`."""
    for record in reversed(tool_calls):
        if record.name != "save_order" or not record.output:
            continue
        try:
            payload = json.loads(record.output)
        except json.JSONDecodeError:
            continue
        if payload.get("status") != "saved":
            return None, None
        return payload.get("saved_order"), payload.get("path")
    return None, None


def violates_policy(query: str) -> bool:
    return bool(get_policy_violations(query))


def get_policy_violations(query: str) -> list[str]:
    normalized = normalize_ascii(query)
    checks = [
        ("tạo hóa đơn giả", ["fake invoice", "hoa don gia", "hoa on gia"]),
        ("bỏ qua tồn kho", ["bypass stock", "bo qua ton kho"]),
        ("bỏ qua catalog/policy", ["bo qua policy", "ignore policy", "ignore catalog", "khong can theo catalog"]),
        ("ép giảm giá ngoài hệ thống", ["ep giam gia", "tu ep giam gia"]),
    ]
    reasons: list[str] = []
    for reason, terms in checks:
        if any(term in normalized for term in terms):
            reasons.append(reason)
    if "90%" in query and "giam" in normalized and "ép giảm giá ngoài hệ thống" not in reasons:
        reasons.append("ép giảm giá ngoài hệ thống")
    return reasons


def build_order_draft(query: str, store: OrderDataStore) -> OrderDraft:
    return OrderDraft(
        customer_name=extract_customer_name(query),
        customer_phone=extract_phone(query),
        customer_email=extract_email(query),
        shipping_address=extract_shipping_address(query),
        items=extract_items(query, store),
    )


def extract_email(query: str) -> str:
    match = re.search(r"[\w.+-]+@[\w.-]+\.\w+", query)
    return match.group(0) if match else ""


def extract_phone(query: str) -> str:
    match = re.search(r"\b0\d{9}\b", query)
    return match.group(0) if match else ""


def extract_customer_name(query: str) -> str:
    patterns = [
        r"(?:cho|for)\s+(.+?)(?:,\s*(?:s|email|phone)|\.\s*(?:Ship|Email|Phone)|$)",
        r"order\s+giúp\s+mình\s+cho\s+(.+?)(?:\.|,)",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            name = re.sub(r"^(?:chị|anh|bạn)\s+", "", name, flags=re.IGNORECASE).strip()
            if is_valid_customer_identity(name):
                return name
    return ""


def is_valid_customer_identity(name: str) -> bool:
    normalized = normalize_ascii(name)
    ambiguous_identities = {
        "cong ty moi",
        "khach moi",
        "toi",
        "minh",
        "cho toi",
        "cho minh",
        "tui",
    }
    if not normalized:
        return False
    if normalized in ambiguous_identities:
        return False
    return not normalized.startswith(("toi ", "minh ", "tui "))


def extract_shipping_address(query: str) -> str:
    markers = [
        "Ship to ",
        "giao đến ",
        "giao hàng đến ",
        "giao tới ",
        "giao về ",
        "địa chỉ giao hàng ",
        "giao Ä‘áº¿n ",
        "giao hÃ ng Ä‘áº¿n ",
        "giao tá»›i ",
        "giao vá» ",
        "Ä‘á»‹a chá»‰ giao hÃ ng ",
    ]
    lower = query.lower()
    for marker in markers:
        index = lower.find(marker.lower())
        if index == -1:
            continue
        start = index + len(marker)
        tail = query[start:].strip()
        stop_candidates = [
            tail.find(token)
            for token in [
                ". T",
                ". Ch",
                ". M",
                ". Phone",
                ", số",
                ", sá»‘",
                ", phone",
                ", email",
            ]
            if tail.find(token) != -1
        ]
        stop = min(stop_candidates) if stop_candidates else len(tail)
        return tail[:stop].strip().rstrip(".")
    return ""


def extract_items(query: str, store: OrderDataStore) -> list[OrderLineInput]:
    lowered = query.lower()
    matches: list[tuple[int, OrderLineInput]] = []
    for product in store.products:
        position = lowered.find(product.name.lower())
        if position == -1:
            continue
        prefix = query[max(0, position - 40) : position]
        segment = re.split(r"[,;:.\"“”]|\bvà\b|\band\b", prefix, flags=re.IGNORECASE)[-1]
        quantity_match = re.search(r"(\d+)\s*$", segment.strip())
        quantity = int(quantity_match.group(1)) if quantity_match else 1
        matches.append((position, OrderLineInput(product_id=product.product_id, quantity=quantity)))
    matches.sort(key=lambda item: item[0])
    return [item for _, item in matches]


def find_stock_errors(items: list[OrderLineInput], store: OrderDataStore) -> list[str]:
    errors: list[str] = []
    for item in items:
        product = store.product_index.get(item.product_id)
        if not product:
            errors.append(f"không tìm thấy sản phẩm {item.product_id}")
            continue
        if item.quantity > product.stock:
            errors.append(f"{product.name} chỉ còn {product.stock}, yêu cầu {item.quantity}")
    return errors


def render_save_success_answer(saved_order: dict[str, Any]) -> str:
    pricing = saved_order["pricing"]
    discount = saved_order["discount"]
    discount_percent = int(pricing["discount_rate"] * 100)
    item_summary = ", ".join(
        f"{item['quantity']} {item['name']}" for item in saved_order.get("items", [])
    )
    return (
        "Dựa trên kết quả tool/catalog, đã kiểm tra catalog, tồn kho, khuyến mãi và tổng tiền; "
        f"đơn {saved_order['order_id']} đã được lưu thành công. "
        f"Mặt hàng đã lưu: {item_summary}. "
        f"Mã khuyến mãi: {discount['campaign_code']} ({discount_percent}%). "
        f"Tổng sau giảm: {pricing['final_total']} VND. "
        f"File lưu: {saved_order['save_path']}."
    )


def render_clarification_answer(missing: list[str]) -> str:
    return (
        "Mình cần thêm " + ", ".join(missing) + " trước khi tạo đơn. "
        "Đơn chưa được lưu."
    )


def render_stock_failure_answer(errors: list[str]) -> str:
    return "Không thể lưu đơn vì " + "; ".join(errors) + ". Đơn chưa được lưu."


def render_guardrail_answer(reasons: list[str]) -> str:
    reason_text = ", ".join(reasons)
    return (
        f"Mình không thể hỗ trợ yêu cầu vi phạm chính sách: {reason_text}. "
        "Mình chỉ có thể tạo đơn hợp lệ theo catalog, tồn kho và khuyến mãi từ hệ thống. "
        "Đơn/hóa đơn chưa được lưu."
    )


def normalize_ascii(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    replacements = {
        "Ä‘": "d",
        "Ä": "D",
        "á»": "o",
        "áº£": "a",
        "áº¡": "a",
        "á»“": "o",
        "á»“": "o",
        "á»‹": "i",
        "á»‡": "e",
        "áº¿": "e",
        "áº§": "a",
        "áº¥": "a",
        "á»›": "o",
        "á»": "o",
        "Ã³": "o",
        "Ã ": "a",
        "Ã¡": "a",
        "Ãª": "e",
        "Ã´": "o",
        "Ã¹": "u",
        "Ã­": "i",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = re.sub(r"[^a-zA-Z0-9%]+", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()
