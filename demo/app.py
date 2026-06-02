from __future__ import annotations

import argparse
import importlib
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from grade.scoring import coerce_result, grade_result, load_cases

CASES_PATH = ROOT_DIR / "data" / "graded_cases.json"
DEMO_OUTPUT_DIR = ROOT_DIR / "artifacts" / "demo_orders"


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>OrderDesk Agent Compare</title>
  <style>
    :root {
      --bg: #f7f5ef;
      --ink: #1d2528;
      --muted: #687276;
      --line: #d8d2c3;
      --panel: #fffdf8;
      --green: #1f7a4d;
      --red: #b13b2e;
      --amber: #b26b12;
      --blue: #2f5d8c;
      --shadow: 0 18px 42px rgba(36, 41, 44, 0.12);
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      background:
        linear-gradient(135deg, rgba(47, 93, 140, 0.08), transparent 34%),
        linear-gradient(225deg, rgba(31, 122, 77, 0.08), transparent 30%),
        var(--bg);
      color: var(--ink);
      font-family: ui-sans-serif, "Segoe UI", Verdana, sans-serif;
    }

    header {
      padding: 28px clamp(18px, 4vw, 44px) 16px;
      border-bottom: 1px solid var(--line);
    }

    h1 {
      margin: 0;
      font-size: clamp(28px, 4vw, 46px);
      letter-spacing: 0;
      line-height: 1.05;
    }

    .sub {
      margin: 10px 0 0;
      color: var(--muted);
      max-width: 880px;
      line-height: 1.55;
    }

    main {
      padding: 22px clamp(18px, 4vw, 44px) 36px;
      display: grid;
      gap: 18px;
    }

    .control {
      background: rgba(255, 253, 248, 0.86);
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
      padding: 18px;
      display: grid;
      gap: 14px;
      border-radius: 8px;
    }

    .row {
      display: grid;
      grid-template-columns: minmax(220px, 360px) 1fr;
      gap: 14px;
      align-items: end;
    }

    label {
      display: grid;
      gap: 7px;
      font-size: 13px;
      color: var(--muted);
      font-weight: 650;
    }

    select, textarea, input[type="text"] {
      width: 100%;
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--ink);
      border-radius: 6px;
      padding: 10px 11px;
      font: inherit;
    }

    textarea {
      min-height: 112px;
      resize: vertical;
      line-height: 1.45;
    }

    .toggles {
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
      align-items: center;
    }

    .check {
      display: inline-flex;
      gap: 8px;
      align-items: center;
      color: var(--ink);
      font-size: 14px;
      font-weight: 600;
    }

    button {
      border: 0;
      background: var(--ink);
      color: white;
      min-height: 42px;
      padding: 0 18px;
      border-radius: 6px;
      font-weight: 750;
      cursor: pointer;
    }

    button:disabled {
      opacity: 0.58;
      cursor: wait;
    }

    .summary {
      display: grid;
      grid-template-columns: repeat(4, minmax(140px, 1fr));
      gap: 12px;
    }

    .metric, .agent {
      background: rgba(255, 253, 248, 0.9);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }

    .metric b {
      display: block;
      font-size: 24px;
      margin-top: 4px;
    }

    .compare {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }

    .agent h2 {
      margin: 0 0 8px;
      font-size: 21px;
    }

    .badge {
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      padding: 0 8px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 800;
      background: #ebe5d6;
      color: var(--ink);
    }

    .badge.pass { background: rgba(31, 122, 77, 0.14); color: var(--green); }
    .badge.fail { background: rgba(177, 59, 46, 0.14); color: var(--red); }
    .badge.warn { background: rgba(178, 107, 18, 0.16); color: var(--amber); }

    .section {
      border-top: 1px solid var(--line);
      margin-top: 12px;
      padding-top: 12px;
    }

    .answer {
      line-height: 1.55;
      white-space: pre-wrap;
    }

    .tools {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }

    .tool {
      border: 1px solid var(--line);
      background: #f3efe5;
      padding: 6px 8px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 750;
    }

    pre {
      max-height: 360px;
      overflow: auto;
      background: #1e2528;
      color: #f8f2e6;
      padding: 12px;
      border-radius: 6px;
      font-size: 12px;
      line-height: 1.45;
    }

    .feedback {
      color: var(--red);
      margin: 8px 0 0;
      padding-left: 18px;
      line-height: 1.4;
    }

    .error {
      color: var(--red);
      font-weight: 700;
      line-height: 1.45;
    }

    @media (max-width: 920px) {
      .row, .compare, .summary {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <header>
    <h1>OrderDesk Agent Compare</h1>
    <p class="sub">Compare the weak baseline with the improved agent across final answer, tool trace, saved order payload, and deterministic grader feedback.</p>
  </header>
  <main>
    <section class="control">
      <div class="row">
        <label>Benchmark case
          <select id="caseSelect"></select>
        </label>
        <label>Provider
          <input id="provider" type="text" value="openai" />
        </label>
      </div>
      <label>Query
        <textarea id="query"></textarea>
      </label>
      <div class="toggles">
        <label class="check"><input id="runBaseline" type="checkbox" checked /> Run baseline</label>
        <label class="check"><input id="runJudge" type="checkbox" /> Run LLM judge</label>
        <label class="check"><input id="showJson" type="checkbox" checked /> Show JSON</label>
        <button id="compareBtn">Compare</button>
      </div>
    </section>

    <section class="summary" id="summary"></section>
    <section class="compare" id="compare"></section>
  </main>

  <script>
    const state = { cases: [] };
    const caseSelect = document.querySelector("#caseSelect");
    const queryBox = document.querySelector("#query");
    const compareBtn = document.querySelector("#compareBtn");
    const summary = document.querySelector("#summary");
    const compare = document.querySelector("#compare");

    function scoreText(result) {
      if (!result || !result.score) return "n/a";
      return `${result.score.score}/${result.score.max_score}`;
    }

    function overallClass(result) {
      if (result?.error) return "fail";
      if (!result?.score) return "warn";
      return result.score.feedback.length ? "warn" : "pass";
    }

    function renderSummary(data) {
      const baselineScore = scoreText(data.baseline);
      const improvedScore = scoreText(data.improved);
      const delta = data.baseline?.score && data.improved?.score
        ? (data.improved.score.score - data.baseline.score.score).toFixed(2)
        : "n/a";
      summary.innerHTML = `
        <div class="metric"><span>Case</span><b>${data.case_id || "custom"}</b></div>
        <div class="metric"><span>Baseline</span><b>${baselineScore}</b></div>
        <div class="metric"><span>Improved</span><b>${improvedScore}</b></div>
        <div class="metric"><span>Delta</span><b>${delta}</b></div>
      `;
    }

    function renderAgent(title, result) {
      if (!result) {
        return `<article class="agent"><h2>${title}</h2><span class="badge warn">skipped</span></article>`;
      }
      if (result.error) {
        return `
          <article class="agent">
            <h2>${title}</h2><span class="badge fail">error</span>
            <div class="section error">${escapeHtml(result.error)}</div>
          </article>
        `;
      }
      const tools = result.tool_calls.map(t => `<span class="tool">${escapeHtml(t.name)}</span>`).join("");
      const feedback = result.score?.feedback?.length
        ? `<ul class="feedback">${result.score.feedback.map(item => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`
        : `<span class="badge pass">no deterministic feedback</span>`;
      const jsonBlock = document.querySelector("#showJson").checked
        ? `<div class="section"><b>saved_order</b><pre>${escapeHtml(JSON.stringify(result.saved_order, null, 2))}</pre></div>`
        : "";
      return `
        <article class="agent">
          <h2>${title}</h2>
          <span class="badge ${overallClass(result)}">${scoreText(result)}</span>
          <div class="section"><b>Final answer</b><div class="answer">${escapeHtml(result.final_answer || "")}</div></div>
          <div class="section"><b>Tool trace</b><div class="tools">${tools || "<span class='badge'>no tools</span>"}</div></div>
          <div class="section"><b>Feedback</b>${feedback}</div>
          ${jsonBlock}
        </article>
      `;
    }

    function renderCompare(data) {
      renderSummary(data);
      compare.innerHTML = renderAgent("Baseline Agent", data.baseline) + renderAgent("Improved Agent", data.improved);
    }

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    async function loadCases() {
      const response = await fetch("/api/cases");
      state.cases = await response.json();
      caseSelect.innerHTML = `<option value="">Custom query</option>` + state.cases
        .map(item => `<option value="${escapeHtml(item.id)}">${escapeHtml(item.id)} (${escapeHtml(item.category)})</option>`)
        .join("");
      if (state.cases.length) {
        caseSelect.value = state.cases[0].id;
        queryBox.value = state.cases[0].query;
      }
    }

    caseSelect.addEventListener("change", () => {
      const selected = state.cases.find(item => item.id === caseSelect.value);
      if (selected) queryBox.value = selected.query;
    });

    compareBtn.addEventListener("click", async () => {
      compareBtn.disabled = true;
      compare.innerHTML = `<article class="agent"><h2>Running</h2><div class="section">Comparing agents...</div></article>`;
      try {
        const response = await fetch("/api/compare", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            case_id: caseSelect.value || null,
            query: queryBox.value,
            provider: document.querySelector("#provider").value || "openai",
            run_baseline: document.querySelector("#runBaseline").checked,
            run_judge: document.querySelector("#runJudge").checked
          })
        });
        renderCompare(await response.json());
      } catch (error) {
        compare.innerHTML = `<article class="agent"><h2>Error</h2><div class="section error">${escapeHtml(error.message)}</div></article>`;
      } finally {
        compareBtn.disabled = false;
      }
    });

    document.querySelector("#showJson").addEventListener("change", () => {
      compareBtn.click();
    });

    loadCases();
  </script>
</body>
</html>
"""


class DemoHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/" or self.path.startswith("/?"):
            self.write_response(200, HTML, "text/html; charset=utf-8")
            return
        if self.path == "/api/cases":
            cases = [
                {
                    "id": case["id"],
                    "category": case.get("category", ""),
                    "query": case["query"],
                    "required_tools": case["expected"].get("required_tools", []),
                }
                for case in load_cases(CASES_PATH)
            ]
            self.write_json(200, cases)
            return
        self.write_json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        if self.path != "/api/compare":
            self.write_json(404, {"error": "Not found"})
            return

        try:
            payload = self.read_json()
            result = compare_agents(payload)
        except Exception as exc:
            self.write_json(500, {"error": str(exc)})
            return
        self.write_json(200, result)

    def read_json(self) -> dict[str, Any]:
        size = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(size).decode("utf-8")
        return json.loads(raw) if raw else {}

    def write_json(self, status: int, payload: Any) -> None:
        self.write_response(status, json.dumps(payload, ensure_ascii=False, indent=2), "application/json; charset=utf-8")

    def write_response(self, status: int, body: str, content_type: str) -> None:
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[demo] {self.address_string()} - {format % args}")


def compare_agents(payload: dict[str, Any]) -> dict[str, Any]:
    cases = load_cases(CASES_PATH)
    case = find_case(cases, payload.get("case_id"), payload.get("query", ""))
    query = str(payload.get("query") or case["query"])
    provider = str(payload.get("provider") or "openai")
    run_judge = bool(payload.get("run_judge", False))

    response: dict[str, Any] = {
        "case_id": case["id"] if case else None,
        "query": query,
        "baseline": None,
        "improved": None,
        "expected": case.get("expected") if case else None,
    }

    if payload.get("run_baseline", True):
        response["baseline"] = run_and_score(
            module_name="simple_solution.agent.graph",
            query=query,
            case=case,
            provider=provider,
            judge_provider=provider if run_judge else None,
            output_dir=DEMO_OUTPUT_DIR / "baseline",
        )

    response["improved"] = run_and_score(
        module_name="src.agent.graph",
        query=query,
        case=case,
        provider=provider,
        judge_provider=provider if run_judge else None,
        output_dir=DEMO_OUTPUT_DIR / "improved",
    )
    return response


def find_case(cases: list[dict[str, Any]], case_id: Any, query: str) -> dict[str, Any] | None:
    if case_id:
        for case in cases:
            if case["id"] == case_id:
                return case
    for case in cases:
        if case["query"] == query:
            return case
    return None


def run_and_score(
    *,
    module_name: str,
    query: str,
    case: dict[str, Any] | None,
    provider: str,
    judge_provider: str | None,
    output_dir: Path,
) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
        raw = module.run_agent(query, provider=provider, output_dir=output_dir, today="2026-06-01")
        result = coerce_result(raw, query=query, provider=provider, model_name=None)
        score = None
        if case:
            score = grade_result(result, case, judge_provider=judge_provider). __dict__
        result["score"] = score
        return result
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "score": None, "tool_calls": [], "final_answer": "", "saved_order": None}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the OrderDesk agent comparison demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    print(f"OrderDesk compare demo: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping demo server.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
