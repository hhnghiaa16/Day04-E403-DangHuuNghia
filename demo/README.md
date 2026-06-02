# OrderDesk Compare Demo

Local web UI to compare the weak baseline agent with the improved `src` agent.

## Run

From the repository root:

```bash
python demo/app.py --host 127.0.0.1 --port 8765
```

Open:

```text
http://127.0.0.1:8765
```

## What It Shows

- Final answer from each agent
- Tool trace sequence
- `saved_order` payload
- Deterministic grader score and feedback
- Score delta between baseline and improved agent

## Controls

- `Benchmark case`: choose a case from `data/graded_cases.json`.
- `Query`: edit or enter a custom request.
- `Provider`: provider name passed to each agent, default `openai`.
- `Run baseline`: run `simple_solution.agent.graph`.
- `Run LLM judge`: include the LLM judge score. This uses real API tokens for cloud providers.
- `Show JSON`: show or hide raw `saved_order` JSON.

## Recommended Usage

For fast demos without rate-limit pressure:

```text
Run baseline: off if baseline hits API/rate-limit errors
Run LLM judge: off
Provider: openai
```

The improved `src` agent is mostly deterministic in `run_agent`, so it can be compared without spending OpenAI tokens unless `Run LLM judge` is enabled.

## Notes

- The demo runs agents sequentially to avoid artifact read/write races.
- Generated demo orders are written under `artifacts/demo_orders/`.
- If `Run LLM judge` is enabled with `openai` or `google`, the judge uses real API keys and tokens.
- If the baseline fails because of missing dependencies or rate limits, the improved agent column can still run.

## Stop The Server

If the server is running in the current terminal, press:

```text
Ctrl+C
```

If it was started in the background on Windows, find and stop the Python process:

```powershell
Get-Process python
Stop-Process -Id <PID>
```
