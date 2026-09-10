# Windows live Ollama validation: passed

Evidence: user-supplied `report.json`, dated 2026-09-10 05:42:22 UTC.
The report was checked for internal consistency; this was not an independent remote
execution on the PC. The unrelated installed-model inventory is omitted from the
committed copy; selected-model details and validation events are preserved.

| Check | Reported result |
| --- | --- |
| Model | `ornith-9b-32k:latest` |
| Runtime | Windows, Python 3.10.11, Ollama 0.33.3 |
| Accepted LLM decisions | 3/3; no errors or fallbacks |
| Strategy ticks | 0, 20, 40 |
| Simulation transitions | 60 |
| Memory save/load | Exact round-trip passed |
| First request | 50.47 seconds |
| Requests 2 and 3 | 1.63 and 1.65 seconds (mean 1.64) |
| Prompt / output tokens | 1,622 / 167 |
| Ollama loaded size / VRAM | Both 5,260,202,475 bytes (4.90 GiB) |
| Active context length | 2,048 tokens |

The loaded-model allocation is reported entirely in VRAM. This is Ollama's
allocation report, not a GPU-utilization trace. The first request's delay could
include model loading, but the report does not separate loading from inference.
The default 10-second demo timeout would not cover this first request. Use:

```bash
python -m experiments.strategy_runner --render --planner ollama --model ornith-9b-32k:latest --llm-timeout 60 --max-calls 10
```

The current planner is synchronous, so the viewer may pause while requests run.
A model already loaded in Ollama may respond faster; three requests do not establish
a stable latency distribution or a guarantee for future calls.

All three decisions selected `explore`. Passing confirms protocol, parsing, event
gating, local control integration and memory persistence in this diagnostic world.
It does not show that LLM strategy improves fitness, that reported confidence is
calibrated, or that every explanation is grounded. This report does not contain the
full input context for auditing individual explanations. Multi-creature load, the
rendered Windows viewer, PPO+LLM interaction and matched live-LLM ablations remain
separate checks. No change to PPO or memory effectiveness claims is warranted.
