# Strategy reliability after the Windows population log

The user-supplied aggregate log contained 40 attempts across several executions:
31 accepted decisions, nine parse/validation fallbacks, and every attempt at
creature tick zero for founders 0–9. It was not a single 40-call run. Of the accepted
decisions, 29 were `explore` and two were `seek_food`. Explanations repeatedly
interpreted `done` episode endings as successes. Generic error types could not
explain the rejected outputs. These observations motivated this targeted update.

## Shared call scheduling

`brains/scheduling.py` adds a deterministic admission step in the policy runner,
before any creature selects its action. It receives local observations, controllers
and the world tick; it does not give the planner hidden world information.

- The session budget is split across requested episodes. Default 10 calls / two
  episodes reserves five each. Integer remainders go to the earlier episodes.
- Minimum gap between calls is `max(20, ceil(episode_horizon / episode_quota))`
  world ticks. For 200 steps and quota five, this is 40 ticks. No burst retries.
- Eligible controllers have a meaningful event/refresh request or a pending
  request previously deferred by scheduling. Per-creature attempt cooldown still applies.
- Choose the least-served eligible memory owner, then longest waiting, then a
  seeded hash tie-break. Attempt counts carry across episodes in that session.
  Newborns can join; dead controllers leave the candidate set.
- At most one selected creature can consume a slot at a scheduling tick. Changing
  dictionary/action processing order does not change that selected owner.
- Failed calls consume both session and episode budgets. Unused quota is not
  transferred across episodes. Short episodes, no eligible controllers, or budgets
  smaller than episode count may mean fewer calls or zero quota in some episodes.

The quota is a ceiling, not a promise of an exact number of calls. Fairness is
among currently eligible creatures in one session; ten calls cannot cover every
creature in a population of hundreds. Global call counts reset on a new run.
Defaults remain synchronous; the viewer can pause while the model is answering.

`scheduled_fallback` means waiting for admission; `episode_budget_fallback` means
that episode's allocation is exhausted; `budget_fallback` means the whole session's
budget is exhausted. These are expected control paths, not provider errors.

## Memory meaning and output diagnostics

Disk memory schema and the 111-value PPO feature vector are preserved. Old `done`
records still load unchanged, but the LLM sees `end_reason: ended_unspecified`.
New transitions record death versus time limit explicitly. Interruption remains
separate. None is labeled successful merely because the episode ended.

The context names low-level movement values `UP`, `DOWN`, `LEFT`, `RIGHT`, `STAY`.
Static instructions explain energy fractions, local resource counts, the meaning
of episode endings, and that aggregate reward does not establish goal-specific
causation. Instructions can reduce ambiguity; they do not guarantee grounded reasoning.

Ollama now receives the allowed goal/confidence/reason JSON schema as `format`,
following [the official structured-output API](https://docs.ollama.com/capabilities/structured-outputs).
Local validation remains strict. Reason length remains capped at 240 characters;
the prompt asks for preferably under 120. Output allowance increases from 96 to
160 tokens to leave room for valid JSON. No automatic second request is made.

Attempt records include prompt version, actual bounded local context, known token
counts, a proposal when valid JSON parses, and explicit error codes such as
`unexpected_fields`, `reason_length_or_type`, `malformed_json`, `incomplete_generation`
or `low_confidence`. Invalid content is captured in at most a 512-character excerpt
with full-content length/hash; output-limit metadata distinguishes a suspected
truncation from an established validation error. A parsed but low-confidence
proposal stays visible even when the controller uses a rule fallback.

This improves diagnostics, not the acceptance rules. A response is never repaired
into an invented success. Unsupported structured-output providers may fall back;
the actual selected model must be revalidated after this update.

## Logs and rerun on the PC

```powershell
git pull
python -m experiments.strategy_runner --render --planner ollama --model ornith-9b-32k:latest --llm-timeout 60 --max-calls 10
```

Send these two files from the output folder afterward:

```text
G:\neuro-genesis-simulator\runs\strategy\latest_strategy.jsonl
G:\neuro-genesis-simulator\runs\strategy\summary.json
```

Every new strategy and creature row has a run UUID, numeric session, episode index
and seed. `latest_strategy.jsonl` is reset at run start; the aggregate `strategy.jsonl`
remains append-only. New runs also have their own `sessions/<run_id>/strategy.jsonl`
and summary. Buffered event streaming preserves events beyond the controller's
128-event viewer history. Summary counters distinguish attempts, accepted LLM
decisions, fallback categories and each episode's quota; they are not all called
successful LLM calls. Use one process per output directory.

## Evidence and limits

The committed [scheduler smoke check](results/strategy-reliability/scheduler-check.json)
uses a deliberately fake transport, 12 founders, reproduction, two 200-step episodes
and ten total calls. It verifies distribution and bookkeeping, not LLM output quality.
Tests cover action-order independence, pending requests, failed-call pacing,
episode reservations, newborn/dead candidates, legacy saves, true termination labels,
bounded diagnostics and run isolation. Previous PPO scores are historical and
unchanged; PPO weights, genome logic and NBB were not modified.

The earlier Windows report demonstrated original live connectivity. A new user-PC
run is still required to measure acceptance rate and behavior with this version.
