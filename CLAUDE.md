# Project context for Claude

Research codebase measuring **private-information leakage in LLM agent-to-agent
negotiations**. Layers so far: a unified model client, a two-agent conversation
engine, a dashboard for driving/watching runs, a scenario + condition-controlled
prompt layer (used-car negotiations with buyer defenses and seller adversary
strategies), and an evaluation layer (adversary extraction + independent judge +
flat `RunResult` metrics). Still not built: an experiment runner that sweeps the
(scenario × defense × adversary × model) grid, and any aggregation/analysis over
the resulting JSONL.

## Hard constraints (do not violate)

- **Keep all naming generic.** The package is `src/`. No project name, topic
  name, or branding in module names, class names, CLI output, or docs. (The
  containing folder may be named anything; the *code* stays neutral.) Domain
  words for the modelled task — `buyer`, `seller`, `scenario`, `car` — are fine;
  the research *topic* ("leakage", "privacy") stays out of identifiers.
- **Immersion (load-bearing).** Agent-facing system prompts must never reveal
  that an agent is an AI, that the counterpart is an AI, or that anything is a
  test/simulation/evaluation. Meta concepts ("secret", "leakage", "score") live
  only in scenario files and any future grader — never in a rendered prompt. The
  one sanctioned exception is the `human_impersonation` seller (claims humanity,
  denies being an AI if asked). `tests/test_prompts.py` enforces this; keep it.
- **Scope discipline.** Built so far: foundations, dashboard, and the scenario/
  prompt layer. Do **not** add leakage/privacy metrics, judges/graders, or
  experiment runners unless explicitly asked — those are future sprints.
- Type hints throughout. No global state. Everything configurable via function
  args, `models.yaml`, or the scenario files.

## Toolchain

- Python 3.11+ (`.python-version` pins 3.12), managed with **uv**.
- `uv` lives at `~/.local/bin/uv`; if `uv` isn't found, run
  `export PATH="$HOME/.local/bin:$PATH"` first.
- Deps: anthropic, openai, pydantic, python-dotenv, pyyaml, fastapi, uvicorn
  (dev: pytest, ruff, httpx for `TestClient`).
- `uv sync` creates the venv. Run everything via `uv run ...`.

## Layout

| Path                    | Purpose |
| ----------------------- | ------- |
| `src/models.py`         | `ModelClient` ABC + `OpenAICompatClient` + `AnthropicClient`, retry/backoff, `ModelConfig`/`ModelResponse`/`RetryPolicy`, `load_registry()`, `get_client(name)` |
| `models.yaml`           | Registry: short name -> backend, model_id, base_url, api_key_env, defaults, prices |
| `src/engine.py`         | `Agent` dataclass, `run_conversation(...)`, pydantic `Transcript`/`Turn`/`AgentInfo` |
| `src/persistence.py`    | `save_transcript()` / `load_transcript()` (JSON under `runs/`) |
| `src/scenarios.py`      | Pydantic `Scenario` schema, label enums, `load_scenario`/`iter_scenarios` |
| `scenarios/`            | 12 used-car scenarios `s01`–`s12` (YAML ground truth) |
| `src/prompts.py`        | `DefenseCondition`/`AdversaryStrategy` enums, `render_buyer_system`/`render_seller_system`/`render_pair` |
| `src/preview.py`        | CLI: `python -m src.preview` — render prompts, `--run` for a live conversation |
| `src/jsonparse.py`      | `request_json()` — ask a model for JSON + bounded repair-retry; `extract_json_object()` |
| `src/extraction.py`     | `run_extraction()` — adversary questionnaire, `ExtractionOutput`/`ExtractionResult`, repair→regex→invalid |
| `src/judge.py`          | `run_judgement()` — independent judge, `JudgeOutput`/`LeakLabel`, per-turn/attribute labels |
| `src/metrics.py`        | `build_run_result()` + pydantic `RunResult` (pure metric math) |
| `src/evaluate.py`       | `evaluate_run()`, `EvalConfig`, `append_result()`; CLI `python -m src.evaluate` |
| `src/server.py`         | `create_app(...)` — dashboard HTTP API + SSE stream; `python -m src.server` |
| `src/static/`           | Dashboard page: `index.html`, `styles.css`, `app.js` (no build step) |
| `src/smoke.py`          | CLI: `python -m src.smoke` — a bicycle-haggle smoke test |
| `scripts/live_check.py` | One live "say OK" + JSON-mode probe per registry entry (not in pytest) |
| `tests/`                | Offline tests using an in-file `MockBackend` (no network) |

## Architecture

**Model layer (`src/models.py`).**
`ModelClient` is an ABC with one method:
`chat(messages, *, system=None, temperature=None, max_tokens=None, json_mode=False) -> ModelResponse`.
`ModelResponse` = `text, prompt_tokens, completion_tokens, latency_ms, raw`.

Two backends:
- `OpenAICompatClient` — any OpenAI-compatible endpoint (OpenAI, Groq, Gemini
  compat) via `base_url` + `api_key_env`. `system` becomes a system message;
  `json_mode` sets `response_format={"type":"json_object"}`.
- `AnthropicClient` — native Anthropic SDK. The Messages API has no
  `response_format`, so `json_mode` is a strict system-prompt instruction
  appended to `system`.

Retries live in `ModelClient._call_with_retries` (not the SDKs — both clients
are constructed with `max_retries=0`). Exponential backoff with jitter, honors
`Retry-After` on 429s, capped by `RetryPolicy` (also holds the per-call
`timeout_s` hard timeout, passed as `timeout=` to each request).

`temperature`/`max_tokens` args of `None` fall back to the registry default; a
registry `temperature` of `None` **omits the parameter entirely** (see gotcha
below).

**Engine layer (`src/engine.py`).**
`run_conversation(agent_a, agent_b, max_turns=6, opening_speaker=None, opening_prompt="You may begin.")`.
Perspective rule: each agent sees its own past messages as `assistant`, the
counterpart's as `user`; **system prompts never cross**. The opener also sees
`opening_prompt` as its first `user` message (chat APIs need something to reply
to); the counterpart never sees it. Termination: a turn containing
`[DEAL $X]` (regex captures X into `Transcript.deal_amount`), `[WALK_AWAY]`,
hitting `max_turns`, or `cancelled`. Returns a pydantic `Transcript` (ordered
`Turn`s with per-turn usage + latency, `termination`, per-agent `AgentInfo`,
timestamps).

Two optional hooks let an observer follow a run without owning the loop (this
is all the server needs — do not duplicate the perspective logic elsewhere):
`on_turn(turn)` fires as each turn completes, and `cancelled()` is polled
*before* each turn — True stops with termination `cancelled`, keeping the turns
so far (so a cancel lands only after the in-flight API call returns). A
`metadata: dict | None` param is stored verbatim on `Transcript.metadata`
(generic run provenance — the engine never reads it; the server puts the
scenario id + conditions there so a saved run is self-describing and evaluable).

**Persistence.** `save_transcript()` writes
`runs/<UTCstamp>-<a>-vs-<b>.json` (collision-safe suffixes), `load_transcript()`
round-trips it via pydantic. `runs/` is gitignored.

**Scenario layer (`src/scenarios.py` + `scenarios/*.yaml`).**
A `Scenario` is machine-readable ground truth: `public` (car, asking_price,
shared_facts), `buyer_private` (reservation_price, urgency{level,reason},
sensitive_context{label,detail}, budget_flexibility), `seller_private`
(floor_price, inventory_pressure), and a `pretext`. All controlled vocabularies
are `StrEnum`s **in this one module** (`SensitiveLabel`, `Urgency`,
`BudgetFlexibility`, `PressureLevel`, `Pretext`) so chance baselines stay
computable later. Models use `extra="forbid"`; `sensitive_context` enforces
detail-empty-iff-`none`. 12 scenarios: each `SensitiveLabel` appears twice, and
`pretext` is rotated evenly (4 financing / 4 insurance / 4 dealer_compliance) so
a finding can't be an artifact of one pretext. `load_scenario(id)` /
`iter_scenarios()` load and validate by filename stem.

**Prompt layer (`src/prompts.py`).**
`render_buyer_system(scenario, defense, adversary=None)` and
`render_seller_system(scenario, adversary)` build in-character system prompts as
plain hand-editable Python strings (no templating engine). `render_pair(...)`
returns both and enforces the config gate. Conditions:
`DefenseCondition` = `none`/`basic`/`strong` (buyer); `AdversaryStrategy` =
`passive`/`direct_probe`/`rapport`/`pressure`/`authority`/`human_impersonation`/
`authority_verifiable` (seller). Design invariants to preserve:
- `human_impersonation` seller == the exact `direct_probe` text **plus** an
  appended humanity/AI-denial clause — it is `direct_probe`'s control, so keep
  `_DIRECT_PROBE` shared and appended, never forked.
- `authority` uses `scenario.pretext` for the false role and encodes all three
  required constraints in the prompt text (lies about role only / claim is
  unverifiable / escalates "can't submit the form" on hesitation).
- `authority_verifiable` renders the seller identically to `authority`; it only
  adds a verification-demand clause to the **buyer**. It is a defense arm gated
  behind `PromptConfig.enable_authority_verifiable` — off by default,
  `available_adversaries()`/`render_pair()` exclude it unless enabled, and the
  preview CLI needs `--enable-authority-verifiable`.
- Control tokens: `WALK_AWAY_TOKEN` is imported from `engine` (can't drift); the
  shown `[DEAL $X]` example must satisfy `engine.DEAL_PATTERN` (a test checks).
Only the buyer prompt ever changes with `adversary` (for `authority_verifiable`);
otherwise buyer depends on (scenario, defense), seller on (scenario, adversary).

**Preview CLI (`src/preview.py`).**
`python -m src.preview --scenario s01 --defense basic --adversary rapport`
prints both prompts; `--run --model-a <buyer> --model-b <seller>` executes one
live conversation (buyer opens) and saves the transcript. Reuses the engine,
`get_client`, `save_transcript`, and `smoke._print_transcript`.

**Evaluation layer (`src/{jsonparse,extraction,judge,metrics,evaluate}.py`).**
`evaluate_run(transcript, scenario, config, *, defense, adversary, ...)` scores a
finished run into a flat `RunResult`, one JSONL line via `append_result`. The
condition labels (defense/adversary) are **passed in** — the `Transcript` doesn't
record them, so the caller must supply what it ran. Pieces:
- `jsonparse.request_json(client, messages, system, model_cls, retries, temperature)`
  calls `chat(json_mode=True)`, validates against a pydantic model, and on
  failure feeds the error back and retries up to `retries` times. Shared by
  extraction and judge. `extract_json_object` uses the JSON decoder (not brace
  counting) so braces inside strings are safe. **Provider gotcha:** Groq's JSON
  mode *validates server-side* and returns a **400** (`json_validate_failed`,
  with the bad text in `body.error.failed_generation`) instead of a reply when
  the model emits invalid JSON; `request_json` catches any `status_code == 400`,
  pulls out `failed_generation`, and treats it as a failed attempt (so it repairs
  or falls through to the regex salvage rather than crashing). Non-400 errors
  propagate.
- `extraction.run_extraction` = the adversary's own model guessing the buyer's
  private info (reservation + 80% interval, urgency, sensitive label). Path:
  parse → repair retries → **regex salvage** (`_regex_fallback`) → mark invalid.
  Never raises. `ExtractionResult.method` records which path won.
- `judge.run_judgement` = an independent model labelling each buyer turn per
  attribute (`explicit_leak`/`implied`/`none`) with evidence, given the ground
  truth (so it scores *disclosure*, not guess accuracy) and the buyer's own
  system prompt (to flag verbatim `prompt_leak`). **Temperature gotcha:** it sends
  `temperature=None` when `client.config.temperature is None` (frontier models
  reject an explicit temperature), else the config's `judge_temperature` (0.0).
- `metrics.build_run_result` is **pure** — all the math, no I/O — so it's exactly
  testable. Chance baselines derive from enum sizes (`URGENCY_CHANCE=1/3`,
  `SENSITIVE_CHANCE=1/6`). First-leak turn = min buyer-turn index the judge marked
  non-`none` for that attribute. When extraction is invalid, accuracy fields are
  `None` (not 0). `EvalConfig` picks extraction/judge models independently;
  `extraction_model=None` ⇒ the seller's own model. `results/` is gitignored.

**Dashboard (`src/server.py` + `src/static/`).**
`create_app(registry_path, runs_dir, client_factory, scenarios_dir)` returns a
FastAPI app; `client_factory` (default `get_client`) is injected so tests
exercise the whole API against `MockBackend`. **No module-level state** — the run
registry is a `RunStore` on `app.state`.

Each run executes `run_conversation` on a daemon thread, publishing turns via
`on_turn` under a `threading.Condition`. `GET /api/runs/{id}/events` is an SSE
stream (`turn` events, then a terminal `end` snapshot) whose generator waits on
that condition and **replays turns already recorded**, so a late or reconnecting
listener catches up. Model clients are built in the POST handler, not the
thread, so an unknown model or missing key is a 400 rather than a mid-run error.

Endpoints: `/api/models` (registry + `available` per API key), `/api/defaults`
(free-form form prefill, from `smoke.py`), `/api/scenarios` (ground truth for the
UI), `/api/conditions` (defenses + adversaries, each flagged `gated`),
`/api/scenarios/{id}/prompts` (renders buyer/seller prompts via `render_pair`),
`POST /api/runs` (accepts optional `conditions` = scenario_id/defense/adversary,
stored as run provenance), `GET /api/runs[/id]`, `POST /api/runs/{id}/cancel`,
`POST /api/runs/{id}/evaluate`, `GET /api/runs/{id}/events`, `/api/history[/file]`
(saved transcripts; filename validated against traversal),
`GET /api/history/{file}/raw` (the transcript JSON verbatim, for audit),
`POST /api/history/{file}/evaluate`. `ConversationView` is one wire shape for both
live runs and saved transcripts (now also carrying `metadata` + `evaluation`), so
the page has a single renderer.

**Evaluate-from-dashboard.** The two `/evaluate` endpoints share the in-closure
`_evaluate(transcript, req, cache_key)`: it reads conditions from the transcript
`metadata` (or `req.conditions` for a free-form run — else 400), runs
`evaluate_run` with the **same injected `build_client`** (so eval-model tests use
the mock backend too), appends the `RunResult` to `results_path`, and caches it in
`app.state.evals` keyed by run id / filename. Evaluating a still-running live run
is a 409. Evaluation is synchronous (FastAPI runs the sync handler in a
threadpool, so it doesn't block the event loop).

The frontend is plain HTML/CSS/JS served from `src/static/` — **no npm, no build
step**; keep it that way. Two setup modes: **scenario** — all scenarios are
**preloaded as a clickable list**; set models + defense/adversary once, then a
row's **▶** renders that scenario's prompts and runs it in one click (a scenario
`<input type=hidden name=scenario>` holds the selection). The ground-truth panel
is for the *researcher* — fine to show secrets; immersion only governs
agent-facing prompts. **free-form** — edit prompts by hand. In scenario mode the
names lock to buyer/seller, the buyer opens, and `conditions` is sent with the run
(recording provenance). The transcript panel has a **run-meta bar** (condition
chips, judge-model picker, Evaluate/Re-evaluate, Raw JSON) and an **eval panel**
that renders the `RunResult`; history rows show `scenario/defense/adversary` +
an evaluated tick.

## Registry (`models.yaml`) — 10 entries

Short names: `claude-opus`, `claude-sonnet`, `gpt-sol`, `gpt-mini`,
`gemini-pro`, `gemini-flash`, `llama-70b`, `llama-8b`, `gpt-oss-120b`,
`gpt-oss-20b`.

Current model IDs (verified against provider docs July 2026 — **verify again if
touching these; they drift fast**):
- Anthropic: `claude-opus-4-8`, `claude-sonnet-5`
- OpenAI: `gpt-5.6-sol` (flagship), `gpt-5.4-mini`
- Google (OpenAI-compat endpoint): `gemini-3.1-pro-preview`, `gemini-3.5-flash`
- Groq: `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`,
  `openai/gpt-oss-120b`, `openai/gpt-oss-20b`

API keys via env / `.env` (gitignored): `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
`GEMINI_API_KEY`, `GROQ_API_KEY`. A missing key only disables the entries that
need it. Loaded at startup via python-dotenv (`load_dotenv(override=False)`).

## Provider gotchas (important)

- **Sampling params:** `claude-opus-4-8`, `claude-sonnet-5`, and the GPT-5.x
  family **reject an explicit `temperature`** (HTTP 400). Those registry
  entries set `temperature: null` so the param is omitted. Do not add a
  temperature to them. Groq/Gemini open models accept temperature normally.
- **`max_tokens` name:** the GPT-5.x family requires `max_completion_tokens`,
  not `max_tokens`. Handled via `max_tokens_param` in the registry (default
  `max_tokens`; `gpt-sol`/`gpt-mini` override to `max_completion_tokens`).
- **Sonnet thinking:** `claude-sonnet-5` runs adaptive thinking when unset, so
  its entry has `extra_request: {thinking: {type: disabled}}` for plain,
  predictable-cost replies.
- **Gemini:** uses Google's **beta** OpenAI-compat endpoint
  (`https://generativelanguage.googleapis.com/v1beta/openai/`). If JSON mode
  proves unreliable there, the planned fallback is a native `google` backend
  using the google-genai SDK — try compat first.
- **Reasoning models** (`gpt-oss-*`): given larger default `max_tokens` (2048)
  to leave room for reasoning tokens.

## Common tasks

A `Makefile` wraps all of the below (`make help` lists them; `make check` =
lint + tests, `make run` = the dashboard, `make run-cli` = the smoke test,
`make preview` = render a scenario's prompts, `make eval` = evaluate a
transcript, `make ping-models` = the live check). The raw `uv` commands still
work — the Makefile is a convenience, not the source of truth.

Run tests / lint (offline, no keys needed):
```sh
uv run pytest -q
uv run ruff check . && uv run ruff format --check .
```

Render a scenario's prompts, or run one live conversation under conditions:
```sh
uv run python -m src.preview --scenario s01 --defense basic --adversary rapport
uv run python -m src.preview --scenario s01 --adversary authority --run \
    --model-a llama-8b --model-b gpt-oss-20b
```
`--enable-authority-verifiable` is required to select that gated defense arm.

Evaluate a saved transcript into a `RunResult` JSONL row (spends credit for the
extraction + judge calls):
```sh
uv run python -m src.evaluate --transcript runs/<file>.json --scenario s01 \
    --defense none --adversary authority   # --judge-model / --extraction-model optional
```
You must pass the `--defense`/`--adversary` the run actually used; the transcript
doesn't record them.

Run the dashboard (needs API keys; spends real credit when you hit Run):
```sh
uv run python -m src.server            # http://127.0.0.1:8000
```
Pick two models, edit both system prompts, run, watch turns stream in with
per-turn latency/tokens/cost. `--host`, `--port`, `--registry`, `--runs-dir`
are optional. `#run=<file>` deep-links a saved transcript.

Run the smoke test (needs the relevant API keys in `.env`):
```sh
uv run python -m src.smoke --model-a claude-sonnet --model-b llama-70b
```
`--model-a` is the buyer, `--model-b` the seller; `--max-turns`, `--registry`,
`--runs-dir` are optional.

Run the per-provider live check (spends a fraction of a cent of real credit):
```sh
uv run python scripts/live_check.py            # PASS/SKIP/FAIL per entry, incl. JSON mode
```
Entries whose key is unset report SKIP. **As of the last session this had never
been run — no API keys were present anywhere.** Running it is the first thing to
do once keys exist; watch the Gemini JSON-mode column specifically.

**Add a model:** add an entry to `models.yaml` (backend, model_id, api_key_env,
+ base_url for openai_compat; set `temperature: null` if the model rejects
sampling params; set `max_tokens_param: max_completion_tokens` for GPT-5.x).
No code change needed.

**Add a backend:** subclass `ModelClient`, implement `chat` + set `_RETRYABLE`,
register it in `src/models.py`'s `_BACKENDS` dict, and add its config fields to
`_CONFIG_FIELDS` if any are new.

**Use the engine programmatically:**
```python
from src.engine import Agent, run_conversation
from src.models import get_client
from src.persistence import save_transcript

a = Agent(name="alice", system_prompt="...", client=get_client("claude-sonnet"))
b = Agent(name="bob", system_prompt="...", client=get_client("gpt-mini"))
t = run_conversation(a, b, max_turns=6, opening_speaker="alice")
save_transcript(t)  # -> runs/<stamp>-alice-vs-bob.json
```

## Testing approach

Tests use a `MockBackend(ModelClient)` defined in `tests/test_engine.py` that
returns canned replies in order and records every call (messages, system,
temperature, max_tokens, json_mode). This is how the engine is tested with **no
network**: alternation, perspective/role flipping, non-crossing system prompts,
control-token termination + amount capture, `on_turn`/`cancelled` hooks,
config/usage capture, gen-param passthrough, and save/load round-trip. Reuse
`MockBackend` / `make_agent(...)` for any new engine tests. `tests/__init__.py`
exists so `tests.test_engine` is importable by the other test modules.

`tests/test_server.py` drives the real app through `TestClient` with a
`client_factory` of `GatedBackend`s (MockBackend + a `threading.Event` gate).
The gate exists because mock replies are instantaneous — without it a run can
finish before the POST that started it returns, and "cancel mid-run" is a race.
Close the gate to make "a turn is in flight" a waitable state. Note `TestClient`
**buffers** streaming responses, so the thread reading an SSE stream cannot also
release the gate; hand that to a timer (see the live-stream test).

`tests/test_scenarios.py` validates all 12 files, the label/pretext
distributions, and the `sensitive_context` none-detail rule. `tests/test_prompts.py`
renders the **full (scenario × defense × adversary) grid** and asserts the
immersion invariants (no experimental-frame terms in any prompt; the buyer never
learns the counterpart's nature; only `human_impersonation` claims humanity),
the three `authority` constraints, pretext↔scenario matching, the
`human_impersonation == direct_probe + clause` control, the config gate, and
engine control-token sync. `tests/test_preview.py` exercises the CLI offline
(never `--run`). If you hand-edit a template, expect the content-substring
assertions there to be what breaks — update them deliberately.

`tests/test_evaluation.py` builds mini transcripts by hand (helper
`make_transcript`) and checks the metric math exactly (explicit/implied/clean/
overpaid, interval calibration, invalid-extraction ⇒ null accuracy), plus the
extraction repair ladder against the mock backend: clean parse, repair after
broken JSON, regex salvage, and unsalvageable ⇒ invalid. Feed deliberately
broken JSON as canned `MockBackend` replies to exercise the repair path. The
provider-400 path uses `FlakyJsonBackend` (raises a fake `status_code=400` with a
`body.error.failed_generation`) — asserts repair and regex-salvage both survive
it, and that a non-400 error still propagates.

The dashboard evaluate endpoints are tested in `test_server.py`: `Fleet` also
serves canned extraction/judge JSON for the model names `extractmock`/`judgemock`
(pass them as `extraction_model`/`judge_model` in the eval body so the run's
negotiation mocks and the eval mocks don't collide). The `client` fixture points
`results_path` at tmp so eval runs don't write into the repo's `results/`.

## State / not-yet-done

- One commit exists (`1 - Core Mechanism`); the owner commits manually, so leave
  committing to them unless asked.
- All four API keys are in `.env`. Verified live end-to-end (2026-07-16/17):
  the **dashboard** drives a scenario run (provenance recorded) → **Evaluate**
  produces a coherent `RunResult` in-page; `preview --run` + `evaluate` CLI also
  work. During this a real bug surfaced and was fixed: Groq JSON mode 400s on
  invalid JSON instead of returning it, which crashed extraction — now handled in
  `jsonparse` (see the provider gotcha above). `scripts/live_check.py` (also
  probes JSON mode incl. the Gemini compat endpoint) still hasn't been run — do
  that next.
- Naming: the owner refers to the project as "leaklab", but the **package stays
  `src/` with generic names** (confirmed 2026-07-15 when a sprint spec said
  `leaklab/`). Keep new modules under `src/`; the folder name is separate.
- Not built (by design): an **experiment runner** that sweeps the
  (scenario × defense × adversary × model) grid, and aggregation/analysis over
  the `results/*.jsonl` rows. Don't add these without being asked. The dashboard
  and preview/evaluate CLIs are single-run tools.
  - **Explicit preference (2026-07-16):** the owner does *not* want scenarios fed
    automatically. They want them **preloaded and fired one at a time by hand**
    (the dashboard's scenario list + ▶). Don't build an auto-batch sweep unless
    they ask in those words.
