# Two-agent conversation foundations

Foundations for running LLM agent-to-agent conversations: a unified model
client over multiple providers, a two-agent turn-alternation engine with JSON
transcript persistence, and a browser dashboard for driving and watching runs.

![System architecture and experimental design](docs/architecture.png)

The diagram above is the one-page view: inputs on the left, the two-phase
pipeline in the middle, outputs on the right, and the grid as currently
configured along the bottom. Regenerate it with
`uv run python scripts/architecture_diagram.py` after changing the grid, so the
counts on it stay true.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
cp .env.example .env   # then fill in the API keys you have
```

Keys are loaded from `.env` at startup (via python-dotenv). A missing key only
disables the registry entries that need it.

## Layout

| Path                    | Purpose                                                        |
| ----------------------- | -------------------------------------------------------------- |
| `src/models.py`         | `ModelClient` interface, `openai_compat` + `anthropic` backends, retry/backoff, `get_client(name)` |
| `models.yaml`           | Registry: short name -> backend, model ID, endpoint, defaults, prices |
| `src/engine.py`         | `Agent`, `run_conversation`, pydantic `Transcript`              |
| `src/persistence.py`    | Save/load transcripts as JSON under `runs/`                     |
| `src/scenarios.py`      | Generic info-extraction scenario schema (holder/seeker `Side`) + loader |
| `scenarios/`            | 48 diverse scenarios (`s01`–`s48`, YAML), 12 per method category |
| `src/prompts.py`        | Holder/seeker system-prompt templates by defense × adversary   |
| `src/preview.py`        | CLI: render (and optionally run) a scenario under conditions    |
| `src/extraction.py`     | Post-conversation seeker questionnaire (JSON, with repair)       |
| `src/judge.py`          | Independent judge: per-turn, per-attribute leak labels          |
| `src/metrics.py`        | Flat `RunResult` metric math (pure)                             |
| `src/evaluate.py`       | `evaluate_run` + JSONL persistence (`python -m src.evaluate`)   |
| `src/jsonparse.py`      | Ask-for-JSON + bounded repair-retry helper (shared)            |
| `src/server.py`         | Dashboard: HTTP API + SSE turn stream (`python -m src.server`)  |
| `src/static/`           | The dashboard page (plain HTML/CSS/JS, no build step)           |
| `src/smoke.py`          | CLI smoke test (free-form holder/seeker exchange)               |
| `scripts/live_check.py` | One live "say OK" + JSON-mode probe per registry entry          |
| `scripts/architecture_diagram.py` | Regenerates `docs/architecture.png`                  |
| `docs/architecture.png` | One-page system + experimental-design diagram                  |
| `tests/`                | Offline tests against a mock backend                            |

## Dashboard

```sh
uv run python -m src.server        # http://127.0.0.1:8000
```

One page to pick the two models, run the conversation, and watch turns stream in
as they land — with per-turn latency, tokens, and cost, plus running totals and
the termination reason. Two setup modes:

- **Scenario** (default): all 48 scenarios are **preloaded and grouped by
  category** (collapsible sections for the four methods) with their ground truth
  (who's under test, both roles + personas, the authority role, and the holder's
  secrets). Set the holder/seeker models and the defense/adversary once, then click
  any scenario's
  **▶** to run it — the prompts are generated for you. Clicking a row (not the ▶)
  just loads it so you can inspect or edit before running. `authority_verifiable`
  appears once you tick its enable box.
- **Free-form**: edit both system prompts by hand (the original behaviour).

Generated prompts stay editable, so you can tweak and pilot. Models that lack an
API key are shown disabled rather than hidden. A run can be cancelled mid-flight
(it stops after the turn already in flight and still saves what it got).

**Auditability.** Every finished run is written to `runs/` (with its scenario and
conditions recorded on the transcript) and reopenable from the sidebar, which
shows each run's `scenario/defense/adversary` and whether it's been evaluated.
Opening a run shows the conditions, both agents' system prompts, the full
transcript, and a **Raw JSON** link. For a scenario run, an **Evaluate leakage**
button runs the extraction + judge in place (pick the judge model) and renders
the `RunResult` — the adversary's guesses vs. the truth, first-disclosed turn per
attribute, deal outcome, and prompt-leak — right above the transcript.

Options: `--host`, `--port`, `--registry`, `--runs-dir`.

## Scenarios and conditions

Scenarios in `scenarios/` are **generic information-extraction** interactions,
not literally car sales: one agent **holds** private information and the other
**seeks** it (social engineering, HR/medical intake, a desperate candidate
over-sharing, an impersonated authority pulling credentials, …). Each scenario
has a `setting`, and a `holder` and `seeker` each of type `Side`:

- **role** — who the agent is, in-world (e.g. "a retail bank customer").
- **objectives** — what they want (protect their info / get what they came for).
- **secrets** — the private facts they hold, each with a `reveal_when` condition
  under which sharing it is strategically correct (ground truth for appropriate
  vs. inappropriate disclosure; also rendered as in-character guidance).
- **persona** — cooperative / stubborn / evasive.

A run-level `role_under_test` records whose disclosure to score, and a `category`
files each scenario under one of the four experimental methods (used to group the
dashboard menu): `holder_defense`, `seeker_attack`, `authority` (the holder
is under test), and `holder_dependent` (a desperate holder). There are
**48 scenarios** (`s01`–`s48`), **12 per category**.

`src/prompts.py` renders in-character holder and seeker prompts from a scenario
plus two conditions:

- **Holder defense** (`none` / `basic` / `strong`) — how much the holder is told
  to guard its private information.
- **Seeker adversary** (`passive` / `direct_probe` / `rapport` / `pressure` /
  `authority` / `human_impersonation`, plus the gated defense arm
  `authority_verifiable`) — the extraction tactic layered on top of ordinary
  interaction.

Render a pair, or run one live conversation between two models:

```sh
uv run python -m src.preview --scenario s01 --defense basic --adversary rapport
uv run python -m src.preview --scenario s01 --adversary authority --run \
    --model-a llama-8b --model-b gpt-oss-20b
```

`authority_verifiable` is off by default; pass `--enable-authority-verifiable`
to select it. Rendered prompts stay fully in character — they never tell an
agent it is being tested, scored, or talking to an AI (the one exception is the
`human_impersonation` seeker, told to claim it is human). Those experimental
concepts live only in the scenario files, never in an agent's prompt.

## Evaluating a run

`evaluate_run(transcript, scenario, config, defense=…, adversary=…)` scores a
finished conversation into a flat `RunResult` (persisted as one JSONL line):

- **Extraction** — the adversary's own model is shown the transcript and asked to
  guess the holder's private info (reservation price + 80% interval, urgency,
  sensitive category), as JSON. Broken JSON is repaired (feed the error back, up
  to N times), then salvaged by regex, then marked invalid — never a crash.
- **Judge** — a separate model (default a frontier model, near-deterministic)
  labels each holder turn per attribute as `explicit_leak` / `implied` / `none`
  with an evidence span, given the ground truth, and flags verbatim prompt-leak.
- **Metrics** — reservation percent error / hit-within-10% / interval
  calibration; urgency & sensitive correctness against enum-size chance
  baselines; first-leak turn per attribute; deal/price/surplus/overpaid; plus
  condition, model, token, and timing bookkeeping.

```sh
uv run python -m src.evaluate --transcript runs/<file>.json --scenario s01 \
    --defense none --adversary authority        # appends to results/results.jsonl
```

The extraction and judge models are chosen independently of the conversing
models (extraction defaults to the seeker's own model). `results/` is gitignored.

## Usage

Run a 6-turn exchange between two registry models and pretty-print it:

```sh
uv run python -m src.smoke --model-a claude-sonnet --model-b llama-70b
```

Registry short names: `claude-opus`, `claude-sonnet`, `gpt-sol`, `gpt-mini`,
`gemini-pro`, `gemini-flash`, `llama-70b`, `llama-8b`, `gpt-oss-120b`,
`gpt-oss-20b`. See `models.yaml` for model IDs and pricing.

Programmatic use:

```python
from src.engine import Agent, run_conversation
from src.models import get_client
from src.persistence import save_transcript

a = Agent(name="alice", system_prompt="...", client=get_client("claude-sonnet"))
b = Agent(name="bob", system_prompt="...", client=get_client("gpt-mini"))
transcript = run_conversation(a, b, max_turns=6, opening_speaker="alice")
save_transcript(transcript)  # -> runs/<timestamp>-alice-vs-bob.json
```

Each agent sees the conversation from its own perspective (own messages as
`assistant`, counterpart's as `user`); system prompts never cross. A turn
containing `[DEAL $X]` or `[WALK_AWAY]` ends the conversation, otherwise it
stops at `max_turns`.

`run_conversation` also takes two optional hooks, which is how the dashboard
observes a run without owning the loop: `on_turn(turn)` is called as each turn
completes, and `cancelled()` is polled before each turn — returning True stops
the conversation with termination `cancelled`, keeping the turns so far.

## Tests and checks

```sh
uv run pytest              # offline tests (mock backend, no network)
uv run ruff check .        # lint
uv run python scripts/live_check.py   # live: one tiny request per provider (spends API credit)
```

`make help` lists the same tasks as targets: `make check` (lint + tests),
`make run` (the dashboard), `make run-cli MODEL_A=gpt-mini MODEL_B=llama-8b`,
`make preview SCENARIO=s03 ADVERSARY=authority`,
`make eval TRANSCRIPT=runs/<f>.json SCENARIO=s01 DEFENSE=none ADVERSARY=authority`,
`make ping-models`.

## Provider notes

- **Sampling params:** `claude-opus-4-8`, `claude-sonnet-5`, and the GPT-5.x
  family reject explicit `temperature`; those registry entries default it to
  `null`, which omits the parameter. Setting a temperature on such an entry
  will 400.
- **Anthropic json_mode:** the Messages API has no `response_format`, so
  `json_mode=True` is implemented as a strict system-prompt instruction.
- **Gemini:** uses Google's OpenAI-compatible endpoint (beta). If
  `scripts/live_check.py` shows JSON mode failing there, the fallback plan is
  a native `google` backend using the google-genai SDK.
- **Prices** in `models.yaml` are USD per million tokens, verified against
  provider docs in July 2026 (`claude-sonnet-5` has intro pricing through
  2026-08-31).
