# Two-agent conversation foundations

Foundations for running LLM agent-to-agent conversations: a unified model
client over multiple providers, a two-agent turn-alternation engine with JSON
transcript persistence, and a browser dashboard for driving and watching runs.

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
| `src/server.py`         | Dashboard: HTTP API + SSE turn stream (`python -m src.server`)  |
| `src/static/`           | The dashboard page (plain HTML/CSS/JS, no build step)           |
| `src/smoke.py`          | CLI smoke test (bicycle haggle)                                 |
| `scripts/live_check.py` | One live "say OK" + JSON-mode probe per registry entry          |
| `tests/`                | Offline tests against a mock backend                            |

## Dashboard

```sh
uv run python -m src.server        # http://127.0.0.1:8000
```

One page to pick the two models, edit each side's system prompt, run the
conversation, and watch turns stream in as they land — with per-turn latency,
tokens, and cost, plus running totals and the termination reason. Models that
lack an API key are shown disabled rather than hidden. A run can be cancelled
mid-flight (it stops after the turn already in flight and still saves what it
got). Every finished run is written to `runs/` and is reopenable from the
sidebar, which also refills the form so you can change one knob and re-run.

Options: `--host`, `--port`, `--registry`, `--runs-dir`.

## Usage

Run a 6-turn haggle between two registry models and pretty-print it:

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
