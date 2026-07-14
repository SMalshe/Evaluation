"""HTTP API + single-page dashboard for running and monitoring conversations.

Wraps the engine: pick two registry models, edit their system prompts, run a
conversation, and watch turns stream in live. Completed runs are saved through
``persistence`` and re-readable from the same UI.

Usage:
    python -m src.server            # http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import uuid
from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .engine import Agent, Transcript, Turn, run_conversation
from .models import (
    MissingAPIKeyError,
    ModelClient,
    ModelConfig,
    RegistryError,
    get_client,
    load_registry,
)
from .persistence import load_transcript, save_transcript
from .smoke import BUYER_SYSTEM, OPENING_PROMPT, SELLER_SYSTEM

STATIC_DIR = Path(__file__).parent / "static"
HEARTBEAT_S = 15.0  # SSE keep-alive interval while a run is idle between turns
MAX_RUNS_IN_MEMORY = 50
SAFE_FILENAME = re.compile(r"^[A-Za-z0-9._-]+\.json$")

ClientFactory = Callable[[str], ModelClient]
RunStatus = Literal["running", "done", "cancelled", "error"]


# --- wire formats -----------------------------------------------------------


class ModelEntry(BaseModel):
    """A registry entry as the UI sees it, including whether its key is set."""

    name: str
    backend: str
    model_id: str
    api_key_env: str
    available: bool
    temperature: float | None
    max_tokens: int
    price_per_mtok_in: float
    price_per_mtok_out: float


class Defaults(BaseModel):
    """Prefill for the run form (the smoke-test personas, freely editable)."""

    agent_a: dict[str, str]
    agent_b: dict[str, str]
    opening_prompt: str
    max_turns: int


class AgentSpec(BaseModel):
    name: str
    model: str  # registry short name
    system_prompt: str
    temperature: float | None = None  # None = registry default
    max_tokens: int | None = None  # None = registry default


class RunRequest(BaseModel):
    agent_a: AgentSpec
    agent_b: AgentSpec
    max_turns: int = 6
    opening_speaker: str | None = None
    opening_prompt: str = "You may begin."


class TurnView(BaseModel):
    """A turn plus the model that produced it and what it cost."""

    index: int
    speaker: str
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    model_name: str
    model_id: str
    cost_usd: float


class AgentView(BaseModel):
    name: str
    model_name: str
    model_id: str
    backend: str
    system_prompt: str
    temperature: float | None
    max_tokens: int | None


class Totals(BaseModel):
    turns: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: float


class ConversationView(BaseModel):
    """One shape for both live runs and saved transcripts, so the UI has one renderer."""

    id: str
    source: Literal["live", "saved"]
    status: RunStatus
    agents: list[AgentView]
    turns: list[TurnView]
    totals: Totals
    termination: str | None = None
    deal_amount: str | None = None
    error: str | None = None
    saved_path: str | None = None
    max_turns: int
    opening_speaker: str
    opening_prompt: str
    started_at: datetime
    ended_at: datetime | None = None


class HistoryEntry(BaseModel):
    file: str
    agents: list[str]
    models: list[str]
    turns: int
    termination: str
    deal_amount: str | None
    started_at: datetime


# --- run bookkeeping --------------------------------------------------------


@dataclass
class RunState:
    """A conversation running on a worker thread, observed by SSE listeners."""

    id: str
    request: RunRequest
    agents: list[AgentView]
    started_at: datetime
    condition: threading.Condition = field(default_factory=threading.Condition)
    cancel: threading.Event = field(default_factory=threading.Event)
    turns: list[TurnView] = field(default_factory=list)
    status: RunStatus = "running"
    transcript: Transcript | None = None
    error: str | None = None
    saved_path: str | None = None
    ended_at: datetime | None = None


class RunStore:
    """Bounded, thread-safe map of run id -> RunState (oldest evicted first)."""

    def __init__(self, limit: int = MAX_RUNS_IN_MEMORY) -> None:
        self._runs: OrderedDict[str, RunState] = OrderedDict()
        self._limit = limit
        self._lock = threading.Lock()

    def add(self, state: RunState) -> None:
        with self._lock:
            self._runs[state.id] = state
            while len(self._runs) > self._limit:
                self._runs.popitem(last=False)

    def get(self, run_id: str) -> RunState | None:
        with self._lock:
            return self._runs.get(run_id)

    def all(self) -> list[RunState]:
        with self._lock:
            return list(self._runs.values())


def _cost_usd(prompt_tokens: int, completion_tokens: int, config: ModelConfig | None) -> float:
    if config is None:  # a saved run referencing a model no longer in the registry
        return 0.0
    return (
        prompt_tokens * config.price_per_mtok_in + completion_tokens * config.price_per_mtok_out
    ) / 1_000_000


def _totals(turns: Sequence[TurnView]) -> Totals:
    return Totals(
        turns=len(turns),
        prompt_tokens=sum(t.prompt_tokens for t in turns),
        completion_tokens=sum(t.completion_tokens for t in turns),
        cost_usd=sum(t.cost_usd for t in turns),
        latency_ms=sum(t.latency_ms for t in turns),
    )


def _turn_view(turn: Turn, config: ModelConfig) -> TurnView:
    return TurnView(
        index=turn.index,
        speaker=turn.speaker,
        text=turn.text,
        prompt_tokens=turn.prompt_tokens,
        completion_tokens=turn.completion_tokens,
        latency_ms=turn.latency_ms,
        model_name=config.name,
        model_id=config.model_id,
        cost_usd=_cost_usd(turn.prompt_tokens, turn.completion_tokens, config),
    )


def _view_of_run(state: RunState) -> ConversationView:
    with state.condition:
        turns = list(state.turns)
        transcript = state.transcript
        return ConversationView(
            id=state.id,
            source="live",
            status=state.status,
            agents=list(state.agents),
            turns=turns,
            totals=_totals(turns),
            termination=transcript.termination if transcript else None,
            deal_amount=transcript.deal_amount if transcript else None,
            error=state.error,
            saved_path=state.saved_path,
            max_turns=state.request.max_turns,
            opening_speaker=state.request.opening_speaker or state.request.agent_a.name,
            opening_prompt=state.request.opening_prompt,
            started_at=state.started_at,
            ended_at=state.ended_at,
        )


def _view_of_transcript(
    transcript: Transcript,
    registry: dict[str, ModelConfig],
    file: str,
) -> ConversationView:
    infos = {info.name: info for info in transcript.agents}
    turns = [
        TurnView(
            index=turn.index,
            speaker=turn.speaker,
            text=turn.text,
            prompt_tokens=turn.prompt_tokens,
            completion_tokens=turn.completion_tokens,
            latency_ms=turn.latency_ms,
            model_name=infos[turn.speaker].model_name,
            model_id=infos[turn.speaker].model_id,
            cost_usd=_cost_usd(
                turn.prompt_tokens,
                turn.completion_tokens,
                registry.get(infos[turn.speaker].model_name),
            ),
        )
        for turn in transcript.turns
    ]
    return ConversationView(
        id=file,
        source="saved",
        status="cancelled" if transcript.termination == "cancelled" else "done",
        agents=[
            AgentView(
                name=info.name,
                model_name=info.model_name,
                model_id=info.model_id,
                backend=info.backend,
                system_prompt=info.system_prompt,
                temperature=info.temperature,
                max_tokens=info.max_tokens,
            )
            for info in transcript.agents
        ],
        turns=turns,
        totals=_totals(turns),
        termination=transcript.termination,
        deal_amount=transcript.deal_amount,
        saved_path=file,
        max_turns=transcript.max_turns,
        opening_speaker=transcript.opening_speaker,
        opening_prompt=transcript.opening_prompt,
        started_at=transcript.started_at,
        ended_at=transcript.ended_at,
    )


def _sse(event: str, payload: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


# --- app --------------------------------------------------------------------


def create_app(
    registry_path: str | Path = "models.yaml",
    runs_dir: str | Path = "runs",
    client_factory: ClientFactory | None = None,
) -> FastAPI:
    """Build the dashboard app.

    ``client_factory`` maps a registry short name to a ``ModelClient``; it is
    injectable so tests can run the whole API against a mock backend.
    """
    load_dotenv(override=False)
    registry_path = Path(registry_path)
    runs_dir = Path(runs_dir)
    build_client: ClientFactory = client_factory or (
        lambda name: get_client(name, registry_path=registry_path)
    )

    app = FastAPI(title="Conversation dashboard", docs_url="/api/docs")
    app.state.runs = RunStore()

    def registry() -> dict[str, ModelConfig]:
        try:
            return load_registry(registry_path)
        except RegistryError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/api/models", response_model=list[ModelEntry])
    def list_models() -> list[ModelEntry]:
        """Registry entries, flagged with whether their API key is actually set."""
        return [
            ModelEntry(
                name=config.name,
                backend=config.backend,
                model_id=config.model_id,
                api_key_env=config.api_key_env,
                available=bool(os.environ.get(config.api_key_env)),
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                price_per_mtok_in=config.price_per_mtok_in,
                price_per_mtok_out=config.price_per_mtok_out,
            )
            for config in registry().values()
        ]

    @app.get("/api/defaults", response_model=Defaults)
    def defaults() -> Defaults:
        return Defaults(
            agent_a={"name": "buyer", "system_prompt": BUYER_SYSTEM},
            agent_b={"name": "seller", "system_prompt": SELLER_SYSTEM},
            opening_prompt=OPENING_PROMPT,
            max_turns=6,
        )

    @app.post("/api/runs", response_model=ConversationView, status_code=201)
    def start_run(request: RunRequest) -> ConversationView:
        if request.agent_a.name == request.agent_b.name:
            raise HTTPException(status_code=400, detail="Agents must have distinct names")
        if request.max_turns < 1:
            raise HTTPException(status_code=400, detail="max_turns must be >= 1")
        names = {request.agent_a.name, request.agent_b.name}
        if request.opening_speaker is not None and request.opening_speaker not in names:
            raise HTTPException(
                status_code=400,
                detail=f"opening_speaker must be one of {sorted(names)}",
            )

        # Build clients up front: a bad model name or missing key is the caller's
        # error (400), not a mid-run failure on the worker thread.
        agents: list[Agent] = []
        for spec in (request.agent_a, request.agent_b):
            try:
                client = build_client(spec.model)
            except (RegistryError, MissingAPIKeyError) as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            agents.append(
                Agent(
                    name=spec.name,
                    system_prompt=spec.system_prompt,
                    client=client,
                    temperature=spec.temperature,
                    max_tokens=spec.max_tokens,
                )
            )

        agent_a, agent_b = agents
        state = RunState(
            id=uuid.uuid4().hex[:12],
            request=request,
            agents=[_agent_view(agent) for agent in agents],
            started_at=datetime.now(UTC),
        )
        app.state.runs.add(state)

        thread = threading.Thread(
            target=_run_worker,
            args=(state, agent_a, agent_b, runs_dir),
            name=f"run-{state.id}",
            daemon=True,
        )
        thread.start()
        return _view_of_run(state)

    @app.get("/api/runs", response_model=list[ConversationView])
    def list_runs() -> list[ConversationView]:
        return [_view_of_run(state) for state in reversed(app.state.runs.all())]

    @app.get("/api/runs/{run_id}", response_model=ConversationView)
    def get_run(run_id: str) -> ConversationView:
        return _view_of_run(_require_run(app, run_id))

    @app.post("/api/runs/{run_id}/cancel", response_model=ConversationView)
    def cancel_run(run_id: str) -> ConversationView:
        state = _require_run(app, run_id)
        state.cancel.set()  # takes effect after the in-flight turn returns
        return _view_of_run(state)

    @app.get("/api/runs/{run_id}/events")
    def run_events(run_id: str) -> StreamingResponse:
        state = _require_run(app, run_id)
        return StreamingResponse(
            _event_stream(state),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/history", response_model=list[HistoryEntry])
    def history() -> list[HistoryEntry]:
        """Saved transcripts, newest first."""
        entries: list[HistoryEntry] = []
        for path in sorted(runs_dir.glob("*.json"), reverse=True):
            try:
                transcript = load_transcript(path)
            except Exception:  # a hand-edited or partial file shouldn't break the list
                continue
            entries.append(
                HistoryEntry(
                    file=path.name,
                    agents=[info.name for info in transcript.agents],
                    models=[info.model_name for info in transcript.agents],
                    turns=len(transcript.turns),
                    termination=transcript.termination,
                    deal_amount=transcript.deal_amount,
                    started_at=transcript.started_at,
                )
            )
        return entries

    @app.get("/api/history/{file}", response_model=ConversationView)
    def get_history(file: str) -> ConversationView:
        if not SAFE_FILENAME.match(file):
            raise HTTPException(status_code=400, detail="Invalid transcript name")
        path = runs_dir / file
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"No such transcript: {file}")
        return _view_of_transcript(load_transcript(path), registry(), file)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def _require_run(app: FastAPI, run_id: str) -> RunState:
    state = app.state.runs.get(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"No such run: {run_id}")
    return state


def _agent_view(agent: Agent) -> AgentView:
    config = agent.client.config
    return AgentView(
        name=agent.name,
        model_name=config.name,
        model_id=config.model_id,
        backend=config.backend,
        system_prompt=agent.system_prompt,
        temperature=agent.temperature if agent.temperature is not None else config.temperature,
        max_tokens=agent.max_tokens if agent.max_tokens is not None else config.max_tokens,
    )


def _run_worker(state: RunState, agent_a: Agent, agent_b: Agent, runs_dir: Path) -> None:
    """Drive one conversation to completion, publishing turns as they land."""
    configs = {agent.name: agent.client.config for agent in (agent_a, agent_b)}

    def on_turn(turn: Turn) -> None:
        view = _turn_view(turn, configs[turn.speaker])
        with state.condition:
            state.turns.append(view)
            state.condition.notify_all()

    try:
        transcript = run_conversation(
            agent_a,
            agent_b,
            max_turns=state.request.max_turns,
            opening_speaker=state.request.opening_speaker,
            opening_prompt=state.request.opening_prompt,
            on_turn=on_turn,
            cancelled=state.cancel.is_set,
        )
    except Exception as exc:  # surface provider/network errors in the UI
        with state.condition:
            state.status = "error"
            state.error = f"{type(exc).__name__}: {exc}"
            state.ended_at = datetime.now(UTC)
            state.condition.notify_all()
        return

    saved_path: str | None = None
    if transcript.turns:  # a run cancelled before its first turn isn't worth a file
        saved_path = str(save_transcript(transcript, runs_dir=runs_dir))

    with state.condition:
        state.transcript = transcript
        state.saved_path = saved_path
        state.status = "cancelled" if transcript.termination == "cancelled" else "done"
        state.ended_at = transcript.ended_at
        state.condition.notify_all()


def _event_stream(state: RunState) -> Iterator[str]:
    """SSE: one ``turn`` event per completed turn, then a final ``end`` snapshot.

    Replays turns already recorded, so a reconnecting or late client catches up.
    """
    sent = 0
    while True:
        with state.condition:
            while len(state.turns) == sent and state.status == "running":
                if not state.condition.wait(timeout=HEARTBEAT_S):
                    break
            new = state.turns[sent:]
            sent = len(state.turns)
            finished = state.status != "running"

        for turn in new:
            yield _sse("turn", turn.model_dump(mode="json"))
        if finished:
            yield _sse("end", _view_of_run(state).model_dump(mode="json"))
            return
        if not new:
            yield ": keep-alive\n\n"


def main(argv: Sequence[str] | None = None) -> int:
    import uvicorn

    parser = argparse.ArgumentParser(
        prog="python -m src.server",
        description="Serve the conversation dashboard.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--registry", default="models.yaml", help="path to the model registry")
    parser.add_argument("--runs-dir", default="runs", help="directory for saved transcripts")
    args = parser.parse_args(argv)

    app = create_app(registry_path=args.registry, runs_dir=args.runs_dir)
    print(f"dashboard: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
