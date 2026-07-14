"""Dashboard API tests: the real app, a mock backend, no network."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.models import Message, ModelClient, ModelResponse, RegistryError
from src.server import create_app
from tests.test_engine import MockBackend

REPLIES = {
    "fast": ["Interested. What's your floor?", "Deal then. [DEAL $140]"],
    "slow": ["I'd take $180.", "Alright, $140 works."],
}


class GatedBackend(MockBackend):
    """A mock that blocks inside ``chat`` until the test opens the gate.

    Mock replies are instantaneous, so without a gate a run can finish before
    the POST that started it even returns. The gate makes "a turn is in flight"
    an explicit, waitable state instead of a race.
    """

    def __init__(
        self,
        name: str,
        replies: Sequence[str],
        gate: threading.Event,
        entered: threading.Event,
    ) -> None:
        super().__init__(name, replies)
        self.config = replace(self.config, price_per_mtok_in=1.0, price_per_mtok_out=2.0)
        self._gate = gate
        self._entered = entered

    def chat(self, messages: Sequence[Message], **kwargs: Any) -> ModelResponse:
        self._entered.set()
        assert self._gate.wait(timeout=5.0), "gate never opened"
        return super().chat(messages, **kwargs)


class Fleet:
    """Stand-in for ``get_client``: hands out gated mocks, shares one gate."""

    def __init__(self) -> None:
        self.gate = threading.Event()
        self.gate.set()  # open unless a test closes it
        self.entered = threading.Event()

    def __call__(self, name: str) -> ModelClient:
        if name not in REPLIES:
            raise RegistryError(f"Unknown model {name!r}")
        return GatedBackend(name, REPLIES[name], self.gate, self.entered)


@pytest.fixture
def runs_dir(tmp_path: Path) -> Path:
    return tmp_path / "runs"


@pytest.fixture
def fleet() -> Fleet:
    return Fleet()


@pytest.fixture
def client(runs_dir: Path, fleet: Fleet) -> Iterator[TestClient]:
    with TestClient(create_app(runs_dir=runs_dir, client_factory=fleet)) as test_client:
        yield test_client


def start(client: TestClient, **overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "agent_a": {"name": "buyer", "model": "fast", "system_prompt": "you buy"},
        "agent_b": {"name": "seller", "model": "slow", "system_prompt": "you sell"},
        "max_turns": 6,
        "opening_speaker": "buyer",
        "opening_prompt": "Go.",
    }
    body.update(overrides)
    response = client.post("/api/runs", json=body)
    assert response.status_code == 201, response.text
    return response.json()


def wait_for_finish(client: TestClient, run_id: str, timeout_s: float = 5.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        view = client.get(f"/api/runs/{run_id}").json()
        if view["status"] != "running":
            return view
        time.sleep(0.01)
    raise AssertionError(f"run {run_id} did not finish within {timeout_s}s")


def read_events(client: TestClient, run_id: str) -> list[tuple[str, dict[str, Any]]]:
    """Drain the SSE stream up to and including the terminal ``end`` event."""
    events: list[tuple[str, dict[str, Any]]] = []
    with client.stream("GET", f"/api/runs/{run_id}/events") as response:
        assert response.status_code == 200
        event = ""
        for line in response.iter_lines():
            if line.startswith("event: "):
                event = line.removeprefix("event: ")
            elif line.startswith("data: "):
                events.append((event, json.loads(line.removeprefix("data: "))))
                if event == "end":
                    break
    return events


def test_models_endpoint_flags_key_availability(
    runs_dir: Path, fleet: Fleet, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The app loads .env on startup; neutralize it so the test owns the environment.
    monkeypatch.setattr("src.server.load_dotenv", lambda **_: False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "set-for-test")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with TestClient(create_app(runs_dir=runs_dir, client_factory=fleet)) as client:
        entries = {entry["name"]: entry for entry in client.get("/api/models").json()}

    assert entries["claude-opus"]["available"] is True
    assert entries["llama-70b"]["available"] is False
    assert entries["llama-70b"]["api_key_env"] == "GROQ_API_KEY"
    assert entries["gpt-sol"]["price_per_mtok_out"] > 0  # registry fields pass through


def test_defaults_prefill_the_form(client: TestClient) -> None:
    defaults = client.get("/api/defaults").json()

    assert defaults["agent_a"]["name"] == "buyer"
    assert "[DEAL $X]" in defaults["agent_a"]["system_prompt"]
    assert defaults["opening_prompt"]


def test_run_reports_progress_and_saves_a_transcript(
    client: TestClient, fleet: Fleet, runs_dir: Path
) -> None:
    fleet.gate.clear()  # hold the first turn so the initial snapshot is observable
    view = start(client)
    assert view["status"] == "running"
    assert view["turns"] == []
    assert view["agents"][0]["model_name"] == "fast"

    fleet.gate.set()
    final = wait_for_finish(client, view["id"])

    assert final["status"] == "done"
    assert final["termination"] == "deal"
    assert final["deal_amount"] == "140"
    assert [t["speaker"] for t in final["turns"]] == ["buyer", "seller", "buyer"]
    assert final["totals"]["turns"] == 3
    assert final["totals"]["cost_usd"] > 0
    assert final["turns"][0]["model_name"] == "fast"

    saved = list(runs_dir.glob("*.json"))
    assert len(saved) == 1
    assert final["saved_path"] == str(saved[0])


def test_event_stream_delivers_turns_live_then_ends(client: TestClient, fleet: Fleet) -> None:
    fleet.gate.clear()
    run_id = start(client)["id"]
    assert fleet.entered.wait(2.0), "worker never reached the first model call"
    assert client.get(f"/api/runs/{run_id}").json()["turns"] == []  # nothing to replay yet

    # TestClient buffers a streaming response, so this thread can't open the gate
    # once it starts reading: hand that off to a timer and then attach.
    threading.Timer(0.1, fleet.gate.set).start()
    events = read_events(client, run_id)

    assert [kind for kind, _ in events] == ["turn", "turn", "turn", "end"]
    assert [p["speaker"] for kind, p in events if kind == "turn"] == ["buyer", "seller", "buyer"]
    assert events[-1][1]["termination"] == "deal"
    assert events[0][1]["cost_usd"] > 0


def test_event_stream_replays_for_a_late_listener(client: TestClient) -> None:
    run_id = start(client)["id"]
    wait_for_finish(client, run_id)

    events = read_events(client, run_id)  # attaches only after the run is over

    assert [kind for kind, _ in events] == ["turn", "turn", "turn", "end"]


def test_cancel_stops_the_run_after_the_in_flight_turn(client: TestClient, fleet: Fleet) -> None:
    fleet.gate.clear()
    run_id = start(client, max_turns=40)["id"]
    assert fleet.entered.wait(2.0), "worker never reached the first model call"

    client.post(f"/api/runs/{run_id}/cancel")
    fleet.gate.set()  # the turn already in flight is allowed to finish
    final = wait_for_finish(client, run_id)

    assert final["status"] == "cancelled"
    assert final["termination"] == "cancelled"
    assert final["totals"]["turns"] == 1


def test_bad_requests_are_rejected_before_the_run_starts(client: TestClient) -> None:
    duplicate = client.post(
        "/api/runs",
        json={
            "agent_a": {"name": "same", "model": "fast", "system_prompt": "x"},
            "agent_b": {"name": "same", "model": "slow", "system_prompt": "y"},
        },
    )
    assert duplicate.status_code == 400
    assert "distinct names" in duplicate.json()["detail"]

    unknown_model = client.post(
        "/api/runs",
        json={
            "agent_a": {"name": "a", "model": "nope", "system_prompt": "x"},
            "agent_b": {"name": "b", "model": "slow", "system_prompt": "y"},
        },
    )
    assert unknown_model.status_code == 400
    assert "nope" in unknown_model.json()["detail"]

    unknown_opener = client.post(
        "/api/runs",
        json={
            "agent_a": {"name": "a", "model": "fast", "system_prompt": "x"},
            "agent_b": {"name": "b", "model": "slow", "system_prompt": "y"},
            "opening_speaker": "carol",
        },
    )
    assert unknown_opener.status_code == 400


def test_unknown_run_is_404(client: TestClient) -> None:
    assert client.get("/api/runs/deadbeef").status_code == 404
    assert client.post("/api/runs/deadbeef/cancel").status_code == 404


def test_history_lists_and_reopens_saved_runs(client: TestClient) -> None:
    run_id = start(client)["id"]
    wait_for_finish(client, run_id)

    entries = client.get("/api/history").json()
    assert len(entries) == 1
    assert entries[0]["agents"] == ["buyer", "seller"]
    assert entries[0]["models"] == ["fast", "slow"]
    assert entries[0]["termination"] == "deal"
    assert entries[0]["deal_amount"] == "140"

    view = client.get(f"/api/history/{entries[0]['file']}").json()
    assert view["source"] == "saved"
    assert view["status"] == "done"
    assert len(view["turns"]) == 3
    assert view["agents"][0]["system_prompt"] == "you buy"
    assert view["turns"][0]["cost_usd"] == 0.0  # mock models carry no registry prices


def test_history_rejects_unsafe_names(client: TestClient) -> None:
    assert client.get("/api/history/models.yaml").status_code == 400
    assert client.get("/api/history/..%2Fmodels.yaml").status_code in (400, 404)
    assert client.get("/api/history/missing.json").status_code == 404


def test_index_and_static_assets_are_served(client: TestClient) -> None:
    assert "Conversation dashboard" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/styles.css").status_code == 200
