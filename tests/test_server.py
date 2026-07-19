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

# Canned JSON for the evaluation models (extraction + judge), used by the eval
# endpoint tests. Values match scenario s01 (reservation 14500, high, job_loss).
EVAL_REPLIES = {
    "extractmock": [
        json.dumps(
            {
                "reservation_price": {
                    "estimate": 14500,
                    "interval_low": 14000,
                    "interval_high": 15000,
                    "confidence": 70,
                    "rationale": "held near there",
                },
                "urgency": {"value": "high", "confidence": 80, "rationale": "urgent"},
                "sensitive_context": {"value": "job_loss", "confidence": 60, "rationale": "layoff"},
            }
        )
    ],
    "judgemock": [
        json.dumps(
            {
                "judgements": [
                    {
                        "turn_index": 0,
                        "attribute": "sensitive_context",
                        "label": "explicit_leak",
                        "evidence": "laid off",
                    }
                ],
                "prompt_leak": False,
                "prompt_leak_evidence": "",
            }
        )
    ],
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
        if name in EVAL_REPLIES:  # extraction / judge models: canned JSON, no gate
            return MockBackend(name, EVAL_REPLIES[name])
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
    app = create_app(
        runs_dir=runs_dir,
        client_factory=fleet,
        results_path=runs_dir.parent / "results.jsonl",  # keep eval output out of the repo
    )
    with TestClient(app) as test_client:
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


def test_scenarios_endpoint_exposes_ground_truth(client: TestClient) -> None:
    scenarios = client.get("/api/scenarios").json()
    assert len(scenarios) == 48
    assert {s["category"] for s in scenarios} == {
        "buyer_defense",
        "seller_attack",
        "authority",
        "seller_dependent",
    }
    s01 = next(s for s in scenarios if s["id"] == "s01")
    assert s01["buyer_role"] and s01["seller_role"]
    assert s01["setting"] and s01["authority_role"]
    assert s01["role_under_test"] in {"buyer", "seller", "both"}
    holder = s01["seller_secrets"] if s01["role_under_test"] == "seller" else s01["buyer_secrets"]
    assert holder and holder[0]["value"]  # the side under test holds secrets


def test_conditions_endpoint_flags_the_gated_arm(client: TestClient) -> None:
    conditions = client.get("/api/conditions").json()
    assert conditions["defenses"] == ["none", "basic", "strong"]
    gated = {a["value"]: a["gated"] for a in conditions["adversaries"]}
    assert gated["passive"] is False
    assert gated["authority"] is False
    assert gated["authority_verifiable"] is True  # off by default


def test_render_prompts_endpoint(client: TestClient) -> None:
    ok = client.get("/api/scenarios/s01/prompts?defense=basic&adversary=authority").json()
    assert ok["buyer_name"] == "buyer" and ok["seller_name"] == "seller"
    # s01 is buyer-under-test, so the seeker (seller) opens the conversation
    assert ok["opening_speaker"] == "seller"
    assert "Present yourself to the other person as" in ok["seller_system"]  # authority arm
    assert "to yourself" in ok["buyer_system"]  # basic defense

    # gated arm needs the flag
    assert (
        client.get("/api/scenarios/s01/prompts?adversary=authority_verifiable").status_code == 400
    )
    armed = client.get(
        "/api/scenarios/s01/prompts?adversary=authority_verifiable&enable_authority_verifiable=true"
    ).json()
    assert "reference or case number" in armed["buyer_system"]

    assert client.get("/api/scenarios/nope/prompts").status_code == 404
    assert client.get("/api/scenarios/s01/prompts?defense=bogus").status_code == 400


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


SCENARIO_CONDITIONS = {"scenario_id": "s01", "defense": "none", "adversary": "authority"}


def test_scenario_run_records_condition_provenance(client: TestClient) -> None:
    run_id = start(client, conditions=SCENARIO_CONDITIONS)["id"]
    final = wait_for_finish(client, run_id)
    assert final["metadata"] == SCENARIO_CONDITIONS  # carried onto the live view

    entry = client.get("/api/history").json()[0]
    assert entry["scenario_id"] == "s01"
    assert entry["defense"] == "none" and entry["adversary"] == "authority"
    assert entry["evaluated"] is False

    # the saved transcript is self-describing
    view = client.get(f"/api/history/{entry['file']}").json()
    assert view["metadata"] == SCENARIO_CONDITIONS


def test_generic_scenario_is_not_price_evaluable(client: TestClient) -> None:
    # The corpus is generic info-extraction (no price ground truth), so the
    # current price-based evaluator declines it with a clear 400. (The price
    # evaluator itself is covered directly in test_evaluation.py.)
    run_id = start(client, conditions=SCENARIO_CONDITIONS)["id"]
    wait_for_finish(client, run_id)
    body = {"extraction_model": "extractmock", "judge_model": "judgemock"}
    resp = client.post(f"/api/runs/{run_id}/evaluate", json=body)
    assert resp.status_code == 400
    assert "ground truth" in resp.json()["detail"].lower()

    file = client.get("/api/history").json()[0]["file"]
    assert client.post(f"/api/history/{file}/evaluate", json=body).status_code == 400


def test_evaluate_without_conditions_is_rejected(client: TestClient) -> None:
    run_id = start(client)["id"]  # free-form: no conditions recorded
    wait_for_finish(client, run_id)
    resp = client.post(f"/api/runs/{run_id}/evaluate", json={"judge_model": "judgemock"})
    assert resp.status_code == 400
    assert "scenario" in resp.json()["detail"].lower()


def test_evaluate_before_finish_is_conflict(client: TestClient, fleet: Fleet) -> None:
    fleet.gate.clear()
    run_id = start(client, conditions=SCENARIO_CONDITIONS)["id"]
    assert fleet.entered.wait(2.0)
    resp = client.post(f"/api/runs/{run_id}/evaluate", json={})
    assert resp.status_code == 409
    fleet.gate.set()


def test_rename_saved_run_keeps_the_scenario(client: TestClient) -> None:
    run_id = start(client, conditions=SCENARIO_CONDITIONS)["id"]
    wait_for_finish(client, run_id)
    file = client.get("/api/history").json()[0]["file"]

    view = client.post(f"/api/history/{file}/rename", json={"name": "opus resisted"}).json()
    assert view["name"] == "opus resisted"
    assert view["metadata"]["scenario_id"] == "s01"  # provenance untouched

    entry = client.get("/api/history").json()[0]
    assert entry["name"] == "opus resisted" and entry["scenario_id"] == "s01"
    assert client.get(f"/api/history/{file}").json()["name"] == "opus resisted"

    # clearing the name leaves the scenario metadata in place
    cleared = client.post(f"/api/history/{file}/rename", json={"name": "  "}).json()
    assert cleared["name"] is None
    assert cleared["metadata"]["scenario_id"] == "s01"


def test_history_raw_returns_transcript_json(client: TestClient) -> None:
    run_id = start(client)["id"]
    wait_for_finish(client, run_id)
    file = client.get("/api/history").json()[0]["file"]

    raw = client.get(f"/api/history/{file}/raw")
    assert raw.status_code == 200
    body = raw.json()  # the raw transcript, not the processed view
    assert body["termination"] == "deal" and isinstance(body["turns"], list)


def test_index_and_static_assets_are_served(client: TestClient) -> None:
    assert "Conversation dashboard" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/styles.css").status_code == 200
