"""Engine tests using a canned-response mock backend (no network)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from src.engine import Agent, run_conversation
from src.models import Message, ModelClient, ModelConfig, ModelResponse


class MockBackend(ModelClient):
    """Returns canned replies in order and records every call it receives."""

    def __init__(self, name: str, replies: Sequence[str]) -> None:
        super().__init__(
            ModelConfig(
                name=name,
                backend="mock",
                model_id=f"mock-{name}",
                api_key_env="UNUSED",
            )
        )
        self._replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> ModelResponse:
        self.calls.append(
            {
                "messages": [dict(m) for m in messages],
                "system": system,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "json_mode": json_mode,
            }
        )
        index = min(len(self.calls) - 1, len(self._replies) - 1)
        return ModelResponse(
            text=self._replies[index],
            prompt_tokens=10,
            completion_tokens=5,
            latency_ms=1.0,
            raw={},
        )


def make_agent(name: str, replies: Sequence[str]) -> Agent:
    return Agent(
        name=name,
        system_prompt=f"secret system prompt for {name}",
        client=MockBackend(name, replies),
    )


def test_turns_alternate_until_max_turns() -> None:
    alice = make_agent("alice", ["a1", "a2"])
    bob = make_agent("bob", ["b1", "b2"])

    transcript = run_conversation(alice, bob, max_turns=4)

    assert [t.speaker for t in transcript.turns] == ["alice", "bob", "alice", "bob"]
    assert transcript.termination == "max_turns"
    assert len(transcript.turns) == 4
    assert transcript.opening_speaker == "alice"


def test_opening_speaker_selects_who_goes_first() -> None:
    alice = make_agent("alice", ["a1"])
    bob = make_agent("bob", ["b1", "b2"])

    transcript = run_conversation(alice, bob, max_turns=3, opening_speaker="bob")

    assert [t.speaker for t in transcript.turns] == ["bob", "alice", "bob"]


def test_perspective_flips_roles_per_agent() -> None:
    alice = make_agent("alice", ["a1", "a2"])
    bob = make_agent("bob", ["b1"])
    alice_client = alice.client
    bob_client = bob.client
    assert isinstance(alice_client, MockBackend) and isinstance(bob_client, MockBackend)

    run_conversation(alice, bob, max_turns=3, opening_prompt="GO")

    # Opener's first call: only the opening prompt, as a user message.
    assert alice_client.calls[0]["messages"] == [{"role": "user", "content": "GO"}]
    # Counterpart's first call: the opener's text as a user message; it never
    # sees the opening prompt.
    assert bob_client.calls[0]["messages"] == [{"role": "user", "content": "a1"}]
    # Opener's second call: own message as assistant, counterpart's as user.
    assert alice_client.calls[1]["messages"] == [
        {"role": "user", "content": "GO"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "b1"},
    ]


def test_system_prompts_never_cross() -> None:
    alice = make_agent("alice", ["a1", "a2"])
    bob = make_agent("bob", ["b1", "b2"])

    run_conversation(alice, bob, max_turns=4)

    for client, other in ((alice.client, bob), (bob.client, alice)):
        assert isinstance(client, MockBackend)
        for call in client.calls:
            assert call["system"] == f"secret system prompt for {client.config.name}"
            for message in call["messages"]:
                assert other.system_prompt not in message["content"]


def test_deal_token_terminates_and_captures_amount() -> None:
    alice = make_agent("alice", ["opening offer"])
    bob = make_agent("bob", ["Fine, agreed. [DEAL $1,450.50]"])

    transcript = run_conversation(alice, bob, max_turns=6)

    assert transcript.termination == "deal"
    assert transcript.deal_amount == "1,450.50"
    assert len(transcript.turns) == 2  # stops immediately at the token


def test_walk_away_token_terminates() -> None:
    alice = make_agent("alice", ["offer", "counter"])
    bob = make_agent("bob", ["No deal. [WALK_AWAY]"])

    transcript = run_conversation(alice, bob, max_turns=6)

    assert transcript.termination == "walk_away"
    assert transcript.deal_amount is None
    assert len(transcript.turns) == 2


def test_transcript_records_config_and_usage() -> None:
    alice = make_agent("alice", ["a1"])
    alice.temperature = 0.3
    bob = make_agent("bob", ["b1"])

    transcript = run_conversation(alice, bob, max_turns=2)

    by_name = {info.name: info for info in transcript.agents}
    assert by_name["alice"].model_id == "mock-alice"
    assert by_name["alice"].temperature == 0.3
    assert by_name["bob"].backend == "mock"
    assert all(t.prompt_tokens == 10 and t.completion_tokens == 5 for t in transcript.turns)
    assert transcript.ended_at >= transcript.started_at


def test_agent_gen_params_are_passed_to_client() -> None:
    alice = make_agent("alice", ["a1"])
    alice.temperature = 0.1
    alice.max_tokens = 77
    bob = make_agent("bob", ["b1"])

    run_conversation(alice, bob, max_turns=2)

    alice_client = alice.client
    assert isinstance(alice_client, MockBackend)
    assert alice_client.calls[0]["temperature"] == 0.1
    assert alice_client.calls[0]["max_tokens"] == 77


def test_on_turn_fires_for_each_turn_in_order() -> None:
    alice = make_agent("alice", ["a1", "a2"])
    bob = make_agent("bob", ["b1", "b2"])
    seen: list[tuple[int, str]] = []

    transcript = run_conversation(
        alice, bob, max_turns=4, on_turn=lambda turn: seen.append((turn.index, turn.speaker))
    )

    assert seen == [(0, "alice"), (1, "bob"), (2, "alice"), (3, "bob")]
    assert [(t.index, t.speaker) for t in transcript.turns] == seen


def test_cancelled_stops_between_turns_and_keeps_prior_turns() -> None:
    alice = make_agent("alice", ["a1", "a2"])
    bob = make_agent("bob", ["b1", "b2"])
    stop_after = 2
    turns_done = 0

    def on_turn(_: object) -> None:
        nonlocal turns_done
        turns_done += 1

    transcript = run_conversation(
        alice,
        bob,
        max_turns=6,
        on_turn=on_turn,
        cancelled=lambda: turns_done >= stop_after,
    )

    assert transcript.termination == "cancelled"
    assert len(transcript.turns) == stop_after  # the in-flight turn still completes
    assert [t.speaker for t in transcript.turns] == ["alice", "bob"]


def test_cancelled_before_first_turn_yields_no_turns() -> None:
    alice = make_agent("alice", ["a1"])
    bob = make_agent("bob", ["b1"])

    transcript = run_conversation(alice, bob, max_turns=4, cancelled=lambda: True)

    assert transcript.termination == "cancelled"
    assert transcript.turns == []


def test_rejects_duplicate_names_and_unknown_opener() -> None:
    alice = make_agent("alice", ["a1"])
    alice_two = make_agent("alice", ["a2"])
    bob = make_agent("bob", ["b1"])

    with pytest.raises(ValueError, match="distinct names"):
        run_conversation(alice, alice_two, max_turns=2)
    with pytest.raises(ValueError, match="opening_speaker"):
        run_conversation(alice, bob, max_turns=2, opening_speaker="carol")
