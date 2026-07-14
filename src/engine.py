"""Two-agent conversation engine.

Agents alternate turns. Each agent sees the transcript from its own
perspective: its own prior messages carry the ``assistant`` role, the
counterpart's carry the ``user`` role, and system prompts never cross.
The opening speaker additionally sees a fixed ``opening_prompt`` as its
first user message (chat APIs require a user message to respond to);
the counterpart never sees that prompt.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel

from .models import Message, ModelClient

DEAL_PATTERN = re.compile(r"\[DEAL\s*\$\s*([0-9][0-9,]*(?:\.[0-9]+)?)\]")
WALK_AWAY_TOKEN = "[WALK_AWAY]"

TerminationReason = Literal["deal", "walk_away", "max_turns", "cancelled"]


@dataclass
class Agent:
    """One conversation participant: a persona bound to a model client."""

    name: str
    system_prompt: str
    client: ModelClient
    temperature: float | None = None  # None = use the client's registry default
    max_tokens: int | None = None  # None = use the client's registry default


class Turn(BaseModel):
    index: int
    speaker: str
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


class AgentInfo(BaseModel):
    """Snapshot of an agent's configuration at conversation time."""

    name: str
    backend: str
    model_name: str
    model_id: str
    system_prompt: str
    temperature: float | None
    max_tokens: int | None


class Transcript(BaseModel):
    turns: list[Turn]
    termination: TerminationReason
    deal_amount: str | None = None  # the X captured from [DEAL $X], if any
    agents: list[AgentInfo]
    opening_speaker: str
    opening_prompt: str
    max_turns: int
    started_at: datetime
    ended_at: datetime


def run_conversation(
    agent_a: Agent,
    agent_b: Agent,
    max_turns: int = 6,
    opening_speaker: str | None = None,
    opening_prompt: str = "You may begin.",
    on_turn: Callable[[Turn], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> Transcript:
    """Alternate turns between two agents until a control token or max_turns.

    Termination: a turn containing ``[DEAL $X]`` ends with reason ``deal``
    (capturing X), ``[WALK_AWAY]`` ends with ``walk_away``, otherwise the
    conversation stops at ``max_turns``.

    ``on_turn`` is called with each completed ``Turn`` as it happens (for live
    observers). ``cancelled`` is polled before each turn; when it returns True
    the conversation stops with termination ``cancelled`` and the turns so far.
    """
    if agent_a.name == agent_b.name:
        raise ValueError("Agents must have distinct names")
    if max_turns < 1:
        raise ValueError("max_turns must be >= 1")

    opener_name = opening_speaker if opening_speaker is not None else agent_a.name
    agents_by_name = {agent_a.name: agent_a, agent_b.name: agent_b}
    if opener_name not in agents_by_name:
        raise ValueError(f"opening_speaker {opener_name!r} is not one of {sorted(agents_by_name)}")
    opener = agents_by_name[opener_name]
    responder = agent_b if opener is agent_a else agent_a

    started_at = datetime.now(UTC)
    history: list[tuple[str, str]] = []  # (speaker name, text) in order
    turns: list[Turn] = []
    termination: TerminationReason = "max_turns"
    deal_amount: str | None = None

    for index in range(max_turns):
        if cancelled is not None and cancelled():
            termination = "cancelled"
            break

        agent = opener if index % 2 == 0 else responder
        messages = _messages_for(agent.name, opener.name, opening_prompt, history)
        response = agent.client.chat(
            messages,
            system=agent.system_prompt,
            temperature=agent.temperature,
            max_tokens=agent.max_tokens,
        )
        turn = Turn(
            index=index,
            speaker=agent.name,
            text=response.text,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            latency_ms=response.latency_ms,
        )
        turns.append(turn)
        history.append((agent.name, response.text))
        if on_turn is not None:
            on_turn(turn)

        deal_match = DEAL_PATTERN.search(response.text)
        if deal_match:
            termination = "deal"
            deal_amount = deal_match.group(1)
            break
        if WALK_AWAY_TOKEN in response.text:
            termination = "walk_away"
            break

    return Transcript(
        turns=turns,
        termination=termination,
        deal_amount=deal_amount,
        agents=[_agent_info(opener), _agent_info(responder)],
        opening_speaker=opener.name,
        opening_prompt=opening_prompt,
        max_turns=max_turns,
        started_at=started_at,
        ended_at=datetime.now(UTC),
    )


def _messages_for(
    agent_name: str,
    opener_name: str,
    opening_prompt: str,
    history: list[tuple[str, str]],
) -> list[Message]:
    """Render the shared history from one agent's perspective."""
    messages: list[Message] = []
    if agent_name == opener_name:
        messages.append({"role": "user", "content": opening_prompt})
    for speaker, text in history:
        role = "assistant" if speaker == agent_name else "user"
        messages.append({"role": role, "content": text})
    return messages


def _agent_info(agent: Agent) -> AgentInfo:
    config = agent.client.config
    return AgentInfo(
        name=agent.name,
        backend=config.backend,
        model_name=config.name,
        model_id=config.model_id,
        system_prompt=agent.system_prompt,
        temperature=agent.temperature if agent.temperature is not None else config.temperature,
        max_tokens=agent.max_tokens if agent.max_tokens is not None else config.max_tokens,
    )
