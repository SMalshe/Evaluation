"""Ask a model for JSON and validate it, with a bounded repair loop.

Small local models routinely emit invalid JSON (fences, trailing prose, a
missing field). ``request_json`` sends a request in JSON mode, and on a parse or
schema failure feeds the error back to the model and asks again, up to
``retries`` times, before giving up. Callers decide what to do with a failure
(regex fallback, mark invalid, ...).
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

from pydantic import BaseModel, ValidationError

from .models import Message, ModelClient

T = TypeVar("T", bound=BaseModel)

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


@dataclass
class JsonAttempt(Generic[T]):
    """Outcome of a request_json call."""

    value: T | None  # None iff every attempt failed
    method: str  # "parsed" (first try) | "repaired" (a retry worked) | "failed"
    attempts: int
    prompt_tokens: int
    completion_tokens: int
    raw: str  # the last raw reply, for a downstream fallback


def extract_json_object(text: str) -> str | None:
    """Return the first balanced JSON object in ``text``, or None.

    Tolerates code fences and trailing prose; uses the JSON decoder itself (not
    brace counting) so braces inside strings don't confuse it.
    """
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1)
    start = text.find("{")
    if start == -1:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    return json.dumps(obj)


def request_json(
    client: ModelClient,
    messages: Sequence[Message],
    system: str,
    model_cls: type[T],
    *,
    retries: int = 3,
    temperature: float | None = None,
) -> JsonAttempt[T]:
    """Request a JSON object matching ``model_cls``, repairing up to ``retries`` times."""
    prompt_tokens = 0
    completion_tokens = 0
    convo: list[Message] = list(messages)
    last_raw = ""

    for attempt in range(retries + 1):
        try:
            response = client.chat(convo, system=system, temperature=temperature, json_mode=True)
        except Exception as exc:  # noqa: BLE001
            # Some providers (e.g. Groq) validate JSON mode server-side and return
            # a 400 with the malformed text in the body instead of a reply. Treat
            # that as a failed attempt to repair, not a crash.
            if getattr(exc, "status_code", None) != 400:
                raise
            reply = _failed_generation(exc)
            last_raw = reply or last_raw
            error = "The provider rejected the reply as invalid JSON."
            convo = _repair_turn(convo, reply, error)
            continue

        prompt_tokens += response.prompt_tokens
        completion_tokens += response.completion_tokens
        last_raw = response.text

        candidate = extract_json_object(response.text)
        if candidate is not None:
            try:
                value = model_cls.model_validate_json(candidate)
                method = "parsed" if attempt == 0 else "repaired"
                return JsonAttempt(
                    value, method, attempt + 1, prompt_tokens, completion_tokens, last_raw
                )
            except ValidationError as exc:
                error = str(exc)
        else:
            error = "No JSON object was found in the reply."

        convo = _repair_turn(convo, response.text, error)

    return JsonAttempt(None, "failed", retries + 1, prompt_tokens, completion_tokens, last_raw)


def _repair_turn(convo: list[Message], reply: str, error: str) -> list[Message]:
    """Append the failed reply and a correction request to the conversation."""
    return [
        *convo,
        *([{"role": "assistant", "content": reply}] if reply else []),
        {
            "role": "user",
            "content": (
                f"That reply was not valid.\n{error}\n"
                "Reply again with ONLY the corrected JSON object, no prose or code fences."
            ),
        },
    ]


def _failed_generation(exc: Exception) -> str:
    """Pull the rejected text out of a provider's JSON-validation 400, if present."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and isinstance(error.get("failed_generation"), str):
            return error["failed_generation"]
    return ""
