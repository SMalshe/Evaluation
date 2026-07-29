"""Unified client interface over multiple LLM chat APIs.

Backends:
  - ``openai_compat``: any OpenAI-compatible chat-completions endpoint
    (OpenAI itself, Groq, Google Gemini's compatibility layer, ...),
    selected via ``base_url`` + ``api_key_env``.
  - ``anthropic``: the native Anthropic SDK.

Model settings live in a YAML registry (see ``models.yaml``); use
``get_client("short-name")`` to build a configured client.
"""

from __future__ import annotations

import os
import random
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import anthropic
import openai
import yaml
from dotenv import load_dotenv

Message = dict[str, str]
"""A chat message: ``{"role": "user" | "assistant", "content": "..."}``."""


class RegistryError(RuntimeError):
    """The model registry is missing, malformed, or lacks the requested entry."""


class MissingAPIKeyError(RuntimeError):
    """The environment variable holding a required API key is not set."""


@dataclass(frozen=True)
class RetryPolicy:
    """Retry/backoff behavior for transient API failures (429, 5xx, timeouts)."""

    max_retries: int = 4
    initial_backoff_s: float = 1.0
    backoff_multiplier: float = 2.0
    max_backoff_s: float = 30.0
    jitter_s: float = 0.5
    timeout_s: float = 120.0  # hard wall-clock timeout per API call


@dataclass(frozen=True)
class ModelConfig:
    """One registry entry: which backend/model to call and its defaults."""

    name: str
    backend: str
    model_id: str
    # None = the endpoint needs no credential (a local OpenAI-compatible server
    # such as Ollama/llama.cpp/vLLM). Such entries are always "available".
    api_key_env: str | None = None
    base_url: str | None = None
    temperature: float | None = None  # None = omit the parameter from requests
    max_tokens: int = 1024
    price_per_mtok_in: float = 0.0
    price_per_mtok_out: float = 0.0
    # openai_compat only: the GPT-5.x family rejects "max_tokens" and requires
    # "max_completion_tokens" instead.
    max_tokens_param: str = "max_tokens"
    extra_request: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    raw: dict[str, Any]


class ModelClient(ABC):
    """A chat client bound to one model configuration."""

    # Exception types worth retrying, set per backend.
    _RETRYABLE: ClassVar[tuple[type[Exception], ...]] = ()

    def __init__(self, config: ModelConfig, retry: RetryPolicy | None = None) -> None:
        self.config = config
        self.retry = retry or RetryPolicy()

    @abstractmethod
    def chat(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> ModelResponse:
        """Send one chat request. ``temperature``/``max_tokens`` of None fall back
        to the registry defaults; a registry temperature of None omits the
        parameter entirely (required by models that reject sampling params)."""

    def _resolve_temperature(self, temperature: float | None) -> float | None:
        return temperature if temperature is not None else self.config.temperature

    def _resolve_max_tokens(self, max_tokens: int | None) -> int:
        return max_tokens if max_tokens is not None else self.config.max_tokens

    def _call_with_retries(self, send: Callable[[], ModelResponse]) -> ModelResponse:
        delay = self.retry.initial_backoff_s
        for attempt in range(self.retry.max_retries + 1):
            try:
                return send()
            except self._RETRYABLE as exc:
                if attempt == self.retry.max_retries:
                    raise
                wait = _retry_after_seconds(exc)
                if wait is None:
                    wait = min(delay, self.retry.max_backoff_s) + random.uniform(
                        0, self.retry.jitter_s
                    )
                    delay *= self.retry.backoff_multiplier
                time.sleep(wait)
        raise AssertionError("unreachable")


def _retry_after_seconds(exc: Exception) -> float | None:
    """Honor a Retry-After header (seconds) if the exception carries one."""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is None:
        return None
    value = headers.get("retry-after")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _require_env(var: str) -> str:
    value = os.environ.get(var)
    if not value:
        raise MissingAPIKeyError(
            f"Environment variable {var} is not set. Add it to your environment or .env file."
        )
    return value


# Local OpenAI-compatible servers ignore the Authorization header, but the SDK
# refuses to construct a client without some non-empty key.
_NO_AUTH_PLACEHOLDER = "not-required"


def _resolve_api_key(config: ModelConfig) -> str:
    """The credential for an entry, or a placeholder when it needs none."""
    if config.api_key_env is None:
        return _NO_AUTH_PLACEHOLDER
    return _require_env(config.api_key_env)


class OpenAICompatClient(ModelClient):
    """Client for any OpenAI-compatible chat-completions endpoint."""

    _RETRYABLE = (
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.APIConnectionError,
        openai.InternalServerError,
    )

    def __init__(self, config: ModelConfig, retry: RetryPolicy | None = None) -> None:
        super().__init__(config, retry)
        self._client = openai.OpenAI(
            api_key=_resolve_api_key(config),
            base_url=config.base_url,
            max_retries=0,  # retries are owned by _call_with_retries
        )

    def chat(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> ModelResponse:
        request_messages: list[Message] = []
        if system:
            request_messages.append({"role": "system", "content": system})
        request_messages.extend(messages)

        payload: dict[str, Any] = {
            "model": self.config.model_id,
            "messages": request_messages,
            self.config.max_tokens_param: self._resolve_max_tokens(max_tokens),
        }
        resolved_temperature = self._resolve_temperature(temperature)
        if resolved_temperature is not None:
            payload["temperature"] = resolved_temperature
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        payload.update(self.config.extra_request)

        def send() -> ModelResponse:
            started = time.monotonic()
            response = self._client.chat.completions.create(timeout=self.retry.timeout_s, **payload)
            latency_ms = (time.monotonic() - started) * 1000
            usage = response.usage
            return ModelResponse(
                text=response.choices[0].message.content or "",
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                latency_ms=latency_ms,
                raw=response.model_dump(),
            )

        return self._call_with_retries(send)


_JSON_MODE_INSTRUCTION = (
    "Respond with a single valid JSON object and nothing else - no prose, no code fences."
)


class AnthropicClient(ModelClient):
    """Client for the native Anthropic Messages API.

    The API has no response_format parameter, so ``json_mode`` is implemented
    as a strict system-prompt instruction.
    """

    _RETRYABLE = (
        anthropic.RateLimitError,
        anthropic.APITimeoutError,
        anthropic.APIConnectionError,
        anthropic.InternalServerError,
    )

    def __init__(self, config: ModelConfig, retry: RetryPolicy | None = None) -> None:
        super().__init__(config, retry)
        self._client = anthropic.Anthropic(
            api_key=_resolve_api_key(config),
            max_retries=0,  # retries are owned by _call_with_retries
        )

    def chat(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> ModelResponse:
        system_prompt = system
        if json_mode:
            system_prompt = (
                f"{system}\n\n{_JSON_MODE_INSTRUCTION}" if system else _JSON_MODE_INSTRUCTION
            )

        payload: dict[str, Any] = {
            "model": self.config.model_id,
            "messages": list(messages),
            "max_tokens": self._resolve_max_tokens(max_tokens),
        }
        if system_prompt:
            payload["system"] = system_prompt
        resolved_temperature = self._resolve_temperature(temperature)
        if resolved_temperature is not None:
            payload["temperature"] = resolved_temperature
        payload.update(self.config.extra_request)

        def send() -> ModelResponse:
            started = time.monotonic()
            response = self._client.messages.create(timeout=self.retry.timeout_s, **payload)
            latency_ms = (time.monotonic() - started) * 1000
            text = "".join(block.text for block in response.content if block.type == "text")
            return ModelResponse(
                text=text,
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                latency_ms=latency_ms,
                raw=response.model_dump(),
            )

        return self._call_with_retries(send)


_BACKENDS: dict[str, type[ModelClient]] = {
    "openai_compat": OpenAICompatClient,
    "anthropic": AnthropicClient,
}

_CONFIG_FIELDS = {
    "backend",
    "model_id",
    "api_key_env",
    "base_url",
    "temperature",
    "max_tokens",
    "price_per_mtok_in",
    "price_per_mtok_out",
    "max_tokens_param",
    "extra_request",
}


def load_registry(path: str | Path = "models.yaml") -> dict[str, ModelConfig]:
    """Parse the YAML registry into validated ``ModelConfig`` entries."""
    registry_path = Path(path)
    if not registry_path.is_file():
        raise RegistryError(f"Model registry not found: {registry_path}")
    data = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("models"), dict):
        raise RegistryError(f"{registry_path} must contain a top-level 'models' mapping")

    registry: dict[str, ModelConfig] = {}
    for name, entry in data["models"].items():
        if not isinstance(entry, dict):
            raise RegistryError(f"Registry entry {name!r} must be a mapping")
        unknown = set(entry) - _CONFIG_FIELDS
        if unknown:
            raise RegistryError(f"Registry entry {name!r} has unknown fields: {sorted(unknown)}")
        missing = {"backend", "model_id"} - set(entry)
        if missing:
            raise RegistryError(f"Registry entry {name!r} is missing fields: {sorted(missing)}")
        registry[name] = ModelConfig(name=name, **entry)
    return registry


def get_client(
    name: str,
    registry_path: str | Path = "models.yaml",
    retry: RetryPolicy | None = None,
) -> ModelClient:
    """Build a configured client for a registry entry by short name."""
    load_dotenv(override=False)
    registry = load_registry(registry_path)
    if name not in registry:
        raise RegistryError(f"Unknown model {name!r}; available: {', '.join(sorted(registry))}")
    config = registry[name]
    backend = _BACKENDS.get(config.backend)
    if backend is None:
        raise RegistryError(
            f"Entry {name!r} has unknown backend {config.backend!r}; "
            f"expected one of {sorted(_BACKENDS)}"
        )
    return backend(config, retry=retry)
