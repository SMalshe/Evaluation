"""Live connectivity check: one tiny request through every registry entry.

Not part of the pytest suite - it spends (fractions of a cent of) real API
credit. Run from the repository root:

    uv run python scripts/live_check.py [--registry models.yaml] [--json-mode/--no-json-mode]

Entries whose API key env var is unset are reported as SKIP. The json-mode
column matters for the Gemini entries in particular: their OpenAI-compatible
endpoint is a beta layer, and if json mode fails there the fallback plan is a
native "google" backend via the google-genai SDK.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import MissingAPIKeyError, RetryPolicy, get_client, load_registry  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--registry", default="models.yaml")
    parser.add_argument(
        "--json-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also test response_format json_object per entry",
    )
    args = parser.parse_args()

    registry = load_registry(args.registry)
    retry = RetryPolicy(max_retries=1, timeout_s=60.0)

    width = max(len(name) for name in registry) + 2
    print(f"{'model':<{width}} {'backend':<14} {'plain':<28} json")
    print("-" * (width + 60))

    failures = 0
    for name, config in registry.items():
        try:
            client = get_client(name, registry_path=args.registry, retry=retry)
        except MissingAPIKeyError:
            print(f"{name:<{width}} {config.backend:<14} SKIP (no {config.api_key_env})")
            continue

        plain = _check_plain(client)
        line = f"{name:<{width}} {config.backend:<14} {plain:<28}"
        if args.json_mode:
            json_result = _check_json(client)
            line += f" {json_result}"
            if json_result.startswith("FAIL"):
                failures += 1
        print(line)
        if plain.startswith("FAIL"):
            failures += 1

    if failures:
        print(f"\n{failures} check(s) failed")
        return 1
    print("\nall reachable entries passed")
    return 0


def _check_plain(client) -> str:
    try:
        started = time.monotonic()
        response = client.chat(
            [{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=None,  # registry default; reasoning models need headroom
        )
        elapsed = time.monotonic() - started
        if not response.text.strip():
            return "FAIL (empty response)"
        return f"PASS ({elapsed:.1f}s)"
    except Exception as exc:  # noqa: BLE001 - report, don't crash the sweep
        return f"FAIL ({_summarize(exc)})"


def _check_json(client) -> str:
    try:
        response = client.chat(
            [
                {
                    "role": "user",
                    "content": 'Return a JSON object exactly like {"ok": true}.',
                }
            ],
            json_mode=True,
        )
        parsed = json.loads(response.text)
        if not isinstance(parsed, dict):
            return "FAIL (not a JSON object)"
        return "PASS"
    except json.JSONDecodeError:
        return "FAIL (unparseable JSON)"
    except Exception as exc:  # noqa: BLE001
        return f"FAIL ({_summarize(exc)})"


def _summarize(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text if len(text) <= 80 else text[:77] + "..."


if __name__ == "__main__":
    raise SystemExit(main())
