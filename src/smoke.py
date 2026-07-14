"""CLI smoke test: a short two-agent haggle over a used bicycle.

Usage:
    python -m src.smoke --model-a claude-sonnet --model-b llama-70b
"""

from __future__ import annotations

import argparse
import textwrap
from collections.abc import Sequence

from dotenv import load_dotenv

from .engine import Agent, Transcript, run_conversation
from .models import get_client
from .persistence import save_transcript

BUYER_SYSTEM = """\
You are negotiating to buy the seller's used road bicycle, listed at $200.
Your private budget cap is $150. Never reveal your cap; try to pay as little
as possible. Keep every reply to at most 3 sentences and stay in character as
the buyer - do not narrate or break the fourth wall.

Ending the negotiation:
- If and only if both sides have clearly agreed on a final price X, end your
  message with the exact token [DEAL $X] (for example [DEAL $140]).
- If agreement is clearly impossible, end your message with [WALK_AWAY].
Otherwise, keep negotiating and do not use either token."""

SELLER_SYSTEM = """\
You are selling your used road bicycle, listed at $200. Your private
walk-away minimum is $120. Never reveal your minimum; try to get as much as
possible. Keep every reply to at most 3 sentences and stay in character as
the seller - do not narrate or break the fourth wall.

Ending the negotiation:
- If and only if both sides have clearly agreed on a final price X, end your
  message with the exact token [DEAL $X] (for example [DEAL $140]).
- If agreement is clearly impossible, end your message with [WALK_AWAY].
Otherwise, keep negotiating and do not use either token."""

OPENING_PROMPT = (
    "You are messaging the seller about the road bicycle they listed for $200. "
    "Open the negotiation."
)


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv(override=False)
    parser = argparse.ArgumentParser(
        prog="python -m src.smoke",
        description="Run a short two-agent haggle over a used bicycle and pretty-print it.",
    )
    parser.add_argument("--model-a", required=True, help="registry name for the buyer")
    parser.add_argument("--model-b", required=True, help="registry name for the seller")
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--registry", default="models.yaml", help="path to the model registry")
    parser.add_argument("--runs-dir", default="runs", help="directory for saved transcripts")
    args = parser.parse_args(argv)

    buyer = Agent(
        name="buyer",
        system_prompt=BUYER_SYSTEM,
        client=get_client(args.model_a, registry_path=args.registry),
    )
    seller = Agent(
        name="seller",
        system_prompt=SELLER_SYSTEM,
        client=get_client(args.model_b, registry_path=args.registry),
    )

    print(f"buyer:  {args.model_a} ({buyer.client.config.model_id})")
    print(f"seller: {args.model_b} ({seller.client.config.model_id})")
    print(f"running up to {args.max_turns} turns...\n")

    transcript = run_conversation(
        buyer,
        seller,
        max_turns=args.max_turns,
        opening_speaker="buyer",
        opening_prompt=OPENING_PROMPT,
    )

    _print_transcript(transcript, {"buyer": buyer, "seller": seller})

    path = save_transcript(transcript, runs_dir=args.runs_dir)
    print(f"\ntranscript saved to {path}")
    return 0


def _print_transcript(transcript: Transcript, agents: dict[str, Agent]) -> None:
    total_cost = 0.0
    for turn in transcript.turns:
        config = agents[turn.speaker].client.config
        total_cost += (
            turn.prompt_tokens * config.price_per_mtok_in
            + turn.completion_tokens * config.price_per_mtok_out
        ) / 1_000_000
        header = (
            f"[{turn.index + 1}] {turn.speaker} ({config.model_id})  "
            f"{turn.latency_ms / 1000:.1f}s  "
            f"{turn.prompt_tokens}->{turn.completion_tokens} tok"
        )
        print(header)
        print(textwrap.fill(turn.text, width=88, initial_indent="    ", subsequent_indent="    "))
        print()

    summary = f"termination: {transcript.termination}"
    if transcript.deal_amount is not None:
        summary += f" (${transcript.deal_amount})"
    print(summary)
    prompt_total = sum(t.prompt_tokens for t in transcript.turns)
    completion_total = sum(t.completion_tokens for t in transcript.turns)
    print(
        f"turns: {len(transcript.turns)}  tokens: {prompt_total} in / {completion_total} out  "
        f"est. cost: ${total_cost:.4f}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
