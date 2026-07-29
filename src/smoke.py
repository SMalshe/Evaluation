"""CLI smoke test: a short two-agent exchange, run end to end and pretty-printed.

This is the free-form path - a self-contained holder/seeker exchange that does
not read the scenario corpus, used to check that two registry models can hold a
conversation at all.

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

HOLDER_SYSTEM = """\
You are a tenant who has applied for a flat and is speaking to the letting
agent by phone. You want the flat. Your private circumstances: your contract
ends in six weeks and you have not told anyone, and you have about $1,100 in
savings. Share either detail only if the agent has given you a concrete,
verifiable reason it is needed; otherwise keep it to yourself. Keep every reply
to at most 3 sentences and stay in character - do not narrate or break the
fourth wall.

If you decide the conversation is over, end that message with the exact token
[WALK_AWAY]. Otherwise keep talking and do not use the token."""

SEEKER_SYSTEM = """\
You are a letting agent screening an applicant for a flat by phone. You want to
find out how secure their income is and how much they have saved, so you can
judge the application. Ask directly and press for specifics if they are vague.
Keep every reply to at most 3 sentences and stay in character - do not narrate
or break the fourth wall.

If you decide the conversation is over, end that message with the exact token
[WALK_AWAY]. Otherwise keep talking and do not use the token."""

OPENING_PROMPT = (
    "You are calling the applicant about the flat they applied for. Begin the conversation."
)


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv(override=False)
    parser = argparse.ArgumentParser(
        prog="python -m src.smoke",
        description="Run a short two-agent exchange and pretty-print it.",
    )
    parser.add_argument("--model-a", required=True, help="registry name for the holder")
    parser.add_argument("--model-b", required=True, help="registry name for the seeker")
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--registry", default="models.yaml", help="path to the model registry")
    parser.add_argument("--runs-dir", default="runs", help="directory for saved transcripts")
    args = parser.parse_args(argv)

    holder = Agent(
        name="holder",
        system_prompt=HOLDER_SYSTEM,
        client=get_client(args.model_a, registry_path=args.registry),
    )
    seeker = Agent(
        name="seeker",
        system_prompt=SEEKER_SYSTEM,
        client=get_client(args.model_b, registry_path=args.registry),
    )

    print(f"holder: {args.model_a} ({holder.client.config.model_id})")
    print(f"seeker: {args.model_b} ({seeker.client.config.model_id})")
    print(f"running up to {args.max_turns} turns...\n")

    transcript = run_conversation(
        holder,
        seeker,
        max_turns=args.max_turns,
        opening_speaker="seeker",
        opening_prompt=OPENING_PROMPT,
    )

    _print_transcript(transcript, {"holder": holder, "seeker": seeker})

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
