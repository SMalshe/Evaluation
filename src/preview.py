"""CLI to render (and optionally run) a scenario under chosen conditions.

    python -m src.preview --scenario s01 --defense basic --adversary rapport
    python -m src.preview --scenario s01 --adversary pressure --run \
        --model-a claude-sonnet --model-b llama-70b

Without ``--run`` it prints the two rendered system prompts. With ``--run`` it
executes one live conversation between the chosen models and prints/saves it.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from dotenv import load_dotenv

from .engine import Agent, run_conversation
from .models import get_client
from .persistence import save_transcript
from .prompts import (
    SCENARIO_OPENING_PROMPT,
    AdversaryStrategy,
    DefenseCondition,
    PromptConfig,
    available_adversaries,
    opening_speaker,
    render_pair,
)
from .scenarios import ScenarioError, load_scenario, scenario_ids
from .smoke import _print_transcript


def _rule(title: str) -> str:
    return f"\n{'=' * 78}\n{title}\n{'=' * 78}"


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv(override=False)
    parser = argparse.ArgumentParser(
        prog="python -m src.preview",
        description="Render a scenario's holder/seeker system prompts under chosen conditions.",
    )
    parser.add_argument("--scenario", required=True, help="scenario id, e.g. s01")
    parser.add_argument("--scenarios-dir", default="scenarios", help="scenario directory")
    parser.add_argument(
        "--defense",
        default=DefenseCondition.none.value,
        choices=[d.value for d in DefenseCondition],
        help="holder defense condition",
    )
    parser.add_argument(
        "--adversary",
        default=AdversaryStrategy.passive.value,
        choices=[a.value for a in AdversaryStrategy],
        help="seeker adversary strategy",
    )
    parser.add_argument(
        "--enable-authority-verifiable",
        action="store_true",
        help="allow the gated authority_verifiable defense arm",
    )
    parser.add_argument("--run", action="store_true", help="execute one live conversation")
    parser.add_argument("--model-a", help="holder model (registry name); required with --run")
    parser.add_argument("--model-b", help="seeker model (registry name); required with --run")
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--registry", default="models.yaml", help="path to the model registry")
    parser.add_argument("--runs-dir", default="runs", help="directory for saved transcripts")
    args = parser.parse_args(argv)

    config = PromptConfig(enable_authority_verifiable=args.enable_authority_verifiable)
    defense = DefenseCondition(args.defense)
    adversary = AdversaryStrategy(args.adversary)
    if adversary not in available_adversaries(config):
        parser.error(
            f"adversary '{adversary.value}' is gated; pass --enable-authority-verifiable to use it"
        )

    try:
        scenario = load_scenario(args.scenario, scenarios_dir=args.scenarios_dir)
    except ScenarioError as exc:
        available = ", ".join(scenario_ids(args.scenarios_dir)) or "(none found)"
        parser.error(f"{exc}\nAvailable scenarios: {available}")

    holder_system, seeker_system = render_pair(scenario, defense, adversary, config)

    print(_rule(f"SCENARIO {scenario.id}: {scenario.title}"))
    print(f"setting:   {scenario.setting}")
    print(f"under test: {scenario.role_under_test.value}   category: {scenario.category.value}")
    print(f"defense:   {defense.value}    adversary: {adversary.value}")
    print(_rule("HOLDER SYSTEM PROMPT"))
    print(holder_system)
    print(_rule("SEEKER SYSTEM PROMPT"))
    print(seeker_system)

    if not args.run:
        return 0

    if not args.model_a or not args.model_b:
        parser.error("--run requires --model-a (holder) and --model-b (seeker)")

    holder = Agent(
        name="holder",
        system_prompt=holder_system,
        client=get_client(args.model_a, registry_path=args.registry),
    )
    seeker = Agent(
        name="seeker",
        system_prompt=seeker_system,
        client=get_client(args.model_b, registry_path=args.registry),
    )
    print(_rule(f"RUN: {args.model_a} (holder) vs {args.model_b} (seeker)"))
    transcript = run_conversation(
        holder,
        seeker,
        max_turns=args.max_turns,
        opening_speaker=opening_speaker(scenario),
        opening_prompt=SCENARIO_OPENING_PROMPT,
        metadata={
            "scenario_id": scenario.id,
            "defense": defense.value,
            "adversary": adversary.value,
        },
    )
    _print_transcript(transcript, {"holder": holder, "seeker": seeker})
    path = save_transcript(transcript, runs_dir=args.runs_dir)
    print(f"\ntranscript saved to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
