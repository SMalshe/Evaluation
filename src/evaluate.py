"""Evaluate one negotiation transcript into a ``RunResult`` and persist it.

``evaluate_run`` runs the adversary extraction and the independent judge, then
computes the metrics. The extraction and judge models are chosen by
``EvalConfig`` independently of the models that negotiated (extraction defaults
to the seller's own model - the natural adversary).

    python -m src.evaluate --transcript runs/<file>.json --scenario s01 \
        --defense none --adversary authority
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from .engine import Transcript
from .extraction import run_extraction
from .judge import run_judgement
from .metrics import RunResult, build_run_result
from .models import ModelClient, get_client
from .persistence import load_transcript
from .prompts import AdversaryStrategy, DefenseCondition
from .scenarios import Scenario, load_scenario

ClientFactory = Callable[[str], ModelClient]
DEFAULT_RESULTS = "results/results.jsonl"


@dataclass(frozen=True)
class EvalConfig:
    """Which models evaluate a run (independent of the negotiating models)."""

    judge_model: str = "claude-sonnet"
    extraction_model: str | None = None  # None => the seller's own model
    judge_temperature: float | None = 0.0
    extraction_retries: int = 3


DEFAULT_EVAL_CONFIG = EvalConfig()  # frozen; safe to share as a default


def _model_of(transcript: Transcript, name: str) -> str:
    for info in transcript.agents:
        if info.name == name:
            return info.model_name
    raise ValueError(f"transcript has no agent named {name!r}")


def evaluate_run(
    transcript: Transcript,
    scenario: Scenario,
    config: EvalConfig = DEFAULT_EVAL_CONFIG,
    *,
    defense: DefenseCondition,
    adversary: AdversaryStrategy,
    buyer_name: str = "buyer",
    seller_name: str = "seller",
    client_factory: ClientFactory | None = None,
) -> RunResult:
    """Score a transcript. ``defense``/``adversary`` are the conditions it was run
    under (the transcript itself doesn't record them)."""
    factory = client_factory or (lambda name: get_client(name))
    buyer_model = _model_of(transcript, buyer_name)
    seller_model = _model_of(transcript, seller_name)
    extraction_model = config.extraction_model or seller_model

    started = time.monotonic()
    extraction = run_extraction(
        factory(extraction_model), transcript, seller_name, retries=config.extraction_retries
    )
    judge = run_judgement(
        factory(config.judge_model),
        transcript,
        scenario,
        buyer_name,
        temperature=config.judge_temperature,
    )
    eval_s = time.monotonic() - started

    return build_run_result(
        scenario=scenario,
        transcript=transcript,
        defense=defense.value,
        adversary=adversary.value,
        buyer_model=buyer_model,
        seller_model=seller_model,
        extraction_model=extraction_model,
        judge_model=config.judge_model,
        extraction=extraction,
        judge=judge,
        eval_s=eval_s,
    )


def append_result(result: RunResult, path: str | Path = DEFAULT_RESULTS) -> Path:
    """Append one result as a JSONL line, creating the file/dir if needed."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        handle.write(result.model_dump_json() + "\n")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv(override=False)
    parser = argparse.ArgumentParser(
        prog="python -m src.evaluate",
        description="Evaluate a saved transcript into a RunResult JSONL row.",
    )
    parser.add_argument("--transcript", required=True, help="path to a saved transcript JSON")
    parser.add_argument("--scenario", required=True, help="scenario id the run used, e.g. s01")
    parser.add_argument("--defense", required=True, choices=[d.value for d in DefenseCondition])
    parser.add_argument("--adversary", required=True, choices=[a.value for a in AdversaryStrategy])
    parser.add_argument("--judge-model", default=EvalConfig.judge_model)
    parser.add_argument(
        "--extraction-model", default=None, help="defaults to the seller's own model"
    )
    parser.add_argument("--scenarios-dir", default="scenarios")
    parser.add_argument("--out", default=DEFAULT_RESULTS, help="JSONL output path")
    args = parser.parse_args(argv)

    transcript = load_transcript(args.transcript)
    scenario = load_scenario(args.scenario, scenarios_dir=args.scenarios_dir)
    config = EvalConfig(judge_model=args.judge_model, extraction_model=args.extraction_model)

    result = evaluate_run(
        transcript,
        scenario,
        config,
        defense=DefenseCondition(args.defense),
        adversary=AdversaryStrategy(args.adversary),
    )
    path = append_result(result, args.out)

    print(f"reservation: true ${result.reservation_true} est {result.reservation_est} ", end="")
    if result.reservation_abs_pct_error is not None:
        print(
            f"({result.reservation_abs_pct_error:.1f}% err, "
            f"{'hit' if result.reservation_hit_10pct else 'miss'})"
        )
    else:
        print("(extraction invalid)")
    print(f"urgency: {result.urgency_true} vs {result.urgency_guess} -> {result.urgency_correct}")
    print(
        f"sensitive: {result.sensitive_true} vs {result.sensitive_guess} "
        f"-> {result.sensitive_correct}"
    )
    print(
        f"first leak turn: res={result.first_leak_turn_reservation} "
        f"urg={result.first_leak_turn_urgency} sens={result.first_leak_turn_sensitive} "
        f"prompt_leak={result.prompt_leak}"
    )
    outcome = f"${result.final_price}" if result.deal_reached else result.termination
    print(f"outcome: {outcome}  buyer_surplus={result.buyer_surplus}  overpaid={result.overpaid}")
    print(f"appended to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
