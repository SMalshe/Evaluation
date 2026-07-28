"""Grid runner: every model pair against every scenario/condition cell.

One *cell* is a fully specified run:
``(scenario, defense, adversary, holder_model, seeker_model)``. For each cell the
runner renders the pair of system prompts, executes the conversation, scores the
side(s) named by ``role_under_test`` with the secret-based scorer, and appends one
flat JSONL row per scored side.

Two properties matter for a sweep that takes hours:

* **Resumable.** Every row carries a deterministic ``cell_id``; ``run_grid``
  skips cells already present in the output file, so an interrupted sweep
  continues where it stopped.
* **Residency-ordered.** Cells are emitted grouped by model pair, so a local
  inference server keeps the same one or two models resident instead of
  reloading weights between cells (which dominates wall-clock time on CPU).

Local weights also impose a memory ceiling: two large models cannot be resident
at once. ``PairPolicy`` encodes that as a footprint budget, and a pair of the
*same* model counts once because the server loads one copy and serves both
agents from it.
"""

from __future__ import annotations

import json
import time
import traceback
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .disclosure import (
    build_disclosure_metrics,
    run_disclosure_judgement,
    scored_sides,
)
from .engine import Agent, Transcript, run_conversation
from .models import ModelClient, ModelConfig, RetryPolicy, get_client, load_registry
from .persistence import save_transcript
from .prompts import (
    AdversaryStrategy,
    DefenseCondition,
    PromptConfig,
    opening_speaker,
    render_pair,
)
from .scenarios import Scenario, load_scenario

ClientFactory = Callable[[str], ModelClient]
DEFAULT_GRID_RESULTS = "results/grid.jsonl"

# Approximate resident size of each local model, in GB (on-disk weight size is a
# good proxy). Hosted entries cost no local memory.
LOCAL_FOOTPRINT_GB: dict[str, float] = {
    "local-qwen-32b": 19.0,
    "local-qwen-14b": 9.0,
    "local-llama-8b": 4.9,
    "local-llama-3b": 2.0,
    "local-llama-1b": 1.3,
}


@dataclass(frozen=True)
class PairPolicy:
    """Which (holder_model, seeker_model) pairs may run on this machine."""

    # Total resident local weights allowed across both agents of one cell.
    footprint_budget_gb: float = 24.0
    # Never run two *different* models from this set together, whatever the
    # budget says.
    heavy_models: frozenset[str] = frozenset({"local-qwen-32b"})

    def footprint_gb(self, holder_model: str, seeker_model: str) -> float:
        """Resident GB for a pair. The same model on both sides loads once."""
        names = {holder_model, seeker_model}
        return sum(LOCAL_FOOTPRINT_GB.get(name, 0.0) for name in names)

    def rejection(self, holder_model: str, seeker_model: str) -> str | None:
        """Why this pair cannot run here, or None if it can."""
        heavy = {m for m in (holder_model, seeker_model) if m in self.heavy_models}
        if len(heavy) > 1:
            return f"two distinct heavy local models ({', '.join(sorted(heavy))})"
        used = self.footprint_gb(holder_model, seeker_model)
        if used > self.footprint_budget_gb:
            return f"local footprint {used:.1f}GB exceeds budget {self.footprint_budget_gb:.1f}GB"
        return None


@dataclass(frozen=True)
class Cell:
    """One fully specified run."""

    scenario_id: str
    defense: DefenseCondition
    adversary: AdversaryStrategy
    holder_model: str
    seeker_model: str

    @property
    def cell_id(self) -> str:
        return (
            f"{self.scenario_id}|{self.defense.value}|{self.adversary.value}"
            f"|{self.holder_model}|{self.seeker_model}"
        )


@dataclass
class GridSpec:
    """The axes of the sweep."""

    models: list[str]
    scenarios: list[str]
    defenses: list[DefenseCondition] = field(default_factory=lambda: [DefenseCondition.none])
    adversaries: list[AdversaryStrategy] = field(
        default_factory=lambda: [AdversaryStrategy.direct_probe]
    )
    # Include a model paired against itself.
    include_self_play: bool = True

    def pairs(self, policy: PairPolicy) -> list[tuple[str, str]]:
        """Admissible ordered (holder, seeker) model pairs."""
        out = []
        for holder in self.models:
            for seeker in self.models:
                if holder == seeker and not self.include_self_play:
                    continue
                if policy.rejection(holder, seeker) is None:
                    out.append((holder, seeker))
        return out

    def skipped_pairs(self, policy: PairPolicy) -> list[tuple[str, str, str]]:
        """(holder, seeker, reason) for every pair the policy rejects."""
        out = []
        for holder in self.models:
            for seeker in self.models:
                if holder == seeker and not self.include_self_play:
                    continue
                reason = policy.rejection(holder, seeker)
                if reason is not None:
                    out.append((holder, seeker, reason))
        return out

    def cells(self, policy: PairPolicy) -> Iterator[Cell]:
        """Cells grouped by model pair, so weights stay resident across a block."""
        for holder_model, seeker_model in self.pairs(policy):
            for scenario_id in self.scenarios:
                for defense in self.defenses:
                    for adversary in self.adversaries:
                        yield Cell(
                            scenario_id=scenario_id,
                            defense=defense,
                            adversary=adversary,
                            holder_model=holder_model,
                            seeker_model=seeker_model,
                        )


@dataclass
class RunnerConfig:
    max_turns: int = 8
    judge_model: str = "local-qwen-14b"
    # Local CPU generation is slow; the default 120s per-call cap is far too
    # tight for a 32B model producing a few hundred tokens.
    timeout_s: float = 900.0
    max_retries: int = 2
    judge_retries: int = 2
    scenarios_dir: str = "scenarios"
    registry_path: str = "models.yaml"
    runs_dir: str = "runs"
    prompt_config: PromptConfig = field(default_factory=PromptConfig)
    save_transcripts: bool = True


def _cost_usd(prompt_tokens: int, completion_tokens: int, config: ModelConfig | None) -> float:
    if config is None:
        return 0.0
    return (
        prompt_tokens * config.price_per_mtok_in + completion_tokens * config.price_per_mtok_out
    ) / 1_000_000


def completed_cell_ids(path: str | Path) -> set[str]:
    """Cell ids already recorded in a results file (for resuming)."""
    out: set[str] = set()
    results = Path(path)
    if not results.is_file():
        return out
    with results.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # a partially written line from a killed process
            cell_id = row.get("cell_id")
            if isinstance(cell_id, str):
                out.add(cell_id)
    return out


def append_rows(rows: list[dict[str, Any]], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _transcript_totals(transcript: Transcript, registry: dict[str, ModelConfig]) -> dict[str, Any]:
    prompt_tokens = sum(t.prompt_tokens for t in transcript.turns)
    completion_tokens = sum(t.completion_tokens for t in transcript.turns)
    by_name = {info.name: info for info in transcript.agents}
    cost = 0.0
    for turn in transcript.turns:
        info = by_name.get(turn.speaker)
        config = registry.get(info.model_name) if info else None
        cost += _cost_usd(turn.prompt_tokens, turn.completion_tokens, config)
    return {
        "n_turns": len(transcript.turns),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost_usd": round(cost, 6),
        "latency_ms_total": round(sum(t.latency_ms for t in transcript.turns), 1),
    }


def run_cell(
    cell: Cell,
    scenario: Scenario,
    config: RunnerConfig,
    *,
    client_factory: ClientFactory,
    judge_client: ModelClient,
    registry: dict[str, ModelConfig],
) -> list[dict[str, Any]]:
    """Execute one cell and return one row per scored side.

    Never raises: a failed conversation or judge produces a row carrying the
    error, so a long sweep is not derailed by one bad cell.
    """
    base = {
        "cell_id": cell.cell_id,
        "scenario_id": cell.scenario_id,
        "scenario_title": scenario.title,
        "category": scenario.category.value,
        "role_under_test": scenario.role_under_test.value,
        "defense": cell.defense.value,
        "adversary": cell.adversary.value,
        "holder_model": cell.holder_model,
        "seeker_model": cell.seeker_model,
        "pair": f"{cell.holder_model} vs {cell.seeker_model}",
        "self_play": cell.holder_model == cell.seeker_model,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
    }

    started = time.time()
    try:
        holder_system, seeker_system = render_pair(
            scenario, cell.defense, cell.adversary, config=config.prompt_config
        )
        agents = {
            "holder": Agent(
                name="holder",
                system_prompt=holder_system,
                client=client_factory(cell.holder_model),
            ),
            "seeker": Agent(
                name="seeker",
                system_prompt=seeker_system,
                client=client_factory(cell.seeker_model),
            ),
        }
        transcript = run_conversation(
            agents["holder"],
            agents["seeker"],
            max_turns=config.max_turns,
            opening_speaker=opening_speaker(scenario),
            metadata={
                "scenario_id": cell.scenario_id,
                "defense": cell.defense.value,
                "adversary": cell.adversary.value,
                "cell_id": cell.cell_id,
            },
        )
    except Exception as exc:  # noqa: BLE001 - one bad cell must not kill the sweep
        return [
            {
                **base,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "error_stage": "conversation",
                "duration_s": round(time.time() - started, 1),
            }
        ]

    transcript_path = ""
    if config.save_transcripts:
        try:
            transcript_path = str(save_transcript(transcript, config.runs_dir))
        except Exception:  # noqa: BLE001 - persistence is not worth losing a run over
            transcript_path = ""

    totals = _transcript_totals(transcript, registry)
    conversation = {
        **base,
        **totals,
        "termination": transcript.termination,
        "deal_amount": transcript.deal_amount,
        "transcript_path": transcript_path,
        "duration_s": round(time.time() - started, 1),
    }

    rows: list[dict[str, Any]] = []
    for side_name in scored_sides(scenario):
        try:
            judgement = run_disclosure_judgement(
                judge_client,
                transcript,
                scenario,
                side_name,
                retries=config.judge_retries,
            )
            metrics = build_disclosure_metrics(scenario, side_name, judgement)
            rows.append(
                {
                    **conversation,
                    "ok": True,
                    "error": "",
                    "scored_side": side_name,
                    "scored_model": (
                        cell.holder_model if side_name == "holder" else cell.seeker_model
                    ),
                    "opponent_model": (
                        cell.seeker_model if side_name == "holder" else cell.holder_model
                    ),
                    **metrics.model_dump(exclude={"side"}),
                    "judge_prompt_tokens": judgement.prompt_tokens,
                    "judge_completion_tokens": judgement.completion_tokens,
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    **conversation,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "error_stage": "judge",
                    "scored_side": side_name,
                }
            )
    return rows


def run_grid(
    spec: GridSpec,
    config: RunnerConfig | None = None,
    *,
    policy: PairPolicy | None = None,
    results_path: str | Path = DEFAULT_GRID_RESULTS,
    client_factory: ClientFactory | None = None,
    resume: bool = True,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Execute the whole grid, appending rows as each cell finishes."""
    config = config or RunnerConfig()
    policy = policy or PairPolicy()
    retry = RetryPolicy(timeout_s=config.timeout_s, max_retries=config.max_retries)

    if client_factory is None:

        def build_client(name: str) -> ModelClient:
            return get_client(name, registry_path=config.registry_path, retry=retry)

        client_factory = build_client

    registry = load_registry(config.registry_path)
    scenarios = {sid: load_scenario(sid, config.scenarios_dir) for sid in spec.scenarios}
    judge_client = client_factory(config.judge_model)

    for holder, seeker, reason in spec.skipped_pairs(policy):
        progress(f"  skip pair {holder} vs {seeker}: {reason}")

    cells = list(spec.cells(policy))
    done = completed_cell_ids(results_path) if resume else set()
    todo = [c for c in cells if c.cell_id not in done]
    progress(
        f"grid: {len(cells)} cells ({len(spec.pairs(policy))} pairs x "
        f"{len(spec.scenarios)} scenarios x {len(spec.defenses)} defenses x "
        f"{len(spec.adversaries)} adversaries); {len(done)} already done, {len(todo)} to run"
    )

    started = time.time()
    failures = 0
    for i, cell in enumerate(todo, start=1):
        cell_started = time.time()
        try:
            rows = run_cell(
                cell,
                scenarios[cell.scenario_id],
                config,
                client_factory=client_factory,
                judge_client=judge_client,
                registry=registry,
            )
        except Exception:  # noqa: BLE001 - defensive; run_cell already guards
            rows = [
                {
                    "cell_id": cell.cell_id,
                    "ok": False,
                    "error": traceback.format_exc(limit=3),
                    "error_stage": "runner",
                }
            ]
        append_rows(rows, results_path)

        bad = sum(1 for r in rows if not r.get("ok"))
        failures += bad
        elapsed = time.time() - started
        rate = elapsed / i
        remaining = rate * (len(todo) - i)
        status = "FAIL" if bad else "ok"
        progress(
            f"[{i}/{len(todo)}] {status} {cell.cell_id} "
            f"({time.time() - cell_started:.0f}s, eta {remaining / 60:.0f}m)"
        )

    return {
        "cells_total": len(cells),
        "cells_run": len(todo),
        "rows_failed": failures,
        "elapsed_s": round(time.time() - started, 1),
        "results_path": str(results_path),
    }


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m src.experiment",
        description="Run every model pair across a scenario/condition grid.",
    )
    parser.add_argument(
        "--models", required=True, help="comma-separated registry names to cross with each other"
    )
    parser.add_argument("--scenarios", required=True, help="comma-separated scenario ids")
    parser.add_argument("--defenses", default="none", help="comma-separated defense conditions")
    parser.add_argument(
        "--adversaries", default="direct_probe", help="comma-separated adversary strategies"
    )
    parser.add_argument("--judge-model", default="local-qwen-14b")
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--results", default=DEFAULT_GRID_RESULTS)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--scenarios-dir", default="scenarios")
    parser.add_argument("--registry", default="models.yaml")
    parser.add_argument(
        "--footprint-budget-gb",
        type=float,
        default=24.0,
        help="max resident local weights across both agents of one cell",
    )
    parser.add_argument("--no-self-play", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-save-transcripts", action="store_true")
    parser.add_argument("--enable-authority-verifiable", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true", help="list the cells that would run, then exit"
    )
    args = parser.parse_args(argv)

    spec = GridSpec(
        models=_csv(args.models),
        scenarios=_csv(args.scenarios),
        defenses=[DefenseCondition(d) for d in _csv(args.defenses)],
        adversaries=[AdversaryStrategy(a) for a in _csv(args.adversaries)],
        include_self_play=not args.no_self_play,
    )
    policy = PairPolicy(footprint_budget_gb=args.footprint_budget_gb)
    config = RunnerConfig(
        max_turns=args.max_turns,
        judge_model=args.judge_model,
        timeout_s=args.timeout_s,
        scenarios_dir=args.scenarios_dir,
        registry_path=args.registry,
        runs_dir=args.runs_dir,
        prompt_config=PromptConfig(enable_authority_verifiable=args.enable_authority_verifiable),
        save_transcripts=not args.no_save_transcripts,
    )

    if args.dry_run:
        for holder, seeker, reason in spec.skipped_pairs(policy):
            print(f"skip pair {holder} vs {seeker}: {reason}")
        cells = list(spec.cells(policy))
        done = set() if args.no_resume else completed_cell_ids(args.results)
        for cell in cells:
            mark = "done" if cell.cell_id in done else "todo"
            print(f"{mark}  {cell.cell_id}")
        print(f"\n{len(cells)} cells, {len(cells) - len(done & {c.cell_id for c in cells})} to run")
        return 0

    summary = run_grid(
        spec,
        config,
        policy=policy,
        results_path=args.results,
        resume=not args.no_resume,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
