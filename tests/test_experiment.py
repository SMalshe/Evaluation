"""Tests for the grid runner: pair policy, cell ordering, resume, error capture."""

from __future__ import annotations

import json
from pathlib import Path

from src.experiment import (
    Cell,
    GridSpec,
    PairPolicy,
    RunnerConfig,
    completed_cell_ids,
    run_grid,
)
from src.models import ModelClient
from src.prompts import AdversaryStrategy, DefenseCondition
from tests.test_engine import MockBackend


def make_client(name: str, replies: list[str] | None = None) -> ModelClient:
    return MockBackend(name, replies or ["Sure, let's talk." for _ in range(40)])


JUDGE_JSON = json.dumps(
    {
        "judgements": [
            {
                "secret_name": "balance",
                "label": "explicit",
                "turn_index": 2,
                "evidence": "about 9,400",
                "condition_met": False,
            }
        ],
        "prompt_leak": False,
        "prompt_leak_evidence": "",
    }
)


def factory(name: str) -> ModelClient:
    if name == "judgemock":
        return make_client(name, [JUDGE_JSON] * 40)
    return make_client(name)


# --- pair policy ------------------------------------------------------------


def test_policy_allows_self_play_of_a_heavy_model() -> None:
    """The same model on both sides loads one copy, so it fits."""
    policy = PairPolicy()
    assert policy.rejection("local-qwen-32b", "local-qwen-32b") is None
    assert policy.footprint_gb("local-qwen-32b", "local-qwen-32b") == 19.0


def test_policy_rejects_two_distinct_heavy_models() -> None:
    policy = PairPolicy(heavy_models=frozenset({"local-qwen-32b", "local-qwen-14b"}))
    reason = policy.rejection("local-qwen-32b", "local-qwen-14b")
    assert reason is not None and "two distinct heavy" in reason


def test_policy_rejects_a_pair_over_the_footprint_budget() -> None:
    policy = PairPolicy(footprint_budget_gb=24.0)
    reason = policy.rejection("local-qwen-32b", "local-qwen-14b")  # 19 + 9 = 28
    assert reason is not None and "exceeds budget" in reason
    assert policy.rejection("local-qwen-32b", "local-llama-3b") is None  # 19 + 2 = 21


def test_hosted_models_have_no_local_footprint() -> None:
    policy = PairPolicy(footprint_budget_gb=1.0)
    assert policy.rejection("claude-opus", "gpt-sol") is None


# --- grid shape -------------------------------------------------------------


def test_cells_are_grouped_by_model_pair() -> None:
    spec = GridSpec(
        models=["a", "b"],
        scenarios=["s01", "s02"],
        defenses=[DefenseCondition.none],
        adversaries=[AdversaryStrategy.direct_probe],
    )
    cells = list(spec.cells(PairPolicy()))
    pairs = [(c.holder_model, c.seeker_model) for c in cells]
    # Every pair's cells must be contiguous, or weights reload between cells.
    assert pairs == sorted(pairs, key=pairs.index)
    assert len(cells) == 4 * 2  # 4 ordered pairs x 2 scenarios
    assert [p for p in dict.fromkeys(pairs)] == [("a", "a"), ("a", "b"), ("b", "a"), ("b", "b")]


def test_self_play_can_be_excluded() -> None:
    spec = GridSpec(models=["a", "b"], scenarios=["s01"], include_self_play=False)
    assert spec.pairs(PairPolicy()) == [("a", "b"), ("b", "a")]


def test_cell_id_is_stable_and_distinct() -> None:
    cell = Cell("s01", DefenseCondition.basic, AdversaryStrategy.rapport, "a", "b")
    assert cell.cell_id == "s01|basic|rapport|a|b"


# --- execution --------------------------------------------------------------


def test_run_grid_writes_a_row_per_scored_side_and_resumes(tmp_path: Path) -> None:
    results = tmp_path / "grid.jsonl"
    spec = GridSpec(
        models=["m1", "m2"],
        scenarios=["s01"],
        defenses=[DefenseCondition.none],
        adversaries=[AdversaryStrategy.direct_probe],
    )
    config = RunnerConfig(max_turns=2, judge_model="judgemock", runs_dir=str(tmp_path / "runs"))

    summary = run_grid(
        spec,
        config,
        results_path=results,
        client_factory=factory,
        progress=lambda _msg: None,
    )
    assert summary["cells_run"] == 4  # 2x2 ordered pairs
    assert summary["rows_failed"] == 0

    rows = [json.loads(line) for line in results.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 4  # s01 has role_under_test=holder -> one scored side
    row = rows[0]
    assert row["scored_side"] == "holder"
    assert row["inappropriate_total"] == 1  # judge said explicit, condition unmet
    assert row["n_turns"] == 2
    assert row["cost_usd"] == 0.0  # mock config has no prices

    # Re-running skips everything already recorded.
    again = run_grid(
        spec, config, results_path=results, client_factory=factory, progress=lambda _m: None
    )
    assert again["cells_run"] == 0
    assert len(results.read_text(encoding="utf-8").splitlines()) == 4


def test_a_failing_cell_is_recorded_and_the_sweep_continues(tmp_path: Path) -> None:
    results = tmp_path / "grid.jsonl"

    def flaky(name: str) -> ModelClient:
        if name == "broken":
            raise RuntimeError("no such model")
        return factory(name)

    spec = GridSpec(models=["m1", "broken"], scenarios=["s01"])
    config = RunnerConfig(max_turns=2, judge_model="judgemock", runs_dir=str(tmp_path / "runs"))

    summary = run_grid(
        spec, config, results_path=results, client_factory=flaky, progress=lambda _m: None
    )
    rows = [json.loads(line) for line in results.read_text(encoding="utf-8").splitlines()]
    failed = [r for r in rows if not r["ok"]]

    assert summary["cells_run"] == 4
    assert len(failed) == 3  # every pair touching "broken"
    assert all(r["error_stage"] == "conversation" for r in failed)
    assert any(r["ok"] for r in rows)  # the healthy pair still produced a result


def test_completed_cell_ids_tolerates_a_truncated_line(tmp_path: Path) -> None:
    results = tmp_path / "grid.jsonl"
    results.write_text('{"cell_id": "a|none|passive|x|y"}\n{"cell_id": "trunc"\n', encoding="utf-8")
    assert completed_cell_ids(results) == {"a|none|passive|x|y"}
