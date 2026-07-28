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


def make_config(tmp_path: Path, **kwargs: object) -> RunnerConfig:
    return RunnerConfig(
        max_turns=2, judge_model="judgemock", runs_dir=str(tmp_path / "runs"), **kwargs
    )


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_both_phases_produce_a_row_per_scored_side_and_resume(tmp_path: Path) -> None:
    conversations = tmp_path / "conversations.jsonl"
    results = tmp_path / "grid.jsonl"
    spec = GridSpec(
        models=["m1", "m2"],
        scenarios=["s01"],
        defenses=[DefenseCondition.none],
        adversaries=[AdversaryStrategy.direct_probe],
    )
    config = make_config(tmp_path)

    summary = run_grid(
        spec,
        config,
        conversations_path=conversations,
        results_path=results,
        client_factory=factory,
        progress=lambda _msg: None,
    )
    assert summary["conversations_run"] == 4  # 2x2 ordered pairs
    assert summary["conversations_failed"] == 0
    assert summary["judgements_failed"] == 0

    conv = read_rows(conversations)
    assert len(conv) == 4
    assert conv[0]["n_turns"] == 2
    assert conv[0]["cost_usd"] == 0.0  # mock config has no prices
    assert conv[0]["transcript_path"]

    rows = read_rows(results)
    assert len(rows) == 4  # s01 puts the holder under test -> one scored side
    assert rows[0]["scored_side"] == "holder"
    assert rows[0]["inappropriate_total"] == 1  # judge said explicit, condition unmet
    assert rows[0]["scored_model"] in {"m1", "m2"}

    # Re-running skips both phases.
    again = run_grid(
        spec,
        config,
        conversations_path=conversations,
        results_path=results,
        client_factory=factory,
        progress=lambda _m: None,
    )
    assert again["conversations_run"] == 0
    assert again["judged"] == 0
    assert len(read_rows(conversations)) == 4
    assert len(read_rows(results)) == 4


def test_phases_can_run_separately(tmp_path: Path) -> None:
    conversations = tmp_path / "conversations.jsonl"
    results = tmp_path / "grid.jsonl"
    spec = GridSpec(models=["m1"], scenarios=["s01"])
    config = make_config(tmp_path)

    run_grid(
        spec,
        config,
        conversations_path=conversations,
        results_path=results,
        client_factory=factory,
        phases=("conversations",),
        progress=lambda _m: None,
    )
    assert len(read_rows(conversations)) == 1
    assert not results.exists()

    run_grid(
        spec,
        config,
        conversations_path=conversations,
        results_path=results,
        client_factory=factory,
        phases=("judge",),
        progress=lambda _m: None,
    )
    assert len(read_rows(results)) == 1


def test_only_small_cells_are_parallelisable() -> None:
    policy = PairPolicy(parallel_budget_gb=6.0)
    assert policy.is_parallelisable("local-llama-3b", "local-llama-1b")  # 3.3GB
    assert policy.is_parallelisable("local-llama-8b", "local-llama-8b")  # 4.9GB, loaded once
    assert not policy.is_parallelisable("local-llama-8b", "local-qwen-14b")  # 13.9GB
    assert not policy.is_parallelisable("local-qwen-32b", "local-qwen-32b")  # heavy


def test_concurrent_workers_write_every_row(tmp_path: Path) -> None:
    conversations = tmp_path / "conversations.jsonl"
    spec = GridSpec(models=["local-llama-1b", "local-llama-3b"], scenarios=["s01"])

    summary = run_grid(
        spec,
        make_config(tmp_path, max_workers=4),
        policy=PairPolicy(parallel_budget_gb=6.0),
        conversations_path=conversations,
        results_path=tmp_path / "grid.jsonl",
        client_factory=factory,
        phases=("conversations",),
        progress=lambda _m: None,
    )
    assert summary["conversations_run"] == 4
    assert summary["conversations_failed"] == 0
    assert len(read_rows(conversations)) == 4  # no interleaved/torn lines


def test_a_failing_cell_is_recorded_and_the_sweep_continues(tmp_path: Path) -> None:
    conversations = tmp_path / "conversations.jsonl"

    def flaky(name: str) -> ModelClient:
        if name == "broken":
            raise RuntimeError("no such model")
        return factory(name)

    spec = GridSpec(models=["m1", "broken"], scenarios=["s01"])

    summary = run_grid(
        spec,
        make_config(tmp_path),
        conversations_path=conversations,
        results_path=tmp_path / "grid.jsonl",
        client_factory=flaky,
        phases=("conversations",),
        progress=lambda _m: None,
    )
    rows = read_rows(conversations)
    failed = [r for r in rows if not r["ok"]]

    assert summary["conversations_run"] == 4
    assert len(failed) == 3  # every pair touching "broken"
    assert all(r["error_stage"] == "conversation" for r in failed)
    assert any(r["ok"] for r in rows)  # the healthy pair still produced a result


def test_completed_cell_ids_tolerates_a_truncated_line(tmp_path: Path) -> None:
    results = tmp_path / "grid.jsonl"
    results.write_text('{"cell_id": "a|none|passive|x|y"}\n{"cell_id": "trunc"\n', encoding="utf-8")
    assert completed_cell_ids(results) == {"a|none|passive|x|y"}
