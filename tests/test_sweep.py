"""Offline tests for the sweep runner (mock backend, no network)."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from src.models import ModelConfig
from src.prompts import AdversaryStrategy, DefenseCondition, PromptConfig
from src.scenarios import Category, load_scenario, scenario_ids
from src.sweep import (
    CATEGORY_CONDITIONS,
    Cell,
    ClientCache,
    ConditionMode,
    SweepIndex,
    build_plan,
    cell_cost,
    conditions_for,
    is_local,
    main,
    pairings,
    run_sweep,
)

from .test_engine import MockBackend

SCENARIOS_DIR = "scenarios"


def _registry() -> dict[str, ModelConfig]:
    return {
        "local-a": ModelConfig(
            name="local-a",
            backend="openai_compat",
            model_id="a",
            api_key_env="X",
            base_url="http://localhost:11434/v1",
        ),
        "local-b": ModelConfig(
            name="local-b",
            backend="openai_compat",
            model_id="b",
            api_key_env="X",
            base_url="http://127.0.0.1:11434/v1",
        ),
        "hosted": ModelConfig(
            name="hosted",
            backend="anthropic",
            model_id="h",
            api_key_env="X",
            price_per_mtok_in=3.0,
            price_per_mtok_out=15.0,
        ),
    }


def _scenarios(*ids: str) -> dict[str, object]:
    return {sid: load_scenario(sid, SCENARIOS_DIR) for sid in ids}


# --- planning --------------------------------------------------------------


def test_pairings_cross_is_every_ordered_pair_including_self() -> None:
    pairs = pairings(cross=["a", "b"])
    assert pairs == [("a", "a"), ("a", "b"), ("b", "a"), ("b", "b")]


def test_pairings_appends_self_play_and_dedupes() -> None:
    pairs = pairings(cross=["a"], self_play=["a", "c"], explicit=[("a", "c")])
    assert pairs == [("a", "a"), ("c", "c"), ("a", "c")]


def test_natural_conditions_follow_the_category_table() -> None:
    scenario = load_scenario("s01", SCENARIOS_DIR)
    assert conditions_for(scenario, ConditionMode.natural) == [
        CATEGORY_CONDITIONS[scenario.category]
    ]


def test_every_category_has_a_natural_condition() -> None:
    assert set(CATEGORY_CONDITIONS) == set(Category)


def test_grid_mode_is_defenses_times_enabled_adversaries() -> None:
    scenario = load_scenario("s01", SCENARIOS_DIR)
    assert len(conditions_for(scenario, ConditionMode.grid)) == 3 * 7
    gated = PromptConfig(enable_authority_verifiable=True)
    assert len(conditions_for(scenario, ConditionMode.grid, config=gated)) == 3 * 8


def test_defense_and_adversary_modes_vary_one_factor() -> None:
    scenario = load_scenario("s01", SCENARIOS_DIR)
    defense_only = conditions_for(
        scenario, ConditionMode.defense, adversary=AdversaryStrategy.rapport
    )
    assert len(defense_only) == 3
    assert {a for _, a in defense_only} == {AdversaryStrategy.rapport}

    adversary_only = conditions_for(
        scenario, ConditionMode.adversary, defense=DefenseCondition.strong
    )
    assert len(adversary_only) == 7
    assert {d for d, _ in adversary_only} == {DefenseCondition.strong}
    assert AdversaryStrategy.subliminal_chat in {a for _, a in adversary_only}


def test_build_plan_multiplies_out_and_keys_are_unique() -> None:
    scenarios = list(_scenarios("s01", "s02").values())
    cells = build_plan(scenarios, [("a", "b"), ("b", "a")], ConditionMode.natural, repeats=2)
    assert len(cells) == 2 * 2 * 1 * 2
    assert len({cell.key for cell in cells}) == len(cells)


def test_full_default_plan_covers_every_scenario_once_per_pairing() -> None:
    scenarios = list(_scenarios(*scenario_ids(SCENARIOS_DIR)).values())
    pairs = pairings(cross=["ollama-3b", "ollama-8b"], self_play=["claude-sonnet", "gpt-mid"])
    cells = build_plan(scenarios, pairs, ConditionMode.natural)
    assert len(scenarios) == 48
    assert len(pairs) == 6
    assert len(cells) == 48 * 6


# --- lanes and cost --------------------------------------------------------


def test_is_local_detects_loopback_base_urls() -> None:
    registry = _registry()
    assert is_local(registry["local-a"])
    assert is_local(registry["local-b"])
    assert not is_local(registry["hosted"])


def test_cell_cost_bills_each_turn_to_the_speaking_model() -> None:
    from src.engine import Transcript, Turn

    transcript = Transcript(
        turns=[
            Turn(
                index=0,
                speaker="buyer",
                text="",
                prompt_tokens=1_000_000,
                completion_tokens=0,
                latency_ms=1,
            ),
            Turn(
                index=1,
                speaker="seller",
                text="",
                prompt_tokens=0,
                completion_tokens=1_000_000,
                latency_ms=1,
            ),
        ],
        termination="max_turns",
        agents=[],
        opening_speaker="buyer",
        opening_prompt="",
        max_turns=2,
        started_at="2026-01-01T00:00:00Z",
        ended_at="2026-01-01T00:00:01Z",
    )
    cell = Cell(
        scenario_id="s01",
        buyer_model="hosted",
        seller_model="local-a",
        defense=DefenseCondition.none,
        adversary=AdversaryStrategy.passive,
    )
    # buyer (hosted) 1M input @ $3 = $3; seller (local) is free.
    assert cell_cost(transcript, _registry(), cell) == pytest.approx(3.0)


# --- index / resume --------------------------------------------------------


def test_index_skips_ok_and_retries_failures(tmp_path: Path) -> None:
    index = SweepIndex(tmp_path / "index.jsonl")
    from datetime import UTC, datetime

    from src.sweep import CellRecord

    common = dict(
        scenario_id="s01",
        buyer_model="a",
        seller_model="b",
        defense="none",
        adversary="passive",
        repeat=0,
        finished_at=datetime.now(UTC),
    )
    index.append(CellRecord(key="good", status="ok", **common))
    index.append(CellRecord(key="bad", status="error", error="boom", **common))

    assert index.done_keys(include_failed=True) == {"good", "bad"}
    assert index.done_keys(include_failed=False) == {"good"}


def test_index_tolerates_a_truncated_line(tmp_path: Path) -> None:
    path = tmp_path / "index.jsonl"
    path.write_text('{"key": "good", "status": "ok"}\n{"key": "trunc"', encoding="utf-8")
    assert SweepIndex(path).done_keys(include_failed=True) == {"good"}


# --- execution -------------------------------------------------------------


def _mock_factory(replies: int = 40):
    def factory(name: str):
        return MockBackend(name, [f"reply from {name}"] * replies)

    return factory


def test_run_sweep_writes_transcripts_metadata_and_index(tmp_path: Path) -> None:
    scenarios = _scenarios("s01", "s02")
    cells = build_plan(list(scenarios.values()), [("hosted", "hosted")], ConditionMode.natural)
    registry = _registry()

    report = run_sweep(
        cells,
        scenarios,
        registry,
        out_dir=tmp_path,
        sweep_name="t",
        client_factory=_mock_factory(),
        max_turns=4,
        remote_workers=2,
    )

    assert report.completed == 2
    assert report.failed == 0
    written = list((tmp_path / "runs").glob("*.json"))
    assert len(written) == 2

    saved = json.loads(written[0].read_text(encoding="utf-8"))
    meta = saved["metadata"]
    assert set(meta) >= {"scenario_id", "defense", "adversary"}
    assert meta["sweep"] == "t"

    rows = [json.loads(line) for line in (tmp_path / "index.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert all(row["status"] == "ok" for row in rows)


def test_concurrent_cells_never_share_a_transcript_file(tmp_path: Path) -> None:
    """Regression: every cell names its agents buyer/seller and the auto-generated
    filename only has second resolution, so parallel cells used to overwrite each
    other's transcripts. One file per completed cell, always."""
    scenarios = _scenarios(*scenario_ids(SCENARIOS_DIR)[:12])
    cells = build_plan(list(scenarios.values()), [("hosted", "hosted")], ConditionMode.natural)

    report = run_sweep(
        cells,
        scenarios,
        _registry(),
        out_dir=tmp_path,
        sweep_name="t",
        client_factory=_mock_factory(),
        max_turns=2,
        remote_workers=8,
    )

    assert report.completed == 12
    assert len(list((tmp_path / "runs").glob("*.json"))) == 12
    assert len({cell.filename for cell in cells}) == 12


def test_rerun_resumes_and_skips_completed_cells(tmp_path: Path) -> None:
    scenarios = _scenarios("s01")
    cells = build_plan(list(scenarios.values()), [("hosted", "hosted")], ConditionMode.natural)
    kwargs = dict(
        out_dir=tmp_path,
        sweep_name="t",
        client_factory=_mock_factory(),
        max_turns=4,
    )

    first = run_sweep(cells, scenarios, _registry(), **kwargs)
    assert (first.completed, first.skipped) == (1, 0)

    second = run_sweep(cells, scenarios, _registry(), **kwargs)
    assert (second.completed, second.skipped) == (0, 1)
    assert len(list((tmp_path / "runs").glob("*.json"))) == 1


def test_a_failing_cell_is_recorded_and_the_sweep_continues(tmp_path: Path) -> None:
    scenarios = _scenarios("s01", "s02", "s03")
    cells = build_plan(list(scenarios.values()), [("hosted", "hosted")], ConditionMode.natural)

    def factory(name: str):
        return MockBackend(name, [])  # runs out of replies immediately -> raises

    broken = run_sweep(
        cells, scenarios, _registry(), out_dir=tmp_path, sweep_name="t", client_factory=factory
    )
    assert broken.completed == 0
    assert broken.failed == 3

    rows = [json.loads(line) for line in (tmp_path / "index.jsonl").read_text().splitlines()]
    assert all(row["status"] == "error" and row["error"] for row in rows)

    # A plain rerun treats errored cells as done, so nothing is repeated.
    rerun = run_sweep(
        cells,
        scenarios,
        _registry(),
        out_dir=tmp_path,
        sweep_name="t",
        client_factory=_mock_factory(),
    )
    assert (rerun.completed, rerun.skipped) == (0, 3)

    # --redo-failed opts into retrying them.
    healed = run_sweep(
        cells,
        scenarios,
        _registry(),
        out_dir=tmp_path,
        sweep_name="t",
        client_factory=_mock_factory(),
        redo_failed=True,
    )
    assert healed.completed == 3


def test_stop_event_leaves_remaining_cells_unstarted(tmp_path: Path) -> None:
    scenarios = _scenarios(*scenario_ids(SCENARIOS_DIR)[:6])
    cells = build_plan(list(scenarios.values()), [("hosted", "hosted")], ConditionMode.natural)
    halt = threading.Event()
    halt.set()

    report = run_sweep(
        cells,
        scenarios,
        _registry(),
        out_dir=tmp_path,
        sweep_name="t",
        client_factory=_mock_factory(),
        stop=halt,
    )
    assert report.completed == 0
    assert not (tmp_path / "index.jsonl").exists()


def test_a_cell_cut_short_mid_conversation_is_not_indexed(tmp_path: Path) -> None:
    """A run the stop signal truncates must not count as done, or a resume would
    keep the partial transcript forever."""
    scenarios = _scenarios("s01")
    cells = build_plan(list(scenarios.values()), [("hosted", "hosted")], ConditionMode.natural)
    halt = threading.Event()

    def factory(name: str):
        # Trip the stop signal once the conversation is already under way.
        class Tripwire(MockBackend):
            def chat(self, *args, **kwargs):
                halt.set()
                return super().chat(*args, **kwargs)

        return Tripwire(name, ["reply"] * 10)

    report = run_sweep(
        cells,
        scenarios,
        _registry(),
        out_dir=tmp_path,
        sweep_name="t",
        client_factory=factory,
        max_turns=6,
        stop=halt,
    )

    assert report.completed == 0
    assert not (tmp_path / "index.jsonl").exists()

    # Resuming reruns the cell rather than skipping it.
    resumed = run_sweep(
        cells,
        scenarios,
        _registry(),
        out_dir=tmp_path,
        sweep_name="t",
        client_factory=_mock_factory(),
        max_turns=6,
    )
    assert (resumed.completed, resumed.skipped) == (1, 0)


def test_client_cache_builds_each_model_once() -> None:
    built: list[str] = []

    def factory(name: str):
        built.append(name)
        return MockBackend(name, ["x"])

    cache = ClientCache(factory)
    for _ in range(5):
        cache.get("a")
        cache.get("b")
    assert built == ["a", "b"]


# --- CLI -------------------------------------------------------------------


def test_dry_run_prints_the_plan_without_calling_anything(capsys) -> None:
    code = main(["--dry-run", "--scenarios", "s01,s02"])
    assert code == 0
    out = capsys.readouterr().out
    assert "cells:" in out
    assert "dry run" in out
    # 2 scenarios x 6 default pairings x 1 natural condition
    assert "cells:      12" in out


def test_dry_run_rejects_an_unknown_model(capsys) -> None:
    assert main(["--dry-run", "--self", "nope"]) == 2
    assert "unknown model" in capsys.readouterr().err


def test_dry_run_rejects_a_gated_adversary(capsys) -> None:
    code = main(
        [
            "--dry-run",
            "--scenarios",
            "s01",
            "--self",
            "claude-sonnet",
            "--conditions",
            "fixed",
            "--adversary",
            "authority_verifiable",
        ]
    )
    assert code == 2
    assert "gated" in capsys.readouterr().err
