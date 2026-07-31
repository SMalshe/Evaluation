"""Tests for report aggregation."""

from __future__ import annotations

from src.report import aggregate, deduplicate


def row(cell: str, side: str = "holder", rate: float = 0.0, **kw):
    base = {
        "cell_id": cell,
        "scored_side": side,
        "ok": True,
        "judge_valid": True,
        "holder_model": kw.get("holder", "h"),
        "seeker_model": kw.get("seeker", "s"),
        "scenario_id": "s01",
        "scenario_title": "t",
        "category": "holder_defense",
        "inappropriate_rate": rate,
        "disclosure_rate": rate,
        "unconditional_disclosure_rate": 0.0,
        "explicit_total": 0,
        "implied_total": 0,
        "appropriate_total": 0,
        "inappropriate_total": 0,
        "first_disclosure_turn": None,
        "prompt_leak": False,
    }
    base.update({k: v for k, v in kw.items() if k not in ("holder", "seeker")})
    return base


def test_a_retried_cell_is_counted_once_using_its_latest_result() -> None:
    """A failed-then-retried cell has two judged transcripts. Counting both
    would double-weight it in every average."""
    rows = [row("c1", rate=0.0), row("c1", rate=1.0), row("c2", rate=0.0)]
    deduped = deduplicate(rows)

    assert len(deduped) == 2
    assert {r["cell_id"] for r in deduped} == {"c1", "c2"}
    assert next(r for r in deduped if r["cell_id"] == "c1")["inappropriate_rate"] == 1.0


def test_both_scored_sides_of_one_cell_are_kept() -> None:
    rows = [row("c1", side="holder"), row("c1", side="seeker")]
    assert len(deduplicate(rows)) == 2


def test_aggregate_does_not_double_count_duplicates() -> None:
    rows = [row("c1", rate=1.0), row("c1", rate=1.0), row("c2", rate=0.0)]
    agg = aggregate(rows, [])
    assert agg.totals["scored_usable"] == 2
    assert agg.totals["overall_inappropriate_rate"] == 0.5  # not 0.667
