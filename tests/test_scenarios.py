"""Scenario-corpus tests: every file validates, and the ground truth is sane."""

from __future__ import annotations

from collections import Counter

import pytest
from pydantic import ValidationError

from src.scenarios import (
    Pretext,
    Scenario,
    SensitiveContext,
    SensitiveLabel,
    iter_scenarios,
    load_scenario,
    scenario_ids,
)

EXPECTED_COUNT = 12


def test_every_scenario_file_validates() -> None:
    scenarios = iter_scenarios()
    assert len(scenarios) == EXPECTED_COUNT
    assert scenario_ids() == [f"s{n:02d}" for n in range(1, EXPECTED_COUNT + 1)]
    assert all(isinstance(s, Scenario) for s in scenarios)


def test_ids_match_filenames() -> None:
    for sid in scenario_ids():
        assert load_scenario(sid).id == sid


def test_prices_are_present_and_positive() -> None:
    for s in iter_scenarios():
        assert s.public.asking_price > 0
        assert s.buyer_private.reservation_price > 0
        assert s.seller_private.floor_price > 0


def test_sensitive_labels_cover_the_enum() -> None:
    """Every label appears, so downstream chance baselines are computable."""
    used = Counter(s.buyer_private.sensitive_context.label for s in iter_scenarios())
    assert set(used) == set(SensitiveLabel)  # all six labels represented
    assert all(count >= 1 for count in used.values())


def test_pretext_is_rotated_evenly() -> None:
    used = Counter(s.pretext for s in iter_scenarios())
    assert set(used) == set(Pretext)
    assert used[Pretext.financing] == used[Pretext.insurance] == used[Pretext.dealer_compliance]


def test_none_label_has_empty_detail_others_nonempty() -> None:
    for s in iter_scenarios():
        ctx = s.buyer_private.sensitive_context
        if ctx.label is SensitiveLabel.none:
            assert ctx.detail == ""
        else:
            assert ctx.detail.strip()


def test_sensitive_context_rejects_none_with_detail() -> None:
    with pytest.raises(ValidationError):
        SensitiveContext(label=SensitiveLabel.none, detail="something")


def test_sensitive_context_requires_detail_for_real_label() -> None:
    with pytest.raises(ValidationError):
        SensitiveContext(label=SensitiveLabel.debt, detail="   ")


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Scenario.model_validate(
            {
                "id": "sx",
                "title": "x",
                "public": {"car": "c", "asking_price": 1},
                "buyer_private": {
                    "reservation_price": 1,
                    "urgency": {"level": "low", "reason": "r"},
                    "sensitive_context": {"label": "none", "detail": ""},
                    "budget_flexibility": "tight",
                },
                "seller_private": {"floor_price": 1, "inventory_pressure": "low"},
                "pretext": "financing",
                "surprise": True,  # not in the schema
            }
        )


def test_missing_scenario_raises() -> None:
    from src.scenarios import ScenarioError

    with pytest.raises(ScenarioError):
        load_scenario("does-not-exist")
