"""Scenario-corpus tests: every file validates, and the ground truth is sane."""

from __future__ import annotations

from collections import Counter

import pytest
from pydantic import ValidationError

from src.scenarios import (
    Category,
    Persona,
    RoleUnderTest,
    Scenario,
    Secret,
    Side,
    iter_scenarios,
    load_scenario,
    scenario_ids,
)

EXPECTED_COUNT = 48


def test_every_scenario_file_validates() -> None:
    scenarios = iter_scenarios()
    assert len(scenarios) == EXPECTED_COUNT
    assert scenario_ids() == [f"s{n:02d}" for n in range(1, EXPECTED_COUNT + 1)]
    assert all(isinstance(s, Scenario) for s in scenarios)


def test_ids_match_filenames() -> None:
    for sid in scenario_ids():
        assert load_scenario(sid).id == sid


def test_twelve_per_category() -> None:
    used = Counter(s.category for s in iter_scenarios())
    assert set(used) == set(Category)
    assert all(count == 12 for count in used.values())  # 12 situations per method


def test_the_holder_is_under_test_in_every_category() -> None:
    """The holder is by definition the side whose disclosure is scored, in all
    four categories - including holder_dependent, where the holder is also the
    party under pressure."""
    for s in iter_scenarios():
        assert s.role_under_test is RoleUnderTest.holder, s.id


def test_the_holder_holds_the_secrets_and_the_seeker_holds_none() -> None:
    for s in iter_scenarios():
        assert s.holder.secrets, f"{s.id}: the holder holds no secrets"
        assert not s.seeker.secrets, f"{s.id}: the seeker should hold no secrets"


def test_every_scenario_has_an_authority_role() -> None:
    for s in iter_scenarios():
        assert s.authority_role.strip()


def test_domains_are_diverse() -> None:
    """The settings should not all be the same domain (no 48 car listings)."""
    settings = [s.setting.lower() for s in iter_scenarios()]
    car_settings = sum("car" in text or "dealership" in text for text in settings)
    assert car_settings <= 2  # at most a token car scenario or two


def test_personas_are_varied() -> None:
    holder_personas = {
        (s.seeker if s.role_under_test is RoleUnderTest.seeker else s.holder).persona
        for s in iter_scenarios()
    }
    assert holder_personas == set(Persona)


def test_secret_names_must_be_unique_within_a_side() -> None:
    with pytest.raises(ValidationError, match="unique"):
        Side.model_validate(
            {
                "role": "x",
                "objectives": {"primary": "y"},
                "persona": "cooperative",
                "secrets": [
                    {"name": "dup", "value": "a"},
                    {"name": "dup", "value": "b"},
                ],
            }
        )


def test_secret_defaults() -> None:
    secret = Secret(name="pin", value="1234")
    assert secret.reveal_when == ""  # empty = never appropriate
    assert secret.kind.value == "other"


def test_default_role_and_category() -> None:
    scenario = Scenario.model_validate(
        {
            "id": "sx",
            "title": "x",
            "setting": "a situation",
            "holder": {"role": "a", "objectives": {"primary": "p"}, "persona": "cooperative"},
            "seeker": {"role": "b", "objectives": {"primary": "q"}, "persona": "cooperative"},
            "authority_role": "an officer",
        }
    )
    assert scenario.role_under_test is RoleUnderTest.holder
    assert scenario.category is Category.holder_defense


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Scenario.model_validate(
            {
                "id": "sx",
                "title": "x",
                "setting": "s",
                "holder": {"role": "a", "objectives": {"primary": "p"}, "persona": "cooperative"},
                "seeker": {"role": "b", "objectives": {"primary": "q"}, "persona": "cooperative"},
                "authority_role": "an officer",
                "surprise": True,
            }
        )


def test_missing_scenario_raises() -> None:
    from src.scenarios import ScenarioError

    with pytest.raises(ScenarioError):
        load_scenario("does-not-exist")
