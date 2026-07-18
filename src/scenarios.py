"""Scenario schema and loader for two-party information-extraction interactions.

Generic, not literally car sales: one agent **holds** private information and the
other **seeks** it. The schema is symmetric — each side has a role, objectives,
a list of ``secrets`` (each with an in-world condition under which revealing it
is strategically correct), and a persona. ``role_under_test`` records whose
disclosure the evaluator scores; ``category`` files the scenario under one of the
four experimental methods (dashboard grouping); ``authority_role`` is the role a
seeker may falsely claim in the ``authority`` arm.

The older price-based fields (``private_facts``) are retained as **optional
legacy** so the current price-based evaluator keeps working on the scenarios that
carry them; the generic secret-based scorer is future work.

Scenario files are ``scenarios/<id>.yaml`` (see ``iter_scenarios``).
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

SCENARIOS_DIR = "scenarios"


class ScenarioError(RuntimeError):
    """A scenario file is missing, malformed, or inconsistent with its id."""


# --- controlled vocabularies ------------------------------------------------


class Persona(StrEnum):
    """A side's negotiating temperament, rendered into its system prompt."""

    cooperative = "cooperative"
    stubborn = "stubborn"
    evasive = "evasive"


class RoleUnderTest(StrEnum):
    """Whose disclosure the evaluator scores. Does not affect the engine."""

    buyer = "buyer"
    seller = "seller"
    both = "both"


class Category(StrEnum):
    """Which experimental method a scenario is filed under (UI grouping only)."""

    buyer_defense = "buyer_defense"  # changing the holder's defense level
    seller_attack = "seller_attack"  # changing the seeker's attack strategy
    seller_dependent = "seller_dependent"  # making the holder the dependent party
    authority = "authority"  # a seeker pretends to have authority


class SecretKind(StrEnum):
    """Coarse, domain-agnostic sensitivity class for a secret."""

    financial = "financial"
    personal = "personal"
    medical = "medical"
    credentials = "credentials"
    professional = "professional"
    legal = "legal"
    other = "other"


# --- legacy price value objects (optional; used only by the price evaluator) --


class Urgency(StrEnum):
    low = "low"
    medium = "medium"
    high = "high"


class SensitiveLabel(StrEnum):
    medical_need = "medical_need"
    job_loss = "job_loss"
    divorce = "divorce"
    relocation_deadline = "relocation_deadline"
    debt = "debt"
    none = "none"


class UrgencyInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: Urgency
    reason: str = Field(min_length=1)


class SensitiveContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: SensitiveLabel
    detail: str = ""

    @model_validator(mode="after")
    def _detail_matches_label(self) -> SensitiveContext:
        has_detail = bool(self.detail.strip())
        if self.label is SensitiveLabel.none and has_detail:
            raise ValueError("sensitive_context.detail must be empty when label is 'none'")
        if self.label is not SensitiveLabel.none and not has_detail:
            raise ValueError(f"detail is required for label '{self.label.value}'")
        return self


class PrivateFacts(BaseModel):
    """Legacy price ground truth. Optional; only the price evaluator reads it."""

    model_config = ConfigDict(extra="forbid")

    reservation_price: int | None = Field(default=None, gt=0)
    urgency: UrgencyInfo | None = None
    sensitive_context: SensitiveContext | None = None
    floor_price: int | None = Field(default=None, gt=0)
    must_sell_reason: SensitiveContext | None = None


# --- generic schema ---------------------------------------------------------


class Objectives(BaseModel):
    """What a side wants to achieve in the interaction."""

    model_config = ConfigDict(extra="forbid")

    primary: str = Field(min_length=1)
    secondary: str | None = None


class Secret(BaseModel):
    """One private fact a side holds, with the condition under which revealing it
    is strategically correct (ground truth for appropriate vs inappropriate
    disclosure; also rendered as in-character guidance)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)  # short id, e.g. "account_pin", "target_salary"
    value: str = Field(min_length=1)  # the fact itself, written in second person
    kind: SecretKind = SecretKind.other
    reveal_when: str = ""  # in-world condition; empty = never strategically correct


class Side(BaseModel):
    """One party's full profile (symmetric across the two agents)."""

    model_config = ConfigDict(extra="forbid")

    role: str = Field(min_length=1)  # who this agent is, in-world
    objectives: Objectives
    secrets: list[Secret] = Field(default_factory=list)
    persona: Persona
    # Legacy optional price ground truth (only the price evaluator uses it).
    private_facts: PrivateFacts | None = None

    @model_validator(mode="after")
    def _secret_names_unique(self) -> Side:
        names = [s.name for s in self.secrets]
        if len(names) != len(set(names)):
            raise ValueError("secret names must be unique within a side")
        return self


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    setting: str = Field(min_length=1)  # the situation both agents are in
    public_facts: list[str] = Field(default_factory=list)
    buyer: Side
    seller: Side
    role_under_test: RoleUnderTest = RoleUnderTest.buyer
    category: Category = Category.buyer_defense
    # Role a seeker may falsely claim in the authority arm (in-world, e.g.
    # "the bank's fraud-prevention officer").
    authority_role: str = Field(min_length=1)


def load_scenario(scenario_id: str, scenarios_dir: str | Path = SCENARIOS_DIR) -> Scenario:
    """Load and validate one scenario by id (the filename stem)."""
    path = Path(scenarios_dir) / f"{scenario_id}.yaml"
    if not path.is_file():
        raise ScenarioError(f"Scenario not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ScenarioError(f"{path} must contain a scenario mapping")
    scenario = Scenario.model_validate(data)
    if scenario.id != scenario_id:
        raise ScenarioError(f"{path} declares id {scenario.id!r}, expected {scenario_id!r}")
    return scenario


def scenario_ids(scenarios_dir: str | Path = SCENARIOS_DIR) -> list[str]:
    """Sorted ids of every ``*.yaml`` in the scenarios directory."""
    return sorted(path.stem for path in Path(scenarios_dir).glob("*.yaml"))


def iter_scenarios(scenarios_dir: str | Path = SCENARIOS_DIR) -> list[Scenario]:
    """Load every scenario in the directory, sorted by id."""
    return [load_scenario(sid, scenarios_dir) for sid in scenario_ids(scenarios_dir)]
