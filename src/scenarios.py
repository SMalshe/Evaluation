"""Scenario schema and loader for two-party used-car negotiations.

A scenario is fully machine-readable ground truth: public facts both sides
know, plus each side's private information. Controlled vocabularies live here as
enums in one place, so downstream analysis can compute chance baselines against
a fixed label set. Free-text details are always paired with an enum label.

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


class Urgency(StrEnum):
    low = "low"
    medium = "medium"
    high = "high"


class BudgetFlexibility(StrEnum):
    tight = "tight"
    moderate = "moderate"
    flexible = "flexible"


class PressureLevel(StrEnum):
    low = "low"
    medium = "medium"
    high = "high"


class SensitiveLabel(StrEnum):
    """The buyer's private situation, from a fixed set. ``none`` = nothing sensitive."""

    medical_need = "medical_need"
    job_loss = "job_loss"
    divorce = "divorce"
    relocation_deadline = "relocation_deadline"
    debt = "debt"
    none = "none"


class Pretext(StrEnum):
    """The institutional role the ``authority`` adversary falsely claims."""

    financing = "financing"
    insurance = "insurance"
    dealer_compliance = "dealer_compliance"


class UrgencyInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: Urgency
    reason: str = Field(min_length=1)


class SensitiveContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: SensitiveLabel
    detail: str = ""  # one concrete sentence; empty iff label is ``none``

    @model_validator(mode="after")
    def _detail_matches_label(self) -> SensitiveContext:
        has_detail = bool(self.detail.strip())
        if self.label is SensitiveLabel.none and has_detail:
            raise ValueError("sensitive_context.detail must be empty when label is 'none'")
        if self.label is not SensitiveLabel.none and not has_detail:
            raise ValueError(f"sensitive_context.detail is required for label '{self.label.value}'")
        return self


class PublicInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    car: str = Field(min_length=1)
    asking_price: int = Field(gt=0)
    shared_facts: list[str] = Field(default_factory=list)


class BuyerPrivate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reservation_price: int = Field(gt=0)  # the most the buyer will pay
    urgency: UrgencyInfo
    sensitive_context: SensitiveContext
    budget_flexibility: BudgetFlexibility


class SellerPrivate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    floor_price: int = Field(gt=0)  # the least the seller will accept
    inventory_pressure: PressureLevel


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    public: PublicInfo
    buyer_private: BuyerPrivate
    seller_private: SellerPrivate
    # Which authority pretext this scenario uses; rotated across the corpus so a
    # finding is not an artifact of one script. Only the authority arms read it.
    pretext: Pretext


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
