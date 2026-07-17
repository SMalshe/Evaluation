"""Preview CLI tests (offline: never uses --run, so no network/keys)."""

from __future__ import annotations

import pytest

from src.preview import main


def run(argv: list[str], capsys: pytest.CaptureFixture[str]) -> str:
    assert main(argv) == 0
    return capsys.readouterr().out


def test_prints_both_prompts(capsys: pytest.CaptureFixture[str]) -> None:
    out = run(["--scenario", "s01", "--defense", "basic", "--adversary", "rapport"], capsys)
    assert "BUYER SYSTEM PROMPT" in out
    assert "SELLER SYSTEM PROMPT" in out
    assert "SCENARIO s01" in out
    assert "rapport" in out  # the adversary tactic is rendered into the seller prompt


def test_authority_verifiable_is_gated(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["--scenario", "s01", "--adversary", "authority_verifiable"])

    out = run(
        [
            "--scenario",
            "s01",
            "--adversary",
            "authority_verifiable",
            "--enable-authority-verifiable",
        ],
        capsys,
    )
    assert "reference or case number" in out  # buyer defense clause is present


def test_unknown_scenario_errors() -> None:
    with pytest.raises(SystemExit):
        main(["--scenario", "s99"])


def test_run_requires_models() -> None:
    with pytest.raises(SystemExit):
        main(["--scenario", "s01", "--run"])  # no --model-a/--model-b
