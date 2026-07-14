"""Round-trip tests for transcript persistence."""

from __future__ import annotations

from pathlib import Path

from src.engine import run_conversation
from src.persistence import load_transcript, save_transcript
from tests.test_engine import make_agent


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    alice = make_agent("alice", ["hello", "sure [DEAL $10]"])
    bob = make_agent("bob", ["hi there"])
    transcript = run_conversation(alice, bob, max_turns=6)

    path = save_transcript(transcript, runs_dir=tmp_path)

    assert path.parent == tmp_path
    assert path.suffix == ".json"
    loaded = load_transcript(path)
    assert loaded == transcript


def test_save_avoids_filename_collisions(tmp_path: Path) -> None:
    alice = make_agent("alice", ["a"])
    bob = make_agent("bob", ["b"])
    transcript = run_conversation(alice, bob, max_turns=2)

    first = save_transcript(transcript, runs_dir=tmp_path)
    second = save_transcript(transcript, runs_dir=tmp_path)

    assert first != second
    assert load_transcript(second) == transcript
