"""Save/load conversation transcripts as JSON under a runs directory."""

from __future__ import annotations

import re
from pathlib import Path

from .engine import Transcript


def save_transcript(transcript: Transcript, runs_dir: str | Path = "runs") -> Path:
    """Write a transcript to ``runs_dir`` and return the file path."""
    directory = Path(runs_dir)
    directory.mkdir(parents=True, exist_ok=True)

    stamp = transcript.started_at.strftime("%Y%m%d-%H%M%S")
    participants = "-vs-".join(_slug(info.name) for info in transcript.agents)
    base = f"{stamp}-{participants}"

    path = directory / f"{base}.json"
    counter = 2
    while path.exists():
        path = directory / f"{base}-{counter}.json"
        counter += 1

    path.write_text(transcript.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_transcript(path: str | Path) -> Transcript:
    return Transcript.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name)
