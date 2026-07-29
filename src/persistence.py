"""Save/load conversation transcripts as JSON under a runs directory."""

from __future__ import annotations

import re
from pathlib import Path

from .engine import Transcript


def save_transcript(
    transcript: Transcript,
    runs_dir: str | Path = "runs",
    *,
    filename: str | None = None,
) -> Path:
    """Write a transcript to ``runs_dir`` and return the file path.

    With ``filename`` the transcript is written to exactly that name (overwriting
    it), which lets a caller choose a deterministic, collision-free name of its
    own. Otherwise the name is derived from the timestamp and participants, with
    a numeric suffix on collision.

    The auto-named path is claimed with an exclusive create, so concurrent
    writers - the dashboard's run threads, a sweep's worker pool - can't both
    pick the same suffix and have one silently overwrite the other. Agent names
    repeat across runs and the stamp only has second resolution, so that race is
    reachable in practice.
    """
    directory = Path(runs_dir)
    directory.mkdir(parents=True, exist_ok=True)
    payload = transcript.model_dump_json(indent=2)

    if filename is not None:
        path = directory / filename
        path.write_text(payload, encoding="utf-8")
        return path

    stamp = transcript.started_at.strftime("%Y%m%d-%H%M%S")
    participants = "-vs-".join(_slug(info.name) for info in transcript.agents)
    base = f"{stamp}-{participants}"

    path = directory / f"{base}.json"
    counter = 2
    while True:
        try:
            with path.open("x", encoding="utf-8") as handle:
                handle.write(payload)
            return path
        except FileExistsError:
            path = directory / f"{base}-{counter}.json"
            counter += 1


def load_transcript(path: str | Path) -> Transcript:
    return Transcript.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name)
