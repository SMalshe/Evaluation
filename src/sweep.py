"""Sweep a (pairing x scenario x condition) grid, saving one transcript per cell.

A *cell* is one full conversation: a buyer model against a seller model, on one
scenario, under one (defense, adversary) condition. Transcripts are written to
the sweep directory with the condition recorded in ``Transcript.metadata`` - the
same shape the dashboard writes - so each one stays independently evaluable.

The sweep is **resumable**. Every finished cell, success or failure, is appended
to ``index.jsonl``; a rerun skips the cells already recorded there. Cells are
independent, so one failure never stops the sweep.

Models served from localhost get their own serialized lane: a local runtime has
one GPU, so issuing parallel requests to it only adds queueing and timeouts.
Hosted models run several at a time.

    python -m src.sweep --dry-run     # print the plan; calls nothing, needs no keys
    python -m src.sweep               # execute it
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import signal
import sys
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import BaseModel

from .engine import Agent, Transcript, run_conversation
from .models import ModelClient, ModelConfig, RegistryError, get_client, load_registry
from .persistence import save_transcript
from .prompts import (
    DEFAULT_PROMPT_CONFIG,
    SCENARIO_OPENING_PROMPT,
    AdversaryStrategy,
    DefenseCondition,
    PromptConfig,
    available_adversaries,
    opening_speaker,
    render_pair,
)
from .scenarios import Category, Scenario, ScenarioError, load_scenario, scenario_ids

BUYER = "buyer"
SELLER = "seller"

DEFAULT_OUT = "sweeps"
INDEX_NAME = "index.jsonl"
DEFAULT_MAX_TURNS = 12

# Base URLs pointing at these hosts share the serialized local lane.
LOCAL_HOSTS: tuple[str, ...] = ("localhost", "127.0.0.1", "0.0.0.0")

# The default plan: every ordered pairing among the local models (including each
# against itself), plus self-play for the two hosted rungs. ollama-14b is
# registered but deliberately absent - it is meant to be swept on other hardware
# via --cross ollama-3b,ollama-8b,ollama-14b.
DEFAULT_CROSS: tuple[str, ...] = ("ollama-3b", "ollama-8b")
DEFAULT_SELF: tuple[str, ...] = ("claude-sonnet", "gpt-mid")


class ConditionMode(StrEnum):
    """How many (defense, adversary) conditions each scenario contributes."""

    natural = "natural"  # one condition, chosen by the scenario's category
    grid = "grid"  # every defense x every enabled adversary
    defense = "defense"  # all three defenses at a fixed adversary
    adversary = "adversary"  # every enabled adversary at a fixed defense
    fixed = "fixed"  # exactly the --defense/--adversary given


# One (defense, adversary) per category for ConditionMode.natural. This is an
# editable research choice - each category names the factor it varies, so the
# single-condition default puts that factor in its characteristic setting and
# leaves the others at their baseline.
CATEGORY_CONDITIONS: dict[Category, tuple[DefenseCondition, AdversaryStrategy]] = {
    Category.buyer_defense: (DefenseCondition.basic, AdversaryStrategy.direct_probe),
    Category.seller_attack: (DefenseCondition.none, AdversaryStrategy.pressure),
    Category.authority: (DefenseCondition.none, AdversaryStrategy.authority),
    Category.seller_dependent: (DefenseCondition.none, AdversaryStrategy.rapport),
}


@dataclass(frozen=True)
class Cell:
    """One unit of work; ``key`` identifies it across resumed runs."""

    scenario_id: str
    buyer_model: str
    seller_model: str
    defense: DefenseCondition
    adversary: AdversaryStrategy
    repeat: int = 0

    @property
    def _parts(self) -> tuple[str, ...]:
        return (
            self.scenario_id,
            self.buyer_model,
            self.seller_model,
            self.defense.value,
            self.adversary.value,
            str(self.repeat),
        )

    @property
    def key(self) -> str:
        return "|".join(self._parts)

    @property
    def filename(self) -> str:
        """Deterministic transcript name for this cell.

        Every cell names its agents buyer/seller, so timestamp-based names would
        collide under concurrency; deriving the name from the cell instead makes
        it unique by construction, self-describing on disk, and stable when a
        failed cell is rerun (it overwrites its own file rather than piling up).
        """
        stem = "__".join(re.sub(r"[^A-Za-z0-9_.-]+", "_", part) for part in self._parts)
        return f"{stem}.json"


class CellRecord(BaseModel):
    """One line of ``index.jsonl``: the outcome of a single cell."""

    key: str
    scenario_id: str
    buyer_model: str
    seller_model: str
    defense: str
    adversary: str
    repeat: int
    status: Literal["ok", "error"]
    transcript: str | None = None
    termination: str | None = None
    turns: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None
    elapsed_s: float = 0.0
    error: str | None = None
    finished_at: datetime


ClientFactory = Callable[[str], ModelClient]


def pairings(
    cross: Sequence[str] = (),
    self_play: Sequence[str] = (),
    explicit: Sequence[tuple[str, str]] = (),
) -> list[tuple[str, str]]:
    """Build the (buyer, seller) list: every ordered pair within ``cross``, then
    self-play for each of ``self_play``, then any ``explicit`` pairs. Order is
    preserved and duplicates are dropped."""
    pairs: list[tuple[str, str]] = []
    pairs.extend(itertools.product(cross, repeat=2))
    pairs.extend((name, name) for name in self_play)
    pairs.extend(explicit)

    seen: set[tuple[str, str]] = set()
    unique: list[tuple[str, str]] = []
    for pair in pairs:
        if pair not in seen:
            seen.add(pair)
            unique.append(pair)
    return unique


def conditions_for(
    scenario: Scenario,
    mode: ConditionMode,
    *,
    defense: DefenseCondition = DefenseCondition.none,
    adversary: AdversaryStrategy = AdversaryStrategy.direct_probe,
    config: PromptConfig = DEFAULT_PROMPT_CONFIG,
) -> list[tuple[DefenseCondition, AdversaryStrategy]]:
    """The (defense, adversary) conditions one scenario contributes under ``mode``.

    Raises ``ValueError`` if a mode that takes an explicit adversary is handed a
    gated one - better to fail while planning than partway through a sweep.
    """
    adversaries = available_adversaries(config)
    if mode is ConditionMode.natural:
        return [CATEGORY_CONDITIONS[scenario.category]]
    if mode is ConditionMode.grid:
        return [(d, a) for d in DefenseCondition for a in adversaries]
    if mode is ConditionMode.adversary:
        return [(defense, a) for a in adversaries]

    if adversary not in adversaries:
        raise ValueError(
            f"adversary {adversary.value!r} is not enabled; "
            "it is gated behind PromptConfig.enable_authority_verifiable"
        )
    if mode is ConditionMode.defense:
        return [(d, adversary) for d in DefenseCondition]
    return [(defense, adversary)]


def build_plan(
    scenarios: Sequence[Scenario],
    pairs: Sequence[tuple[str, str]],
    mode: ConditionMode,
    *,
    defense: DefenseCondition = DefenseCondition.none,
    adversary: AdversaryStrategy = AdversaryStrategy.direct_probe,
    repeats: int = 1,
    config: PromptConfig = DEFAULT_PROMPT_CONFIG,
) -> list[Cell]:
    """Expand (pairing x scenario x condition x repeat) into an ordered cell list.

    Scenario is the outer loop so an interrupted sweep still covers pairings
    evenly rather than finishing one model and none of the others.
    """
    cells: list[Cell] = []
    for scenario in scenarios:
        conditions = conditions_for(
            scenario, mode, defense=defense, adversary=adversary, config=config
        )
        for buyer_model, seller_model in pairs:
            for cell_defense, cell_adversary in conditions:
                for repeat in range(repeats):
                    cells.append(
                        Cell(
                            scenario_id=scenario.id,
                            buyer_model=buyer_model,
                            seller_model=seller_model,
                            defense=cell_defense,
                            adversary=cell_adversary,
                            repeat=repeat,
                        )
                    )
    return cells


def is_local(config: ModelConfig) -> bool:
    """Whether an entry is served from this machine (gets the serialized lane)."""
    base_url = config.base_url or ""
    return any(host in base_url for host in LOCAL_HOSTS)


def cell_cost(transcript: Transcript, registry: dict[str, ModelConfig], cell: Cell) -> float:
    """USD for one conversation, billing each turn to the model that spoke it."""
    by_speaker = {
        BUYER: registry.get(cell.buyer_model),
        SELLER: registry.get(cell.seller_model),
    }
    total = 0.0
    for turn in transcript.turns:
        config = by_speaker.get(turn.speaker)
        if config is None:
            continue
        total += turn.prompt_tokens / 1e6 * config.price_per_mtok_in
        total += turn.completion_tokens / 1e6 * config.price_per_mtok_out
    return total


class ClientCache:
    """Builds each registry entry's client once and shares it across threads.

    The provider SDKs are safe for concurrent requests; rebuilding a client per
    cell would re-read the registry and open a new connection pool every time.
    """

    def __init__(self, factory: ClientFactory) -> None:
        self._factory = factory
        self._clients: dict[str, ModelClient] = {}
        self._lock = threading.Lock()

    def get(self, name: str) -> ModelClient:
        with self._lock:
            if name not in self._clients:
                self._clients[name] = self._factory(name)
            return self._clients[name]


class SweepIndex:
    """Append-only record of finished cells, used to resume."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def done_keys(self, *, include_failed: bool) -> set[str]:
        """Keys to skip. Failed cells are retried unless ``include_failed``."""
        if not self.path.is_file():
            return set()
        keys: set[str] = set()
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # a partial line from a hard kill; ignore and redo the cell
            if row.get("status") == "ok" or include_failed:
                keys.add(row["key"])
        return keys

    def append(self, record: CellRecord) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(record.model_dump_json() + "\n")


def run_cell(
    cell: Cell,
    scenario: Scenario,
    clients: ClientCache,
    registry: dict[str, ModelConfig],
    *,
    runs_dir: Path,
    max_turns: int,
    sweep_name: str,
    config: PromptConfig = DEFAULT_PROMPT_CONFIG,
    cancelled: Callable[[], bool] | None = None,
) -> CellRecord:
    """Execute one cell and return its record. Never raises: a failure becomes a
    record with ``status="error"`` so the sweep can carry on."""
    started = time.monotonic()
    base = {
        "key": cell.key,
        "scenario_id": cell.scenario_id,
        "buyer_model": cell.buyer_model,
        "seller_model": cell.seller_model,
        "defense": cell.defense.value,
        "adversary": cell.adversary.value,
        "repeat": cell.repeat,
    }
    try:
        buyer_system, seller_system = render_pair(scenario, cell.defense, cell.adversary, config)
        buyer = Agent(name=BUYER, system_prompt=buyer_system, client=clients.get(cell.buyer_model))
        seller = Agent(
            name=SELLER, system_prompt=seller_system, client=clients.get(cell.seller_model)
        )
        transcript = run_conversation(
            buyer,
            seller,
            max_turns=max_turns,
            opening_speaker=opening_speaker(scenario),
            opening_prompt=SCENARIO_OPENING_PROMPT,
            cancelled=cancelled,
            # Same keys the dashboard records, so a swept transcript is
            # self-describing and can be evaluated later on its own.
            metadata={
                "scenario_id": cell.scenario_id,
                "defense": cell.defense.value,
                "adversary": cell.adversary.value,
                "sweep": sweep_name,
                "repeat": cell.repeat,
            },
        )
        path = save_transcript(transcript, runs_dir, filename=cell.filename)
        return CellRecord(
            **base,
            status="ok",
            transcript=str(path),
            termination=transcript.termination,
            turns=len(transcript.turns),
            prompt_tokens=sum(t.prompt_tokens for t in transcript.turns),
            completion_tokens=sum(t.completion_tokens for t in transcript.turns),
            cost_usd=cell_cost(transcript, registry, cell),
            elapsed_s=time.monotonic() - started,
            finished_at=datetime.now(UTC),
        )
    except Exception as exc:  # noqa: BLE001 - one cell must never kill the sweep
        return CellRecord(
            **base,
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            elapsed_s=time.monotonic() - started,
            finished_at=datetime.now(UTC),
        )


@dataclass
class SweepReport:
    """Totals for one sweep invocation."""

    completed: int = 0
    failed: int = 0
    skipped: int = 0
    cost_usd: float = 0.0
    elapsed_s: float = 0.0


def run_sweep(
    cells: Sequence[Cell],
    scenarios: dict[str, Scenario],
    registry: dict[str, ModelConfig],
    *,
    out_dir: Path,
    sweep_name: str,
    client_factory: ClientFactory,
    max_turns: int = DEFAULT_MAX_TURNS,
    local_workers: int = 1,
    remote_workers: int = 4,
    resume: bool = True,
    redo_failed: bool = False,
    config: PromptConfig = DEFAULT_PROMPT_CONFIG,
    stop: threading.Event | None = None,
    on_record: Callable[[CellRecord], None] | None = None,
) -> SweepReport:
    """Execute ``cells``, writing transcripts and an index under ``out_dir``.

    Local-model cells run in a serialized lane; hosted cells run ``remote_workers``
    at a time. ``stop`` (when set) makes in-flight conversations end at their next
    turn boundary and leaves the remaining cells unstarted.
    """
    runs_dir = out_dir / "runs"
    index = SweepIndex(out_dir / INDEX_NAME)
    clients = ClientCache(client_factory)
    halt = stop or threading.Event()
    report = SweepReport()
    started = time.monotonic()

    pending = list(cells)
    if resume:
        done = index.done_keys(include_failed=not redo_failed)
        before = len(pending)
        pending = [cell for cell in pending if cell.key not in done]
        report.skipped = before - len(pending)

    local_cells = [
        cell
        for cell in pending
        if is_local(registry[cell.buyer_model]) or is_local(registry[cell.seller_model])
    ]
    local_keys = {cell.key for cell in local_cells}
    remote_cells = [cell for cell in pending if cell.key not in local_keys]

    def work(cell: Cell) -> CellRecord | None:
        if halt.is_set():
            return None
        record = run_cell(
            cell,
            scenarios[cell.scenario_id],
            clients,
            registry,
            runs_dir=runs_dir,
            max_turns=max_turns,
            sweep_name=sweep_name,
            config=config,
            cancelled=halt.is_set,
        )
        # A conversation the stop signal cut short is incomplete, not done. Keep
        # it out of the index so a resumed sweep reruns it from the start; its
        # partial transcript is overwritten then (the filename is deterministic).
        if record.termination == "cancelled":
            return None
        index.append(record)
        return record

    futures: list[Future[CellRecord | None]] = []
    with (
        ThreadPoolExecutor(max_workers=max(1, local_workers), thread_name_prefix="local") as local,
        ThreadPoolExecutor(
            max_workers=max(1, remote_workers), thread_name_prefix="remote"
        ) as remote,
    ):
        futures.extend(local.submit(work, cell) for cell in local_cells)
        futures.extend(remote.submit(work, cell) for cell in remote_cells)
        for future in as_completed(futures):
            record = future.result()
            if record is None:
                continue
            if record.status == "ok":
                report.completed += 1
                report.cost_usd += record.cost_usd or 0.0
            else:
                report.failed += 1
            if on_record is not None:
                on_record(record)

    report.elapsed_s = time.monotonic() - started
    return report


# --- CLI -------------------------------------------------------------------


def _split(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_pair(value: str) -> tuple[str, str]:
    buyer, sep, seller = value.partition(":")
    if not sep or not buyer.strip() or not seller.strip():
        raise argparse.ArgumentTypeError(f"--pair expects BUYER:SELLER, got {value!r}")
    return buyer.strip(), seller.strip()


def _describe_plan(
    cells: Sequence[Cell], registry: dict[str, ModelConfig], pairs: Sequence[tuple[str, str]]
) -> str:
    paid = [
        cell
        for cell in cells
        if registry[cell.buyer_model].price_per_mtok_out > 0
        or registry[cell.seller_model].price_per_mtok_out > 0
    ]
    scenarios = sorted({cell.scenario_id for cell in cells})
    conditions = sorted({(c.defense.value, c.adversary.value) for c in cells})
    lines = [
        f"cells:      {len(cells)}  ({len(cells) - len(paid)} free / {len(paid)} billed)",
        f"scenarios:  {len(scenarios)}",
        f"pairings:   {len(pairs)}",
        f"conditions: {len(conditions)} distinct -> "
        + ", ".join(f"{d}/{a}" for d, a in conditions[:6])
        + (" ..." if len(conditions) > 6 else ""),
        "",
        "per pairing:",
    ]
    for buyer_model, seller_model in pairs:
        count = sum(
            1 for c in cells if c.buyer_model == buyer_model and c.seller_model == seller_model
        )
        local = is_local(registry[buyer_model]) and is_local(registry[seller_model])
        lane = "local" if local else "hosted"
        lines.append(
            f"  {buyer_model:>14} (buyer) vs {seller_model:<14} (seller)  {count:>5} cells  [{lane}]"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv(override=False)
    parser = argparse.ArgumentParser(
        prog="python -m src.sweep",
        description="Run a (pairing x scenario x condition) grid of conversations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Default plan: every ordered pairing among "
            + ", ".join(DEFAULT_CROSS)
            + "; self-play for "
            + ", ".join(DEFAULT_SELF)
            + ".\nStart with --dry-run: it prints the plan and calls nothing."
        ),
    )
    parser.add_argument("--out", default=DEFAULT_OUT, help="sweep output root (default: sweeps/)")
    parser.add_argument("--name", default=None, help="sweep name (default: a UTC timestamp)")
    parser.add_argument("--registry", default="models.yaml")
    parser.add_argument("--scenarios-dir", default="scenarios")
    parser.add_argument(
        "--scenarios", default=None, help="comma-separated ids (default: every scenario)"
    )
    parser.add_argument(
        "--cross",
        action="append",
        default=[],
        help="comma-separated models; runs every ordered pair within the group (repeatable)",
    )
    parser.add_argument(
        "--self",
        dest="self_play",
        action="append",
        default=[],
        help="self-play a model (repeatable)",
    )
    parser.add_argument(
        "--pair",
        action="append",
        type=_parse_pair,
        default=[],
        help="an explicit BUYER:SELLER pairing (repeatable)",
    )
    parser.add_argument(
        "--conditions",
        default=ConditionMode.natural.value,
        choices=[mode.value for mode in ConditionMode],
        help="how many conditions per scenario (default: natural, one per category)",
    )
    parser.add_argument(
        "--defense",
        default=DefenseCondition.none.value,
        choices=[d.value for d in DefenseCondition],
        help="fixed defense, for --conditions fixed/adversary",
    )
    parser.add_argument(
        "--adversary",
        default=AdversaryStrategy.direct_probe.value,
        choices=[a.value for a in AdversaryStrategy],
        help="fixed adversary, for --conditions fixed/defense",
    )
    parser.add_argument("--enable-authority-verifiable", action="store_true")
    parser.add_argument("--repeats", type=int, default=1, help="runs per cell (default: 1)")
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument("--local-workers", type=int, default=1, help="parallel local cells")
    parser.add_argument("--remote-workers", type=int, default=4, help="parallel hosted cells")
    parser.add_argument(
        "--limit", type=int, default=None, help="cap the number of cells (smoke test)"
    )
    parser.add_argument("--no-resume", action="store_true", help="rerun cells already in the index")
    parser.add_argument("--redo-failed", action="store_true", help="also retry cells that errored")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = parser.parse_args(argv)

    config = PromptConfig(enable_authority_verifiable=args.enable_authority_verifiable)

    try:
        registry = load_registry(args.registry)
    except RegistryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    cross: list[str] = []
    for group in args.cross:
        cross.extend(_split(group))
    self_play: list[str] = []
    for group in args.self_play:
        self_play.extend(_split(group))
    if not cross and not self_play and not args.pair:
        cross, self_play = list(DEFAULT_CROSS), list(DEFAULT_SELF)

    pairs = pairings(cross, self_play, args.pair)
    unknown = sorted({m for pair in pairs for m in pair} - set(registry))
    if unknown:
        print(
            f"error: unknown model(s): {', '.join(unknown)}\n"
            f"available: {', '.join(sorted(registry))}",
            file=sys.stderr,
        )
        return 2

    ids = _split(args.scenarios) if args.scenarios else scenario_ids(args.scenarios_dir)
    try:
        scenarios = {sid: load_scenario(sid, args.scenarios_dir) for sid in ids}
    except ScenarioError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        cells = build_plan(
            list(scenarios.values()),
            pairs,
            ConditionMode(args.conditions),
            defense=DefenseCondition(args.defense),
            adversary=AdversaryStrategy(args.adversary),
            repeats=max(1, args.repeats),
            config=config,
        )
    except ValueError as exc:  # a gated adversary was requested
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.limit is not None:
        cells = cells[: args.limit]

    sweep_name = args.name or datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out) / sweep_name

    print(_describe_plan(cells, registry, pairs))
    print(f"\noutput:     {out_dir}")
    print(f"max turns:  {args.max_turns}")
    if args.dry_run:
        print("\ndry run - nothing was called.")
        return 0

    halt = threading.Event()

    def handle_interrupt(signum: int, frame: Any) -> None:
        if halt.is_set():
            print("\nsecond interrupt - exiting now.", file=sys.stderr)
            raise KeyboardInterrupt
        print(
            "\ninterrupt - finishing in-flight conversations, skipping the rest. Rerun to resume.",
            file=sys.stderr,
        )
        halt.set()

    previous = signal.signal(signal.SIGINT, handle_interrupt)
    total = len(cells)
    done = 0

    def progress(record: CellRecord) -> None:
        nonlocal done
        done += 1
        flag = "ok " if record.status == "ok" else "ERR"
        detail = (
            f"{record.turns} turns, {record.termination}"
            if record.status == "ok"
            else (record.error or "")[:70]
        )
        print(
            f"[{done}/{total}] {flag} {record.scenario_id} "
            f"{record.buyer_model}/{record.seller_model} "
            f"{record.defense}/{record.adversary}  {detail}"
        )

    print()
    try:
        report = run_sweep(
            cells,
            scenarios,
            registry,
            out_dir=out_dir,
            sweep_name=sweep_name,
            client_factory=lambda name: get_client(name, args.registry),
            max_turns=args.max_turns,
            local_workers=args.local_workers,
            remote_workers=args.remote_workers,
            resume=not args.no_resume,
            redo_failed=args.redo_failed,
            config=config,
            stop=halt,
            on_record=progress,
        )
    finally:
        signal.signal(signal.SIGINT, previous)

    print(
        f"\ncompleted {report.completed}, failed {report.failed}, "
        f"skipped {report.skipped} in {report.elapsed_s:.0f}s"
    )
    print(f"billed roughly ${report.cost_usd:.2f}")
    print(f"index: {out_dir / INDEX_NAME}")
    return 1 if report.failed and not report.completed else 0


if __name__ == "__main__":
    raise SystemExit(main())
