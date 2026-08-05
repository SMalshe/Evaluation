# Legacy

Superseded material, kept for provenance rather than use. Nothing here is
imported by `src/`, referenced by the README, or exercised by the tests — it can
be deleted once you are sure you don't need it.

| Path | What it is | Why it moved |
| ---- | ---------- | ------------ |
| `runs-pre-rename/` | 33 transcripts named `*-buyer-vs-seller.json` | Written before the schema renamed `buyer`/`seller` to `holder`/`seeker`. Their `agents[].name` fields no longer match what `src/disclosure.py` and `src/subliminal_chat.py` look up, so they cannot be scored by the current code. |
| `architecture-orphan-copy.png` | A second copy of the architecture diagram | `scripts/architecture_diagram.py` writes `docs/architecture.png`, and that is the path the README embeds. This copy lived in `reports/`, was byte-different from the canonical one, and nothing referenced it. |

## Regenerating rather than restoring

The architecture diagram is generated, so prefer re-running the script over
reviving the copy:

```sh
uv run python scripts/architecture_diagram.py   # -> docs/architecture.png
```

The pre-rename transcripts are not regenerable, but they are also not scoreable.
If you need equivalent data under the current schema, re-run the grid:

```sh
make cloud-plan    # dry-run first, calls nothing
make cloud-grid
```
