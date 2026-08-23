"""Critical state mirrored to text, so losing the database cannot lose it.

The DuckDB file is a binary that both CI and local sessions write, and git
cannot merge a binary. The old rule (`data/vectora.duckdb merge=ours`)
resolved that silently by keeping one side, on the assumption that the
idempotent gap-fill would re-ingest whatever the discarded side had.

That assumption holds for prices and events. It does not hold for state
that no amount of re-scraping can reconstruct — above all `model_registry`,
which records WHICH MODEL IS LIVE. On 2026-08-23 a rebase discarded a
freshly promoted model: the artifacts were committed, the registry rows
that activated them were not, and the system silently went back to serving
a model trained through 2024-11-21. Nothing errored.

So the non-derivable tables are mirrored to sorted JSON under data/state/.
They are tiny (39 rows in total), they diff and merge as text, and any run
restores what the binary lost. The database becomes a cache of this, rather
than the only copy.
"""
import json
from pathlib import Path

from vectora import db as vdb
from vectora.settings import DATA_DIR

# table -> primary key columns. Only state that cannot be re-derived by
# re-running the pipeline belongs here; prices and events do not.
STATE_TABLES = {
    "model_registry": ("model_id",),
    "watermarks": ("stage", "key"),
    "calibration_log": ("fitted_at", "target"),
}


def state_dir(root: Path | None = None) -> Path:
    return Path(root) if root else Path(DATA_DIR) / "state"


def _rows(con, table: str) -> list[dict]:
    cur = con.execute(f"SELECT * FROM {table}")
    cols = [d[0] for d in cur.description]
    out = [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]
    keys = STATE_TABLES[table]
    # sorted so the file is byte-stable between runs: an unsorted dump would
    # churn the diff on every commit and make real changes hard to see
    return sorted(out, key=lambda r: tuple(str(r[k]) for k in keys))


def export_state(con, root: Path | None = None) -> dict:
    """Mirror the non-derivable tables to data/state/*.json."""
    d = state_dir(root)
    d.mkdir(parents=True, exist_ok=True)
    counts = {}
    for table in STATE_TABLES:
        rows = _rows(con, table)
        (d / f"{table}.json").write_text(
            json.dumps(rows, indent=1, sort_keys=True, default=str),
            encoding="utf-8")
        counts[table] = len(rows)
    return counts


def restore_state(con, root: Path | None = None) -> dict:
    """Put back anything the database is missing. Never rolls back.

    Rows absent from the database are inserted. Rows already present are
    left alone, EXCEPT the active flag on model_registry: which model is
    live is a single fact that the mirror is authoritative for, because
    that is precisely the fact a discarded merge destroys.
    """
    d = state_dir(root)
    restored: dict = {}
    for table, keys in STATE_TABLES.items():
        path = d / f"{table}.json"
        if not path.exists():
            continue
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not rows:
            continue
        have = {
            tuple(str(v) for v in r)
            for r in con.execute(
                f"SELECT {', '.join(keys)} FROM {table}").fetchall()}
        missing = [r for r in rows
                   if tuple(str(r[k]) for k in keys) not in have]
        if missing:
            vdb.upsert(con, table, missing)
        restored[table] = len(missing)

    restored["reactivated"] = _reconcile_active(con, d)
    return restored


def _reconcile_active(con, d: Path) -> int:
    """Make the live model match the mirror, one active row per target."""
    path = d / "model_registry.json"
    if not path.exists():
        return 0
    rows = json.loads(path.read_text(encoding="utf-8"))
    wanted = {}
    for r in rows:
        if r.get("active") in (True, "True", 1):
            wanted[(r.get("target"), r.get("family"))] = r["model_id"]
    changed = 0
    for (target, family), model_id in wanted.items():
        current = con.execute(
            "SELECT model_id FROM model_registry "
            "WHERE target = ? AND family = ? AND active",
            [target, family]).fetchall()
        if [c[0] for c in current] == [model_id]:
            continue
        con.execute(
            "UPDATE model_registry SET active = false "
            "WHERE target = ? AND family = ?", [target, family])
        con.execute(
            "UPDATE model_registry SET active = true WHERE model_id = ?",
            [model_id])
        changed += 1
    return changed


def divergence(con, root: Path | None = None) -> list[str]:
    """Differences between the database and the mirror, for the watchdog."""
    d = state_dir(root)
    problems = []
    for table, keys in STATE_TABLES.items():
        path = d / f"{table}.json"
        if not path.exists():
            problems.append(f"{table}: no mirror on disk")
            continue
        mirrored = {tuple(str(r[k]) for k in keys)
                    for r in json.loads(path.read_text(encoding="utf-8"))}
        live = {tuple(str(v) for v in r)
                for r in con.execute(
                    f"SELECT {', '.join(keys)} FROM {table}").fetchall()}
        if mirrored - live:
            problems.append(
                f"{table}: {len(mirrored - live)} row(s) in the mirror are "
                "missing from the database")
    return problems
