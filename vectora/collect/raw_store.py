"""Immutable raw layer: every scrape saved before parsing, with metadata."""
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


def save_raw(raw_dir: Path, source: str, run_date: str, name: str, payload: str) -> Path:
    day_dir = Path(raw_dir) / source / run_date
    day_dir.mkdir(parents=True, exist_ok=True)
    path = day_dir / f"{name}.html"
    path.write_text(payload, encoding="utf-8")
    meta = {
        "source": source,
        "run_date": run_date,
        "fetched_at": datetime.now(UTC).isoformat(),
        "bytes": len(payload.encode("utf-8")),
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return path
