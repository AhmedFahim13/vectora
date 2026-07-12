"""Immutable raw layer: every scrape saved (gzipped) before parsing, with metadata.

Re-running the same source/date/name overwrites the previous file — this is
the idempotent-rerun design; page content is assumed stable within a date.
"""
import gzip
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


def save_raw(
    raw_dir: Path,
    source: str,
    run_date: str,
    name: str,
    payload: str,
    url: str | None = None,
) -> Path:
    day_dir = Path(raw_dir) / source / run_date
    day_dir.mkdir(parents=True, exist_ok=True)
    data = payload.encode("utf-8")
    # mtime=0 keeps the gzip bytes deterministic so unchanged content
    # doesn't produce a new git blob on re-runs
    compressed = gzip.compress(data, mtime=0)
    path = day_dir / f"{name}.html.gz"
    if path.exists() and path.read_bytes() == compressed:
        return path  # unchanged content: keep original meta, no new git blobs
    path.write_bytes(compressed)
    meta = {
        "source": source,
        "run_date": run_date,
        "url": url,
        "fetched_at": datetime.now(UTC).isoformat(),
        "bytes": len(data),
        "compressed_bytes": len(compressed),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    (day_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return path
