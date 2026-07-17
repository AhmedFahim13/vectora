"""Obsidian vault generator v1 (spec §7): Journal, signal Prediction notes,
Company notes, Home dashboard. Machine content lives strictly between the
markers; anything a human writes outside them survives regeneration
byte-identical (tested). Notes use [[wiki-links]] so Obsidian's graph view
is the knowledge graph."""
from pathlib import Path

from vectora.settings import VAULT_DIR

MACHINE_BEGIN = "<!-- vectora:begin -->"
MACHINE_END = "<!-- vectora:end -->"


def _write_machine(path: Path, content: str) -> None:
    block = f"{MACHINE_BEGIN}\n{content.rstrip()}\n{MACHINE_END}"
    if path.exists():
        text = path.read_text(encoding="utf-8")
        if MACHINE_BEGIN in text and MACHINE_END in text:
            pre = text.split(MACHINE_BEGIN, 1)[0]
            post = text.split(MACHINE_END, 1)[1]
            new = pre + block + post
        else:
            new = text.rstrip() + "\n\n" + block + "\n"
    else:
        new = block + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(new, encoding="utf-8")


def generate(con, date_str: str, vault_dir: Path = VAULT_DIR) -> dict:
    n = 0
    preds = con.execute(
        """
        SELECT p.id, p.symbol, p.target, p.probability, p.is_signal,
               p.suppressed_reason, e.rendered
        FROM predictions p LEFT JOIN explanations e ON e.prediction_id = p.id
        WHERE p.date = ? ORDER BY p.probability DESC
        """, [date_str]).fetchall()
    signals = [p for p in preds if p[4]]
    quality = con.execute(
        "SELECT score FROM data_quality WHERE date = ? AND source='dse_eod'",
        [date_str]).fetchone()
    events = con.execute(
        """
        SELECT e.symbol, e.title, coalesce(l.event_type, 'unclassified')
        FROM events e LEFT JOIN event_labels l ON l.event_id = e.id
        WHERE e.post_date = ? AND e.symbol IS NOT NULL
          AND coalesce(l.materiality, 1) >= 1
        ORDER BY coalesce(l.materiality, 1) DESC
        LIMIT 20
        """, [date_str]).fetchall()

    # Journal ---------------------------------------------------------------
    regime = con.execute(
        "SELECT regime FROM regimes WHERE date = ?", [date_str]).fetchone()
    q = quality[0] if quality else "n/a"
    reg = regime[0] if regime else "unclassified"
    lines = [f"# Journal {date_str}", "",
             f"{len(preds)} predictions | {len(signals)} signal(s) | "
             f"quality {q} | regime {reg}", ""]
    if signals:
        lines.append("## Signals")
        lines += [f"- [[{s[1]}]] {s[2]} at {s[3]:.0%} "
                  f"([[Predictions/{date_str[:7]}/{s[0]}|note]])"
                  for s in signals]
    if events:
        lines.append("")
        lines.append("## Company events")
        lines += [f"- [[{sym}]] ({etype}): {title}"
                  for sym, title, etype in events]
    _write_machine(vault_dir / "Journal" / f"{date_str}.md", "\n".join(lines))
    n += 1

    # Prediction notes (signals only) ----------------------------------------
    for pid, symbol, target, prob, _sig, _rea, rendered in signals:
        body = [f"# {pid}", "",
                f"[[{symbol}]] | target {target} | {prob:.0%} calibrated", "",
                rendered or "(no explanation stored)", "",
                "## Outcome", "_pending resolution_"]
        _write_machine(
            vault_dir / "Predictions" / date_str[:7] / f"{pid}.md",
            "\n".join(body))
        n += 1

    # Company notes for symbols touched today (signals + events) -------------
    touched = sorted({s[1] for s in signals} | {e[0] for e in events})
    for symbol in touched:
        meta = con.execute(
            "SELECT sector, category FROM symbols WHERE symbol = ?",
            [symbol]).fetchone()
        stats = con.execute(
            """
            SELECT count(*),
                   sum(CASE WHEN o.hit THEN 1 ELSE 0 END)
            FROM predictions p JOIN outcomes o ON o.prediction_id = p.id
            WHERE p.symbol = ?
            """, [symbol]).fetchone()
        resolved, hits = (stats[0] or 0), (stats[1] or 0)
        sector = meta[0] if meta else None
        category = meta[1] if meta else None
        body = [f"# {symbol}", "",
                f"sector: {sector} | category: {category}", "",
                f"Prediction scorecard: {hits}/{resolved} resolved hits", ""]
        _write_machine(vault_dir / "Companies" / f"{symbol}.md",
                       "\n".join(body))
        n += 1

    # Home dashboard ----------------------------------------------------------
    total = con.execute("SELECT count(*) FROM predictions").fetchone()[0]
    resolved = con.execute("SELECT count(*) FROM outcomes").fetchone()[0]
    home = ["# Vectora", "",
            f"Latest run: [[Journal/{date_str}|{date_str}]] | "
            f"{len(signals)} signal(s)", "",
            f"Lifetime: {total} predictions, {resolved} resolved", "",
            "_Research tool, not investment advice._"]
    _write_machine(vault_dir / "Home.md", "\n".join(home))
    n += 1
    return {"notes": n, "signals": len(signals)}
