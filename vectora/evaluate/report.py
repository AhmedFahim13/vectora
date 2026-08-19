"""Calibration accounting + error autopsy (spec §17.2-17.3).

Grades every resolved prediction: Brier, hit rate, reliability bins per
target, segmented by regime and category. Every miss gets a rule-based
cause tag (first match wins): event-shock (materiality-3 event inside the
horizon window), liquidity (exit_days > LIQ_DAYS), regime-shift (regime
at prediction date differs from any regime inside the window), else
model-error. Tags accumulate in outcome_tags for Phase 6 pattern notes.
"""
import datetime as dt
import re
from pathlib import Path

from vectora import db as vdb
from vectora.settings import REPORTS_DIR, VAULT_DIR
from vectora.train.models import brier as brier_score
from vectora.train.models import reliability_table
from vectora.vault.generator import _write_machine

LIQ_DAYS = 5.0
Z95 = 1.96
_H_RE = re.compile(r"_h(\d+)$")



def cohort_stats(rows: list[tuple]) -> dict:
    """Evidence accounting in COHORTS, not rows.

    Every prediction made on the same date faces the same market that day,
    so the rows inside a cohort are strongly correlated. Treating them as
    independent is what lets a track record of nine trading days present
    itself as three thousand observations, and it shrinks the confidence
    interval by roughly the square root of the cohort size.

    The honest unit is the cohort: take each date's hit rate as one
    observation and put the interval around those. `se_inflation` reports
    how much wider that is than the naive row-wise interval — if it is
    well above 1, the row-wise number was fiction.

    rows: (prediction_date, hit) pairs.
    """
    if not rows:
        return {"cohorts": 0, "n": 0, "cohort_mean": None, "cohort_se": None,
                "naive_se": None, "se_inflation": None, "ci95": None,
                "per_cohort": []}
    by_date: dict = {}
    for d, hit in rows:
        agg = by_date.setdefault(str(d), [0, 0])
        agg[0] += int(hit)
        agg[1] += 1
    per = [{"date": d, "hits": h, "n": n, "hit_rate": h / n}
           for d, (h, n) in sorted(by_date.items())]
    n_total = sum(c["n"] for c in per)
    hits_total = sum(c["hits"] for c in per)
    pooled = hits_total / n_total
    naive_se = (pooled * (1 - pooled) / n_total) ** 0.5

    k = len(per)
    cohort_mean = sum(c["hit_rate"] for c in per) / k
    out = {"cohorts": k, "n": n_total, "pooled_hit_rate": pooled,
           "cohort_mean": cohort_mean, "naive_se": naive_se,
           "per_cohort": per, "cohort_se": None, "se_inflation": None,
           "ci95": None}
    if k < 2:
        # one date is one observation, however many rows it carries: there
        # is no spread to estimate and no honest interval to quote
        return out
    var = sum((c["hit_rate"] - cohort_mean) ** 2 for c in per) / (k - 1)
    se = var ** 0.5 / k ** 0.5
    out["cohort_se"] = se
    out["se_inflation"] = se / naive_se if naive_se else 0.0
    out["ci95"] = (cohort_mean - Z95 * se, cohort_mean + Z95 * se)
    return out



def _cohort_section(c: dict) -> list[str]:
    """How much evidence there actually is, stated in cohorts."""
    if not c or c["cohorts"] == 0:
        return []
    lines = ["### Independent evidence", "",
             f"{c['n']} resolved rows, but only **{c['cohorts']} independent "
             f"prediction date(s)**. Rows sharing a date share a market, so "
             f"the cohort is the honest unit of evidence."]
    if c["cohorts"] < 2:
        lines += ["", "A single date carries no spread, so no confidence "
                  "interval can be quoted. Any hit rate here is one "
                  "observation wearing many hats.", ""]
    else:
        lo, hi = c["ci95"]
        lines += ["",
                  f"- Mean of per-date hit rates: **{c['cohort_mean']:.1%}**",
                  f"- 95% CI across dates: **{lo:.1%} to {hi:.1%}** "
                  f"(+/-{Z95 * c['cohort_se']:.1%})",
                  f"- Row-wise CI would be +/-{Z95 * c['naive_se']:.1%}, "
                  f"**{c['se_inflation']:.1f}x too narrow**", ""]
    lines += ["| date | n | hit rate |", "|---|---|---|"]
    lines += [f"| {x['date']} | {x['n']} | {x['hit_rate']:.1%} |"
              for x in c["per_cohort"]]
    lines.append("")
    return lines


def evaluate(con, reports_dir: Path = REPORTS_DIR,
             vault_dir: Path = VAULT_DIR, seg_min: int = 5) -> dict:
    rows = con.execute(
        """
        SELECT p.id, p.symbol, p.date, p.target, p.probability,
               o.hit, r.exit_days,
               coalesce(g.regime, 'unclassified') AS regime,
               coalesce(s.category, '?') AS category
        FROM predictions p
        JOIN outcomes o ON o.prediction_id = p.id
        LEFT JOIN risk_blocks r ON r.prediction_id = p.id
        LEFT JOIN regimes g ON g.date = p.date
        LEFT JOIN symbols s ON s.symbol = p.symbol
        """).fetchall()
    if not rows:
        return {"resolved": 0}

    targets: dict = {}
    seg_lines = []
    for tgt in sorted({r[3] for r in rows}):
        sub = [r for r in rows if r[3] == tgt]
        ys = [int(r[5]) for r in sub]
        ps = [float(r[4]) for r in sub]
        targets[tgt] = {
            "n": len(sub), "hit_rate": sum(ys) / len(ys),
            "brier": brier_score(ys, ps),
            "reliability": reliability_table(ys, ps),
            "cohorts": cohort_stats([(r[2], r[5]) for r in sub]),
        }
        for seg_idx, seg_name in ((7, "regime"), (8, "category")):
            for val in sorted({r[seg_idx] for r in sub}):
                seg = [r for r in sub if r[seg_idx] == val]
                if len(seg) < seg_min:
                    continue
                hr = sum(int(r[5]) for r in seg) / len(seg)
                seg_lines.append(
                    f"| {tgt} | {seg_name}={val} | {len(seg)} | {hr:.0%} |")

    tags = []
    for pid, symbol, d, tgt, _p, hit, exit_days, regime, _cat in rows:
        if hit:
            continue
        h = int(_H_RE.search(tgt).group(1)) if _H_RE.search(tgt) else 10
        end = (d + dt.timedelta(days=int(h * 1.6))).isoformat()
        ev = con.execute(
            """
            SELECT 1 FROM events e JOIN event_labels l ON l.event_id = e.id
            WHERE e.symbol = ? AND l.materiality >= 3
              AND e.post_date > ? AND e.post_date <= ? LIMIT 1
            """, [symbol, str(d), end]).fetchone()
        if ev:
            tag = "event-shock"
        elif exit_days is not None and exit_days > LIQ_DAYS:
            tag = "liquidity"
        else:
            shift = con.execute(
                "SELECT 1 FROM regimes WHERE date > ? AND date <= ? "
                "AND regime <> ? LIMIT 1", [str(d), end, regime]).fetchone()
            tag = "regime-shift" if shift else "model-error"
        tags.append({"prediction_id": pid, "tag": tag})
    if tags:
        vdb.upsert(con, "outcome_tags", tags)

    today = dt.date.today().isoformat()
    lines = [f"# Evaluation {today}", "",
             f"{len(rows)} resolved predictions", ""]
    for tgt, m in targets.items():
        lines += [f"## {tgt}", "",
                  f"n={m['n']} | hit rate {m['hit_rate']:.0%} | "
                  f"Brier {m['brier']:.4f}", "",
                  "| bin | n | predicted | realized |", "|---|---|---|---|"]
        lines += [f"| {b['bin_lo']:.1f}-{b['bin_hi']:.1f} | {b['n']} "
                  f"| {b['p_mean']:.3f} | {b['y_rate']:.3f} |"
                  for b in m["reliability"]]
        lines.append("")
        lines += _cohort_section(m["cohorts"])
    if seg_lines:
        lines += [f"## Segments (n>={seg_min})", "",
                  "| target | segment | n | hit rate |", "|---|---|---|---|"]
        lines += seg_lines
    tag_counts = con.execute(
        "SELECT tag, count(*) FROM outcome_tags GROUP BY 1 ORDER BY 2 DESC"
    ).fetchall()
    if tag_counts:
        lines += ["", "## Miss autopsy", ""]
        lines += [f"- {t}: {n}" for t, n in tag_counts]
    body = "\n".join(lines) + "\n"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / f"eval_{today}.md").write_text(body, encoding="utf-8")
    _write_machine(Path(vault_dir) / "Evaluations" / f"{today}.md", body)

    return {"resolved": len(rows), "targets": targets,
            "misses_tagged": len(tags)}
