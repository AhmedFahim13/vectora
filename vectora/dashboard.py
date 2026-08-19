"""Static client-facing dashboard generator (Phase 6, viability build).

Renders one self-contained, theme-aware HTML file from the live DuckDB:
regime + KPIs, today's signals (or top-ranked setups when the market is
quiet), the Z-watch, model calibration, and event-impact studies. No
external assets — inline CSS + inline SVG so it works as a local file AND
on a static host (GitHub/Cloudflare Pages).

Everything is framed PAPER / unvalidated: there is no live track record
until outcomes mature (~2026-07-30). This is a research tool, not
investment advice, and the template says so prominently.
"""
import datetime as dt
import html
import json

from vectora.evaluate.report import cohort_stats
from vectora.settings import SIGNAL_THRESHOLDS

_TRACK_RECORD_START = "2026-07-30"
_REGIME_TONE = {
    "Bull": "good", "Recovery": "good", "SpeculativeHeat": "warning",
    "Sideways": "muted", "LowLiquidity": "warning", "Bear": "serious",
    "Panic": "critical",
}


def _esc(x) -> str:
    return html.escape(str(x))


def _bar_chart(rows: list[tuple], threshold: float) -> str:
    """Horizontal probability bars; rows = [(symbol, prob, is_signal)]."""
    if not rows:
        return "<p class='muted'>No predictions for this date.</p>"
    h, gap, left, w = 26, 8, 96, 460
    thr_x = left + threshold * w
    svg = [f"<svg viewBox='0 0 {left + w + 60} {len(rows) * (h + gap) + 24}' "
           "role='img' class='chart'>"]
    for i, (sym, prob, is_sig) in enumerate(rows):
        y = i * (h + gap) + 8
        bw = max(prob * w, 2)
        cls = "bar-sig" if is_sig else "bar-plain"
        svg.append(
            f"<text x='{left - 8}' y='{y + h / 2 + 4}' text-anchor='end' "
            f"class='bar-lbl'>{_esc(sym)}</text>")
        svg.append(
            f"<rect x='{left}' y='{y}' width='{bw:.1f}' height='{h}' rx='4' "
            f"class='{cls}'/>")
        svg.append(
            f"<text x='{left + bw + 6:.1f}' y='{y + h / 2 + 4}' "
            f"class='bar-val'>{prob:.0%}</text>")
    top = 8
    bot = len(rows) * (h + gap)
    svg.append(f"<line x1='{thr_x:.1f}' y1='{top - 4}' x2='{thr_x:.1f}' "
               f"y2='{bot}' class='thr'/>")
    svg.append(f"<text x='{thr_x:.1f}' y='{bot + 16}' text-anchor='middle' "
               f"class='thr-lbl'>signal bar {threshold:.0%}</text>")
    svg.append("</svg>")
    return "".join(svg)


def _kpi(label: str, value: str, tone: str = "muted", sub: str = "") -> str:
    sub_html = f"<div class='kpi-sub'>{_esc(sub)}</div>" if sub else ""
    return (f"<div class='kpi kpi-{tone}'><div class='kpi-val'>{_esc(value)}"
            f"</div><div class='kpi-lbl'>{_esc(label)}</div>{sub_html}</div>")


def build_html(con, date_str: str | None = None) -> str:
    date_str = date_str or str(con.execute(
        "SELECT max(date) FROM predictions").fetchone()[0])
    regime = con.execute(
        "SELECT regime FROM regimes WHERE date = ?", [date_str]).fetchone()
    regime = regime[0] if regime else "unclassified"
    quality = con.execute(
        "SELECT score FROM data_quality WHERE date = ? AND source='dse_eod'",
        [date_str]).fetchone()
    quality = quality[0] if quality else None
    n_pred, n_sig, n_sym = con.execute(
        "SELECT count(*), sum(CASE WHEN is_signal THEN 1 ELSE 0 END), "
        "count(DISTINCT symbol) FROM predictions WHERE date = ?",
        [date_str]).fetchone()
    thr = SIGNAL_THRESHOLDS.get("g5_h10", 0.55)

    model = con.execute(
        "SELECT metrics FROM model_registry WHERE active AND target='g5_h10' "
        "ORDER BY trained_at DESC LIMIT 1").fetchone()
    brier = json.loads(model[0]).get("brier") if model else None

    life_pred = con.execute("SELECT count(*) FROM predictions").fetchone()[0]
    life_res = con.execute("SELECT count(*) FROM outcomes").fetchone()[0]
    # track record: only real once predictions have matured against prices
    hit_rate, cohorts = con.execute(
        """
        SELECT avg(CASE WHEN o.hit THEN 1.0 ELSE 0 END),
               count(DISTINCT p.date)
        FROM outcomes o JOIN predictions p ON p.id = o.prediction_id
        """).fetchone()
    if life_res:
        # the honest interval comes from the spread BETWEEN dates: rows
        # sharing a date share a market, so a row-wise interval is fiction
        pairs = con.execute(
            """
            SELECT p.date, CASE WHEN o.hit THEN 1 ELSE 0 END
            FROM outcomes o JOIN predictions p ON p.id = o.prediction_id
            """).fetchall()
        cs = cohort_stats(pairs)
        mean_p = con.execute(
            """
            SELECT avg(p.probability) FROM outcomes o
            JOIN predictions p ON p.id = o.prediction_id
            """).fetchone()[0] or 0.0
        head = (f"The live track record: <b>{life_res}</b> predictions "
                f"resolved against realized prices across <b>{cohorts}</b> "
                f"maturity date{'s' if (cohorts or 0) != 1 else ''}, "
                f"hit rate <b>{hit_rate:.0%}</b>. ")
        if cs["cohorts"] >= 2:
            lo, hi = cs["ci95"]
            paper_note = (
                head + f"Measured across dates rather than rows, that is "
                f"<b>{lo:.0%} to {hi:.0%}</b> with 95% confidence — the "
                f"row-wise interval would be {cs['se_inflation']:.1f}x too "
                f"narrow, because predictions made on the same day share the "
                f"same market. The model predicted <b>{mean_p:.0%}</b> on "
                f"average against <b>{hit_rate:.0%}</b> realized, so it "
                "remains overconfident.")
        else:
            paper_note = (
                head + "That is still a small, highly correlated sample — "
                "read it as a first reading, not a record.")
    else:
        paper_note = ("The live track record (hit rates, realized "
                      f"calibration) begins <b>{_TRACK_RECORD_START}</b> "
                      "once predictions mature.")

    # --- KPI row ---
    kpis = [
        _kpi("Market regime", regime, _REGIME_TONE.get(regime, "muted")),
        _kpi("Signals today", str(n_sig or 0),
             "good" if (n_sig or 0) else "muted",
             "admitted through all gates"),
        _kpi("Watchlist", str(n_sym or 0), "muted",
             f"{n_pred or 0} scored predictions"),
        _kpi("Data quality", f"{quality}/100" if quality is not None else "n/a",
             "good" if (quality or 0) >= 80 else "warning"),
        _kpi("Model Brier", f"{brier:.3f}" if brier else "n/a", "muted",
             "walk-forward, lower=better"),
    ]

    # --- signals / top setups ---
    top = con.execute(
        """
        SELECT p.symbol, p.probability, p.is_signal,
               r.expected_up, r.expected_down, r.rr_ratio, r.exit_days,
               r.category, e.rendered
        FROM predictions p
        LEFT JOIN risk_blocks r ON r.prediction_id = p.id
        LEFT JOIN explanations e ON e.prediction_id = p.id
        WHERE p.date = ? AND p.target = 'g5_h10'
        ORDER BY p.probability DESC LIMIT 8
        """, [date_str]).fetchall()
    chart = _bar_chart([(r[0], r[1], r[2]) for r in top], thr)

    if n_sig:
        setups_head = f"Signals ({n_sig})"
        setups_note = "Setups that cleared every admission gate today."
    else:
        setups_head = "Top-ranked setups"
        setups_note = ("No setup cleared the signal bar today — a quiet, "
                       "disciplined market read. Highest-ranked candidates "
                       "below, shown for transparency (not signals).")
    def _cell(val, fmt, cls=""):
        c = f" class='{cls}'" if cls else ""
        return f"<td{c}>{fmt.format(val)}</td>" if val is not None \
            else f"<td{c}>&ndash;</td>"

    rows_html = []
    for sym, prob, is_sig, up, dn, rr, exitd, cat, _rend in top:
        tag = " <span class='pill pill-sig'>signal</span>" if is_sig else ""
        rows_html.append(
            f"<tr><td class='sym'>{_esc(sym)}{tag}</td>"
            f"<td class='num'>{prob:.1%}</td>"
            + _cell(up, "{:+.1%}", "num pos")
            + _cell(dn, "{:+.1%}", "num neg")
            + _cell(rr, "{:.1f}", "num")
            + _cell(exitd, "{:.1f}d", "num")
            + f"<td>{_esc(cat or '–')}</td></tr>")
    exit_note = ("(shown as &ndash; until 21 trading days of turnover "
                 "history accumulate &mdash; live collection began "
                 "2026-07-12) ") if any(r[6] is None for r in top) else ""
    setups_table = (
        "<table class='tbl'><thead><tr><th>Symbol</th><th>Prob</th>"
        "<th>Exp. up</th><th>Exp. down</th><th>R/R</th><th>Exit</th>"
        "<th>Cat</th></tr></thead><tbody>" + "".join(rows_html)
        + "</tbody></table>"
        "<div class='legend'><b>Prob</b> = calibrated probability of a +5% "
        "move within 10 trading days &middot; <b>Exp. up/down</b> = median "
        "gain/loss of the 20 most similar historical setups &middot; "
        "<b>R/R</b> = reward-to-risk &middot; <b>Exit</b> = days to unwind a "
        "position without moving the price " + exit_note
        + "&middot; <b>Cat</b> = DSE category (Z names are scored but never "
        "signalled).</div>")

    # top explanation
    expl = next((r[8] for r in top if r[8]), None)
    expl_html = (f"<div class='expl'><h3>Why {_esc(top[0][0])}?</h3>"
                 f"<pre>{_esc(expl)}</pre></div>") if expl else ""

    # --- Z-watch (top pumps + top footprints, balanced so neither buries
    # the other under a single LIMIT) ---
    zw = con.execute(
        """
        SELECT symbol, kind, score, phase FROM (
            SELECT symbol, kind, score, phase,
                   row_number() OVER (PARTITION BY kind ORDER BY score DESC) rn
            FROM zwatch WHERE date = ?
        ) WHERE rn <= 6
        ORDER BY kind = 'pump' DESC, score DESC
        """, [date_str]).fetchall()
    if zw:
        zrows = "".join(
            f"<tr><td class='sym'>{_esc(s)}</td>"
            f"<td>{'pump' if k == 'pump' else 'pre-announce footprint'}</td>"
            f"<td class='num'>{sc:.0f}</td><td>{_esc(ph or '–')}</td></tr>"
            for s, k, sc, ph in zw)
        zwatch_html = (
            "<h2>Z-watch <span class='muted small'>&mdash; unusual public "
            "activity, warnings not signals</span></h2>"
            "<table class='tbl'><thead><tr><th>Symbol</th><th>Kind</th>"
            "<th>Score</th><th>Phase</th></tr></thead><tbody>"
            + zrows + "</tbody></table>")
    else:
        zwatch_html = ""

    # --- event studies (curated) ---
    ev = con.execute(
        """
        SELECT event_type, horizon, n, mean_abn_ret, pos_share
        FROM event_studies
        WHERE horizon IN (5, 10) AND n >= 100
          AND event_type IN ('category_change','dividend_disbursement',
                             'earnings_release','dividend_declared',
                             'business_update','board_meeting')
        ORDER BY event_type, horizon
        """).fetchall()
    if ev:
        erows = "".join(
            f"<tr><td>{_esc(t)}</td><td class='num'>h{h}</td>"
            f"<td class='num'>{n}</td>"
            f"<td class='num {'pos' if m >= 0 else 'neg'}'>{m:+.2%}</td>"
            f"<td class='num'>{p:.0%}</td></tr>"
            for t, h, n, m, p in ev)
        ev_html = (
            "<h2>Event-impact studies <span class='muted small'>&mdash; "
            "market-adjusted return after each announcement type</span></h2>"
            "<table class='tbl'><thead><tr><th>Event</th><th>Horizon</th>"
            "<th>n</th><th>Mean abnormal</th><th>Share &gt;0</th></tr></thead>"
            "<tbody>" + erows + "</tbody></table>"
            "<div class='legend'><b>h</b> = trading days measured after the "
            "announcement &middot; <b>n</b> = how many past announcements of "
            "that type went into the number (bigger = more trustworthy) "
            "&middot; <b>mean abnormal</b> = the stock's return minus the "
            "market's over the same window &middot; <b>share &gt;0</b> = how "
            "often the outcome was positive at all. A high mean with a low "
            "share means a few large winners carry the average.</div>")
    else:
        ev_html = ""

    return _TEMPLATE.format(
        style=STYLE,
        date=_esc(date_str), regime=_esc(regime),
        regime_tone=_REGIME_TONE.get(regime, "muted"),
        paper_note=paper_note,
        generated=dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        kpis="".join(kpis), chart=chart, setups_head=_esc(setups_head),
        setups_note=_esc(setups_note), setups_table=setups_table,
        expl=expl_html, zwatch=zwatch_html, events=ev_html,
        life_pred=life_pred, life_res=life_res)


STYLE = """
:root {
  --bg:#f7f8fa; --surface:#ffffff; --ink:#12151b; --ink2:#4a5568;
  --muted:#8a94a6; --line:#e6e9ef; --accent:#2563c9; --sig:#1f9d6b;
  --good:#1f9d6b; --warning:#c2820a; --serious:#c0563a; --critical:#c0392b;
  --pos:#1f9d6b; --neg:#c0392b;
}
@media (prefers-color-scheme: dark) {
  :root { --bg:#0e1116; --surface:#161b22; --ink:#e6edf3; --ink2:#aab4c2;
    --muted:#7d8794; --line:#242c37; --accent:#5b9cf6; --sig:#3fb984; }
}
:root[data-theme="dark"] { --bg:#0e1116; --surface:#161b22; --ink:#e6edf3;
  --ink2:#aab4c2; --muted:#7d8794; --line:#242c37; --accent:#5b9cf6;
  --sig:#3fb984; }
:root[data-theme="light"] { --bg:#f7f8fa; --surface:#ffffff; --ink:#12151b;
  --ink2:#4a5568; --muted:#8a94a6; --line:#e6e9ef; --accent:#2563c9;
  --sig:#1f9d6b; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
  font:15px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; }
.wrap { max-width:1000px; margin:0 auto; padding:28px 20px 60px; }
header { display:flex; justify-content:space-between; align-items:flex-start;
  flex-wrap:wrap; gap:12px; }
h1 { font-size:24px; margin:0; letter-spacing:-.3px; }
h1 .dot { color:var(--accent); }
.tag { color:var(--ink2); font-size:13px; margin-top:3px; }
.badge { padding:6px 12px; border-radius:999px; font-size:13px;
  font-weight:600; border:1px solid var(--line); background:var(--surface); }
.badge-good { color:var(--good); } .badge-warning { color:var(--warning); }
.badge-serious { color:var(--serious); } .badge-critical { color:var(--critical); }
.badge-muted { color:var(--ink2); }
.paper { margin:18px 0; padding:12px 16px; border-radius:10px;
  border:1px solid var(--warning); background:color-mix(in srgb,var(--warning) 8%,transparent);
  color:var(--ink); font-size:13.5px; }
.paper b { color:var(--warning); }
.kpis { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
  gap:12px; margin:18px 0; }
.kpi { background:var(--surface); border:1px solid var(--line);
  border-radius:12px; padding:14px 16px; }
.kpi-val { font-size:26px; font-weight:700; letter-spacing:-.5px; }
.kpi-lbl { color:var(--ink2); font-size:13px; margin-top:2px; }
.kpi-sub { color:var(--muted); font-size:11.5px; margin-top:4px; }
.kpi-good .kpi-val { color:var(--good); }
.kpi-warning .kpi-val { color:var(--warning); }
.kpi-serious .kpi-val { color:var(--serious); }
.kpi-critical .kpi-val { color:var(--critical); }
section { background:var(--surface); border:1px solid var(--line);
  border-radius:14px; padding:20px 22px; margin:16px 0; }
h2 { font-size:17px; margin:0 0 4px; }
h2 .small { font-size:12.5px; font-weight:400; }
.note { color:var(--ink2); font-size:13.5px; margin:2px 0 14px; }
.legend { color:var(--muted); font-size:12px; line-height:1.65;
  margin-top:12px; padding-top:10px; border-top:1px dashed var(--line); }
.legend b { color:var(--ink2); font-weight:600; }
.chart { width:100%; height:auto; max-width:620px; display:block; margin:8px 0 6px; }
.bar-lbl { fill:var(--ink2); font-size:12px; font-weight:600; }
.bar-val { fill:var(--ink2); font-size:12px; }
.bar-sig { fill:var(--sig); } .bar-plain { fill:var(--accent); opacity:.55; }
.thr { stroke:var(--serious); stroke-width:2; stroke-dasharray:4 3; }
.thr-lbl { fill:var(--serious); font-size:11px; }
.tbl { width:100%; border-collapse:collapse; font-size:13.5px; margin-top:6px; }
.tbl th { text-align:left; color:var(--muted); font-weight:600;
  font-size:12px; text-transform:uppercase; letter-spacing:.4px;
  padding:6px 10px; border-bottom:1px solid var(--line); }
.tbl td { padding:7px 10px; border-bottom:1px solid var(--line); }
.tbl tr:last-child td { border-bottom:none; }
.num { text-align:right; font-variant-numeric:tabular-nums; }
.sym { font-weight:600; }
.pos { color:var(--pos); } .neg { color:var(--neg); }
.pill { font-size:10.5px; font-weight:700; padding:2px 7px; border-radius:999px;
  vertical-align:middle; }
.pill-sig { background:var(--sig); color:#fff; }
.expl { margin-top:16px; border-top:1px solid var(--line); padding-top:14px; }
.expl h3 { font-size:14px; margin:0 0 8px; color:var(--ink2); }
.expl pre { white-space:pre-wrap; font:12.5px/1.55 ui-monospace,Menlo,Consolas,monospace;
  color:var(--ink2); background:var(--bg); border:1px solid var(--line);
  border-radius:8px; padding:12px 14px; margin:0; }
.muted { color:var(--muted); } .small { font-size:12.5px; }
footer { color:var(--muted); font-size:12px; margin-top:26px; line-height:1.7; }
footer b { color:var(--ink2); }
.navlink { color:var(--accent); text-decoration:none; font-size:13px; }
"""

_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vectora — DSE Market Intelligence</title>
<style>
{style}</style></head>
<body>
<div class="wrap">
<header>
  <div>
    <h1>Vectora<span class="dot">.</span></h1>
    <div class="tag">Dhaka Stock Exchange market intelligence &middot; as of {date}</div>
  </div>
  <div style="text-align:right">
    <div class="badge badge-{regime_tone}">Regime: {regime}</div>
    <div style="margin-top:8px"><a class="navlink"
       href="screener.html">Technical screener &rarr;</a></div>
  </div>
</header>

<div class="paper">
  <b>PAPER — unvalidated.</b> Every prediction here is calibrated
  probability with documented uncertainty, generated by a deterministic
  model on public data. {paper_note}
  This is a research tool, <b>not investment advice</b>.
</div>

<div class="kpis">{kpis}</div>

<section>
  <h2>{setups_head}</h2>
  <div class="note">{setups_note}</div>
  {chart}
  {setups_table}
  {expl}
</section>

<section>{zwatch}</section>
<section>{events}</section>

<footer>
  <b>Methodology.</b> Calibrated LightGBM over ~330 liquid DSE equities,
  walk-forward validated with an embargo; signals admitted only above the
  probability bar, in a compatible regime, on non-Z names, at adequate
  liquidity and data quality. Explanations are SHAP feature attributions
  with historical analogs.<br>
  <b>Lifetime.</b> {life_pred} predictions generated, {life_res} resolved
  against realized prices.<br>
  Generated {generated}. Zero-cost pipeline &middot; research use only &middot;
  not investment advice.
</footer>
</div>
</body></html>"""
