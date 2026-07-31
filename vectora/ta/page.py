"""Client-facing technical screener page (spec: Phase 6B).

Shows a 5-band posture for every symbol with a drop-down of exactly which
indicators voted which way — and, unlike commercial screeners, prints the
MEASURED historical hit rate of each band beside the market base rate so a
reader can see whether the label has ever meant anything on this exchange.
"""
import datetime as dt
import html
from pathlib import Path

from vectora.dashboard import STYLE
from vectora.settings import REPO_ROOT
from vectora.ta.screener import load_watchlist, ranked

DEFAULT_OUT = REPO_ROOT / "docs" / "dashboard" / "screener.html"
_TONE = {"Strong Buy": "good", "Buy": "sig", "Hold": "muted",
         "Sell": "warning", "Strong Sell": "critical"}
_EXTRA_CSS = """
.pill-band { font-size:11px; font-weight:700; padding:3px 9px;
  border-radius:999px; color:#fff; white-space:nowrap; }
.pb-good { background:var(--good); } .pb-sig { background:var(--sig); }
.pb-muted { background:var(--muted); } .pb-warning { background:var(--warning); }
.pb-critical { background:var(--critical); }
details { margin:4px 0 0; }
details summary { cursor:pointer; color:var(--accent); font-size:12px;
  list-style:none; }
details summary::-webkit-details-marker { display:none; }
details summary:before { content:"\\25B8 why"; }
details[open] summary:before { content:"\\25BE why"; }
details ul { margin:6px 0 10px; padding-left:18px; color:var(--ink2);
  font-size:12.5px; line-height:1.6; }
.vote-up { color:var(--good); font-weight:600; }
.vote-dn { color:var(--critical); font-weight:600; }
.vote-0 { color:var(--muted); }
.grp { margin-top:22px; }
.grp h3 { font-size:14px; margin:0 0 6px; color:var(--ink2);
  text-transform:uppercase; letter-spacing:.5px; }
.evidence td.edge-pos { color:var(--good); font-weight:600; }
.evidence td.edge-neg { color:var(--critical); font-weight:600; }
.navlink { color:var(--accent); text-decoration:none; font-size:13px; }
"""


def _esc(x) -> str:
    return html.escape(str(x))


def _band_pill(band: str) -> str:
    return (f"<span class='pill-band pb-{_TONE.get(band, 'muted')}'>"
            f"{_esc(band)}</span>")


def _votes_details(votes: list[dict]) -> str:
    items = []
    for v in votes:
        cls = "vote-up" if v["vote"] > 0 else (
            "vote-dn" if v["vote"] < 0 else "vote-0")
        sign = f"{v['vote']:+d}" if v["vote"] else "0"
        items.append(f"<li><span class='{cls}'>{_esc(v['indicator'])} "
                     f"{sign}</span> — {_esc(v['reason'])}</li>")
    return ("<details><summary></summary><ul>" + "".join(items)
            + "</ul></details>")


def _rows(entries: list[dict]) -> str:
    out = []
    for e in entries:
        rsi = f"{e['rsi']:.0f}" if e.get("rsi") is not None else "&ndash;"
        trend = "up" if (e.get("st_dir") or 0) > 0 else (
            "down" if (e.get("st_dir") or 0) < 0 else "&ndash;")
        out.append(
            f"<tr><td class='sym'>{_esc(e['symbol'])}"
            f"{_votes_details(e['votes'])}</td>"
            f"<td>{_band_pill(e['band'])}</td>"
            f"<td class='num'>{e['score']:+d}</td>"
            f"<td class='num'>{rsi}</td>"
            f"<td>{trend}</td>"
            f"<td>{_esc(e.get('category') or '?')}</td></tr>")
    return "".join(out)


def _table(entries: list[dict]) -> str:
    if not entries:
        return "<p class='muted'>No ratings for these symbols.</p>"
    return ("<table class='tbl'><thead><tr><th>Symbol</th><th>Posture</th>"
            "<th>Score</th><th>RSI</th><th>Trend</th><th>Cat</th></tr></thead>"
            "<tbody>" + _rows(entries) + "</tbody></table>")


def _evidence(con, horizon: int = 10) -> str:
    rows = con.execute(
        "SELECT band, n, hit_rate, base_rate, mean_fwd FROM ta_band_stats "
        "WHERE horizon = ? ORDER BY hit_rate DESC", [horizon]).fetchall()
    if not rows:
        return ""
    body = "".join(
        f"<tr><td>{_esc(b)}</td><td class='num'>{n:,}</td>"
        f"<td class='num'>{hit:.1%}</td><td class='num'>{base:.1%}</td>"
        f"<td class='num {'edge-pos' if edge > 0 else 'edge-neg'}'>"
        f"{edge * 100:+.1f} pp</td></tr>"
        for b, n, hit, base, edge in rows)
    return (
        "<section><h2>Does the posture mean anything? "
        "<span class='muted small'>&mdash; measured, not asserted</span></h2>"
        f"<div class='note'>Every rating in this table was replayed across "
        f"13 years of DSE history and scored against what actually happened "
        f"next: did the stock gain 5% within {horizon} trading days? "
        "&ldquo;Edge&rdquo; is the band&rsquo;s hit rate minus the market&rsquo;s "
        "own base rate over the same windows. A band is only worth attention "
        "if that number is positive.</div>"
        "<table class='tbl evidence'><thead><tr><th>Posture</th><th>n</th>"
        "<th>Hit rate</th><th>Base rate</th><th>Edge</th></tr></thead><tbody>"
        + body + "</tbody></table>"
        "<div class='legend'>Read with care: this measures whether price "
        "<i>touched</i> +5% at any point in the window, which is not the same "
        "as a profitable round trip — it ignores costs, slippage and exit "
        "timing. It also assumes every one of these instances was tradable at "
        "the shown liquidity.</div></section>")


def build(con, date_str: str | None = None,
          out_path: Path = DEFAULT_OUT, board_n: int = 20) -> Path:
    date_str = date_str or str(con.execute(
        "SELECT max(date) FROM ta_ratings").fetchone()[0])
    groups = load_watchlist()
    sections = []

    all_rows = ranked(con, date_str, limit=10000)
    if not all_rows:
        body = "<section><p>No technical ratings for this date.</p></section>"
    else:
        counts: dict = {}
        for r in all_rows:
            counts[r["band"]] = counts.get(r["band"], 0) + 1
        kpi = "".join(
            f"<div class='kpi kpi-{_TONE.get(b, 'muted')}'>"
            f"<div class='kpi-val'>{counts.get(b, 0)}</div>"
            f"<div class='kpi-lbl'>{_esc(b)}</div></div>"
            for b in ("Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"))
        sections.append(f"<div class='kpis'>{kpi}</div>")
        sections.append(_evidence(con))

        wl = []
        for name, syms in groups.items():
            entries = ranked(con, date_str, symbols=list(syms), limit=200)
            missing = sorted(set(syms) - {e["symbol"] for e in entries})
            note = (f"<div class='legend'>Not rated today: "
                    f"{', '.join(missing)}</div>" if missing else "")
            wl.append(f"<div class='grp'><h3>{_esc(name)}</h3>"
                      + _table(entries) + note + "</div>")
        sections.append(
            "<section><h2>Your watchlist</h2><div class='note'>Ranked within "
            "each group by technical score. Open <i>why</i> on any row to see "
            "exactly which indicators voted and in which direction.</div>"
            + "".join(wl) + "</section>")

        top = all_rows[:board_n]
        bottom = ranked(con, date_str, limit=board_n, ascending=True)
        sections.append(
            "<section><h2>Quick scan</h2>"
            f"<div class='grp'><h3>Strongest {board_n}</h3>" + _table(top)
            + "</div>"
            f"<div class='grp'><h3>Weakest {board_n}</h3>" + _table(bottom)
            + "</div></section>")

        # every rated symbol on the exchange, grouped by sector
        by_sector: dict = {}
        for r in all_rows:
            by_sector.setdefault(r.get("sector") or "Unclassified", []).append(r)
        blocks = []
        for sector in sorted(by_sector):
            rows = sorted(by_sector[sector], key=lambda x: -x["score"])
            blocks.append(f"<div class='grp'><h3>{_esc(sector)} "
                          f"<span class='muted'>({len(rows)})</span></h3>"
                          + _table(rows) + "</div>")
        sections.append(
            f"<section><h2>Every listed security &mdash; {len(all_rows)} "
            "rated today</h2><div class='note'>Complete coverage of the "
            "exchange, grouped by sector and ranked by technical score within "
            "each. Every row opens its own rationale.</div>"
            + "".join(blocks) + "</section>")

        body = "".join(sections)

    page = _TEMPLATE.format(
        style=STYLE + _EXTRA_CSS, date=_esc(date_str), body=body,
        generated=dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        n=len(all_rows))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(page, encoding="utf-8")
    return Path(out_path)


_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vectora — Technical Screener</title>
<style>
{style}</style></head>
<body>
<div class="wrap">
<header>
  <div>
    <h1>Vectora<span class="dot">.</span></h1>
    <div class="tag">Technical screener &middot; {n} symbols &middot; as of {date}</div>
  </div>
  <a class="navlink" href="index.html">&larr; back to daily brief</a>
</header>

<div class="paper">
  <b>A technical posture is a mechanical summary of six indicators</b>
  (MACD, RSI, Bollinger, moving-average cross, SuperTrend, candlestick
  patterns) — not a forecast, and <b>not investment advice</b>. Each band's
  measured historical edge is printed below so you can judge it on evidence.
  Vectora's separately-validated model probability lives on the
  <a class="navlink" href="index.html">daily brief</a>.
</div>

{body}

<footer>
  Ratings recomputed from end-of-day prices each trading day.
  Generated {generated}. Zero-cost pipeline &middot; research use only &middot;
  not investment advice.
</footer>
</div>
</body></html>
"""
