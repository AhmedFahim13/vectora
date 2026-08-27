"""Client-facing technical screener page (spec: Phase 6B).

Shows a 5-band posture for every symbol with a drop-down of exactly which
indicators voted which way — and, unlike commercial screeners, prints the
MEASURED historical hit rate of each band beside the market base rate so a
reader can see whether the label has ever meant anything on this exchange.
"""
import datetime as dt
import html
from pathlib import Path

from vectora import sectors
from vectora.collect import dse_company
from vectora.dashboard import STYLE
from vectora.settings import REPO_ROOT
from vectora.ta import overlay
from vectora.ta.screener import gauges_for, levels_for, load_watchlist, ranked

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
.dhead { font-size:11px; font-weight:700; letter-spacing:.6px; margin:10px 0 0;
  color:var(--ink2); text-transform:uppercase; }
.scroll { overflow-x:auto; }
/* 399 rows x 32 collapsed rationales is a lot of DOM; let the browser skip
   layout for sector blocks that are off-screen */
.grp { content-visibility:auto; contain-intrinsic-size:auto 700px; }
.scroll table { min-width:900px; }
.pill-screen { font-size:10px; font-weight:700; padding:2px 7px; margin:0 3px 0 0;
  border-radius:999px; color:#fff; white-space:nowrap; display:inline-block; }
.ps-good { background:var(--good); } .ps-sig { background:var(--sig); }
.ps-warning { background:var(--warning); } .ps-critical { background:var(--critical); }
.ps-muted { background:var(--muted); }
"""


def _esc(x) -> str:
    return html.escape(str(x))


def _band_pill(band: str) -> str:
    return (f"<span class='pill-band pb-{_TONE.get(band, 'muted')}'>"
            f"{_esc(band)}</span>")


def _vote_list(votes: list[dict]) -> str:
    items = []
    for v in votes:
        cls = "vote-up" if v["vote"] > 0 else (
            "vote-dn" if v["vote"] < 0 else "vote-0")
        sign = f"{v['vote']:+d}" if v["vote"] else "0"
        items.append(f"<li><span class='{cls}'>{_esc(v['indicator'])} "
                     f"{sign}</span> — {_esc(v['reason'])}</li>")
    return "<ul>" + "".join(items) + "</ul>"


def _tally(g: dict, prefix: str) -> str:
    return (f"<span class='vote-up'>{g[prefix + '_buy']} buy</span> · "
            f"<span class='vote-0'>{g[prefix + '_neutral']} neutral</span> · "
            f"<span class='vote-dn'>{g[prefix + '_sell']} sell</span>")


def _votes_details(votes: list[dict], g: dict | None = None,
                   fund: dict | None = None, lv: dict | None = None,
                   entry: dict | None = None) -> str:
    """The drop-down rationale: six-family votes always, and when the full
    26-component gauges are available, each one named with its own reason,
    followed by the price levels and fundamental screens for that stock."""
    parts = ["<div class='dhead'>Six-family posture</div>", _vote_list(votes)]
    if g:
        parts += [
            f"<div class='dhead'>Moving averages &mdash; {_esc(g['ma_band'])} "
            f"<span class='muted'>({_tally(g, 'ma')})</span></div>",
            _vote_list(g["votes"]["ma"]),
            f"<div class='dhead'>Oscillators &mdash; {_esc(g['osc_band'])} "
            f"<span class='muted'>({_tally(g, 'osc')})</span></div>",
            _vote_list(g["votes"]["osc"])]
    if lv is not None:
        parts += ["<div class='dhead'>Levels</div>",
                  _levels_list(lv, (fund or {}).get("close"))]
    if entry is not None:
        parts += ["<div class='dhead'>Fundamental screens</div>",
                  _screens_list(entry)]
    if fund:
        parts += ["<div class='dhead'>Fundamentals</div>", _fund_list(fund)]
    return "<details><summary></summary>" + "".join(parts) + "</details>"


def _fund_list(f: dict) -> str:
    def fmt(key, label, suffix="", scale=1.0, dp=2):
        v = f.get(key)
        if v is None:
            return ""
        return (f"<li><b>{label}:</b> {v * scale:,.{dp}f}{suffix}</li>")
    # cash and bonus are shown separately on purpose: a bonus issue changes
    # the share count, not the holder's income
    yr = f" for {f['dividend_year']}" if f.get("dividend_year") else ""
    div = ""
    if f.get("latest_dividend_pct") is not None:
        div += (f"<li><b>Cash dividend:</b> {f['latest_dividend_pct']:,.2f}% of "
                f"face{yr}"
                + (f" (৳{f['dividend_per_share']:,.2f}/share)"
                   if f.get("dividend_per_share") else "") + "</li>")
    if f.get("latest_bonus_pct") is not None:
        div += (f"<li><b>Bonus issue:</b> {f['latest_bonus_pct']:,.2f}%"
                f"{yr} <span class='muted'>&mdash; stock, not income; "
                "excluded from the yield</span></li>")
    return ("<ul>"
            + fmt("trailing_pe", "Trailing P/E")
            + fmt("eps_trailing", "Trailing EPS", " ৳")
            + fmt("dividend_yield", "Dividend yield", "%", 100.0)
            + div
            + fmt("market_cap_mn", "Market cap", " mn ৳", 1.0, 0)
            + fmt("free_float_mcap_mn", "Free-float cap", " mn ৳", 1.0, 0)
            + fmt("reserve_surplus_mn", "Reserve & surplus", " mn ৳", 1.0, 0)
            + (f"<li><b>Listed:</b> {f['listing_year']}</li>"
               if f.get("listing_year") else "")
            + (f"<li><b>Year end:</b> {_esc(f['year_end'])}</li>"
               if f.get("year_end") else "")
            + "</ul>")


def _num_or_dash(v, fmt: str = "{:.0f}") -> str:
    return fmt.format(v) if v is not None else "&ndash;"


def _rows(entries: list[dict], gmap: dict, fmap: dict, lmap: dict,
          full: bool = False) -> str:
    out = []
    for e in entries:
        g = gmap.get(e["symbol"])
        f = fmap.get(e["symbol"])
        trend = "up" if (e.get("st_dir") or 0) > 0 else (
            "down" if (e.get("st_dir") or 0) < 0 else "&ndash;")
        summary = _band_pill(g["summary_band"]) if g else "&ndash;"
        ma = _band_pill(g["ma_band"]) if g else "&ndash;"
        osc = _band_pill(g["osc_band"]) if g else "&ndash;"
        pe = _num_or_dash((f or {}).get("trailing_pe"), "{:.1f}")
        yld = ("{:.1f}%".format((f or {}).get("dividend_yield") * 100)
               if (f or {}).get("dividend_yield") is not None else "&ndash;")
        cap = _num_or_dash((f or {}).get("market_cap_mn"), "{:,.0f}")
        detail = _votes_details(
            e["votes"], g if full else None, f if full else None,
            lmap.get(e["symbol"]) if full else None, e if full else None)
        out.append(
            f"<tr><td class='sym'>{_esc(e['symbol'])}{detail}"
            f"{_screen_badges(e) if not full else ''}</td>"
            f"<td>{summary}</td><td>{ma}</td><td>{osc}</td>"
            f"<td>{_band_pill(e['band'])}</td>"
            f"<td class='num'>{e['score']:+d}</td>"
            f"<td class='num'>{_num_or_dash(e.get('rsi'))}</td>"
            f"<td>{trend}</td>"
            f"<td class='num'>{pe}</td><td class='num'>{yld}</td>"
            f"<td class='num'>{cap}</td>"
            f"<td>{_esc(e.get('category') or '?')}</td></tr>")
    return "".join(out)


def _table(entries: list[dict], gmap: dict | None = None,
           fmap: dict | None = None, lmap: dict | None = None,
           full: bool = False) -> str:
    if not entries:
        return "<p class='muted'>No ratings for these symbols.</p>"
    return ("<div class='scroll'><table class='tbl'><thead><tr><th>Symbol</th>"
            "<th>Summary</th><th>MAs</th><th>Oscillators</th>"
            "<th>Six-family</th><th>Score</th><th>RSI</th><th>Trend</th>"
            "<th>P/E</th><th>Yield</th><th>Mkt cap (mn)</th><th>Cat</th>"
            "</tr></thead><tbody>"
            + _rows(entries, gmap or {}, fmap or {}, lmap or {}, full)
            + "</tbody></table></div>")


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


def _fundamentals(con, date_str: str) -> dict:
    """Latest fundamentals per symbol, with the price-dependent metrics
    (yield, trailing EPS) derived against the close on the rating date."""
    rows = con.execute("""
        WITH latest AS (
            SELECT symbol, max(as_of) AS as_of FROM fundamentals GROUP BY 1)
        SELECT f.symbol, f.market_cap_mn, f.free_float_mcap_mn,
               f.reserve_surplus_mn, f.trailing_pe, f.latest_dividend_pct,
               f.latest_bonus_pct, f.dividend_year, f.face_value,
               f.listing_year, f.year_end,
               (SELECT c.paid_up_capital_mn FROM company_snapshot c
                 WHERE c.symbol = f.symbol
                 ORDER BY c.as_of DESC LIMIT 1) AS paid_up_capital_mn,
               (SELECT p.close FROM prices p
                 WHERE p.symbol = f.symbol AND p.date <= ?
                 ORDER BY p.date DESC LIMIT 1) AS close
        FROM fundamentals f JOIN latest l
          ON l.symbol = f.symbol AND l.as_of = f.as_of
    """, [date_str]).fetchall()
    cols = ("symbol", "market_cap_mn", "free_float_mcap_mn",
            "reserve_surplus_mn", "trailing_pe", "latest_dividend_pct",
            "latest_bonus_pct", "dividend_year", "face_value", "listing_year",
            "year_end", "paid_up_capital_mn", "close")
    out = {}
    for row in rows:
        d = dict(zip(cols, row, strict=True))
        close = d.pop("close")
        d.update(dse_company.derive_metrics(d, close))
        d["close"] = close
        out[d["symbol"]] = d
    return out


def _gauge_evidence(con, horizon: int = 10) -> str:
    rows = con.execute(
        "SELECT gauge, band, n, hit_rate, base_rate FROM ta_gauge_stats "
        "WHERE horizon = ? ORDER BY gauge, hit_rate DESC", [horizon]).fetchall()
    if not rows:
        return ""
    # counts and edges read from the data, never written into the prose: the
    # component list was pruned on 2026-08-26 and hardcoded copy would have
    # gone on claiming 26 of them
    from vectora.ta import gauges as _g
    probe = {"close": 100.0}
    n_ma = len(_g.ma_votes(probe))
    n_osc = len(_g.osc_votes(probe))
    # sum across one gauge's bands: every symbol-day lands in exactly one
    # band, so max() would report the largest band rather than the corpus
    scored = sum(n for g_, _b, n, _h, _base in rows if g_ == "summary")
    def _edge(gauge, band):
        for g_, b, _n, h, base in rows:
            if g_ == gauge and b == band:
                return (h - base) * 100
        return None
    sum_sb, six_sb = _edge("summary", "Strong Buy"), None
    six = con.execute(
        "SELECT hit_rate, base_rate FROM ta_band_stats "
        "WHERE band = 'Strong Buy' AND horizon = ?", [horizon]).fetchone()
    if six:
        six_sb = (six[0] - six[1]) * 100
    labels = {"summary": "Summary (all 26)", "moving_averages": "Moving averages (15)",
              "oscillators": "Oscillators (11)"}
    body = "".join(
        f"<tr><td>{_esc(labels.get(g, g))}</td><td>{_esc(b)}</td>"
        f"<td class='num'>{n:,}</td><td class='num'>{hit:.1%}</td>"
        f"<td class='num'>{base:.1%}</td>"
        f"<td class='num {'edge-pos' if hit > base else 'edge-neg'}'>"
        f"{(hit - base) * 100:+.1f} pp</td></tr>"
        for g, b, n, hit, base in rows)
    return (
        f"<section><h2>Do the {n_ma + n_osc} indicators beat the six? "
        f"<span class='muted small'>&mdash; measured on {scored:,} "
        "symbol-days</span></h2>"
        f"<div class='note'>The same replay, applied to the "
        f"{n_ma}-average and {n_osc}-oscillator set the client selected. "
        + (f"The summary gauge&rsquo;s Strong Buy carries a larger edge than "
           f"the six-family Strong Buy ({sum_sb:+.1f}pp vs {six_sb:+.1f}pp), "
           "so the second read earns its place. "
           if (sum_sb is not None and six_sb is not None) else "")
        + "Two results deserve scepticism rather "
        "than excitement: the oscillator gauge beats the base rate at "
        "<i>both</i> extremes, and its Strong Sell (+12.6pp) reads better than "
        "its Buy. That is not directional skill — a stock whose oscillators "
        "are screaming in either direction is simply a volatile stock, and "
        "this label only asks whether price <i>touched</i> +5%. Read the "
        "oscillator gauge as a volatility flag; read the moving-average gauge "
        "for direction.</div>"
        "<div class='scroll'><table class='tbl evidence'><thead><tr>"
        "<th>Gauge</th><th>Posture</th><th>n</th><th>Hit rate</th>"
        "<th>Base rate</th><th>Edge</th></tr></thead><tbody>"
        + body + "</tbody></table></div></section>")


_QUAD_TONE = {"Leading": "good", "Improving": "sig",
              "Weakening": "warning", "Lagging": "critical"}
_SCREEN_TONE = {"Value": "good", "Income": "sig", "Quality": "good",
                "Impaired": "critical", "Thin float": "warning"}


def _screen_badges(entry: dict) -> str:
    return "".join(
        f"<span class='pill-screen ps-{_SCREEN_TONE.get(s['screen'], 'muted')}'>"
        f"{_esc(s['screen'])}</span>" for s in entry.get("screens", []))


def _levels_list(lv: dict, close: float | None) -> str:
    if not lv or lv.get("pivot_point") is None:
        return ("<ul><li class='muted'>No completed prior month yet, so no "
                "pivot levels.</li></ul>")

    def row(label: str, key: str) -> str:
        v = lv.get(key)
        if v is None:
            return ""
        gap = (f" <span class='muted'>({(v / close - 1) * 100:+.1f}%)</span>"
               if close else "")
        return f"<li><b>{label}:</b> {v:,.2f}{gap}</li>"

    room = ""
    if lv.get("room_up") is not None:
        room += (f"<li><b>Room to resistance:</b> {lv['room_up'] * 100:.1f}% "
                 f"(at {lv['nearest_res']:,.2f})</li>")
    if lv.get("room_dn") is not None:
        room += (f"<li><b>Room to support:</b> {lv['room_dn'] * 100:.1f}% "
                 f"(at {lv['nearest_sup']:,.2f})</li>")
    return ("<ul>" + room + row("R2", "r2") + row("R1", "r1")
            + row("Pivot", "pivot_point") + row("S1", "s1") + row("S2", "s2")
            + row("60-day high", "hi_60d") + row("60-day low", "lo_60d")
            + row("52-week high", "hi_252d") + row("52-week low", "lo_252d")
            + "</ul>")


def _screens_list(entry: dict) -> str:
    if not entry.get("screens"):
        return "<ul><li class='muted'>No fundamental screen matched.</li></ul>"
    return ("<ul>" + "".join(
        f"<li><b>{_esc(s['screen'])}:</b> {_esc(s['detail'])}</li>"
        for s in entry["screens"]) + "</ul>")


def _rotation(con, date_str: str) -> str:
    rows = [r for r in sectors.load(con, date_str)
            if r["ret_21d"] is not None]
    if not rows:
        return ""
    body = "".join(
        f"<tr><td>{_esc(r['sector'])}</td>"
        f"<td><span class='pill-band pb-{_QUAD_TONE.get(r['quadrant'], 'muted')}'>"
        f"{_esc(r['quadrant'])}</span></td>"
        f"<td class='num'>{r['n_symbols']}</td>"
        f"<td class='num'>{r['ret_21d'] * 100:+.1f}%</td>"
        f"<td class='num {'edge-pos' if r['rs_21d'] > 0 else 'edge-neg'}'>"
        f"{r['rs_21d'] * 100:+.1f} pp</td>"
        f"<td class='num {'edge-pos' if (r['rs_momentum'] or 0) > 0 else 'edge-neg'}'>"
        f"{(r['rs_momentum'] or 0) * 100:+.1f} pp</td>"
        f"<td class='num'>{(r['ret_63d'] or 0) * 100:+.1f}%</td></tr>"
        for r in rows)
    return (
        "<section><h2>Sector rotation <span class='muted small'>"
        "&mdash; relative strength, not raw return</span></h2>"
        "<div class='note'>In a rising market almost every sector goes up, so "
        "raw return says little. <b>Relative strength</b> is the sector&rsquo;s "
        "21-day return minus the market&rsquo;s over the same window, and "
        "<b>momentum</b> is whether that lead is still growing. A sector ahead "
        "but decelerating (<i>Weakening</i>) and one behind but accelerating "
        "(<i>Improving</i>) are usually the two that matter. Sectors are "
        "equal-weighted so a few giants cannot stand in for the whole group."
        "</div>"
        "<div class='scroll'><table class='tbl'><thead><tr><th>Sector</th>"
        "<th>Phase</th><th>n</th><th>21d return</th><th>vs market</th>"
        "<th>Momentum</th><th>63d return</th></tr></thead><tbody>"
        + body + "</tbody></table></div>"
        "<div class='legend'>Relative strength across sectors does not sum to "
        "zero: compounding is convex, so compounded sector returns average "
        "slightly above the compounded market return. The residual is small "
        "next to the spread being measured.</div></section>")


def _confluence(entries: list[dict], gmap: dict) -> str:
    picks = overlay.confluence(entries, gmap)
    tally = overlay.counts(entries)
    counts_txt = ", ".join(f"{k} {v}" for k, v in sorted(tally.items()))
    caveat = (
        "<div class='legend'>These screens rank the PRESENT only. Vectora "
        "holds a single fundamentals snapshot, so this data is deliberately "
        "kept out of model training &mdash; attaching today&rsquo;s P/E to a "
        "2015 row would let the model see which companies later turned out "
        "profitable.</div></section>")
    if not picks:
        return ("<section><h2>Technicals and fundamentals together</h2>"
                "<div class='note'>No stock currently clears both a bullish "
                "posture and a supporting fundamental screen. That is a real "
                "answer rather than a gap, so it is printed instead of padded."
                f"<br><span class='muted'>Board-wide screen hits: "
                f"{_esc(counts_txt)}</span></div>" + caveat)
    body = "".join(
        f"<tr><td class='sym'>{_esc(e['symbol'])}</td>"
        f"<td>{_band_pill(e['band'])}</td>"
        f"<td>{_screen_badges(e)}</td>"
        f"<td>{_esc(e.get('sector') or '?')}</td>"
        f"<td class='small'>"
        f"{'; '.join(_esc(s['detail']) for s in e['screens'])}</td></tr>"
        for e in picks[:40])
    return (
        "<section><h2>Technicals and fundamentals together <span class='muted "
        f"small'>&mdash; {len(picks)} of {len(entries)}</span></h2>"
        "<div class='note'>Stocks whose technical posture is bullish "
        "<i>and</i> that clear at least one of the value, income or quality "
        "screens &mdash; with anything carrying accumulated losses or a thin "
        "free float excluded outright. A technical Strong Buy on a hollow "
        "company is a different proposition and does not belong ranked beside "
        f"these.<br><span class='muted'>Board-wide screen hits: "
        f"{_esc(counts_txt)}</span></div>"
        "<div class='scroll'><table class='tbl'><thead><tr><th>Symbol</th>"
        "<th>Posture</th><th>Screens</th><th>Sector</th><th>Why</th>"
        "</tr></thead><tbody>" + body + "</tbody></table></div>" + caveat)


def _verification(con, symbols: list[str], days: int = 12) -> str:
    """What we said about each watchlist stock, and what it then did.

    The aggregate evidence table measures the system correctly and convinces
    nobody: a reader cannot check "+8.2pp across 133,592 cases" against
    anything. This section is the same claim at a scale a person can verify
    — one stock, one day, one reading, and the move that followed. Rows too
    recent to have finished say pending rather than borrowing a number.
    """
    from vectora import verify
    panel = verify.load_panel(con)
    blocks = []
    for sym in symbols:
        rows = verify.history(con, sym, days=days, panel=panel)
        if not rows:
            continue
        card = verify.scorecard(rows, horizon=5)
        body = []
        for r in rows:
            def move(v):
                if v is None:
                    return "<td class='num muted'>pending</td>"
                cls = "pos" if v > 0 else ("neg" if v < 0 else "")
                return f"<td class='num {cls}'>{v * 100:+.2f}%</td>"
            rd = r["readings"]
            body.append(
                f"<tr><td class='num'>{_esc(r['date'])}</td>"
                f"<td class='num'>{r['close']:,.2f}</td>"
                f"<td>{_band_pill(r['summary'])}</td>"
                f"<td class='num'>{_esc(rd['RSI(14)'] or '-')}</td>"
                f"<td class='num'>{_esc(rd['MACD hist'] or '-')}</td>"
                f"<td class='num'>{_esc(rd['CCI(20)'] or '-')}</td>"
                f"<td class='num'>{_esc(rd['ADX(14)'] or '-')}</td>"
                f"{move(r['ret_5d'])}{move(r['ret_10d'])}</tr>")
        note = ""
        if card["bullish_n"]:
            note = (f"<div class='legend'>Over these {card['n']} gradable "
                    f"days, {card['bullish_n']} carried a bullish posture and "
                    f"{card['bullish_up']:.0%} of those were higher five days "
                    f"later (mean {card['mean_after_bullish'] * 100:+.2f}%). "
                    "A dozen days is a spot check, not evidence — the "
                    "measured edge above rests on 13 years.</div>")
        blocks.append(
            f"<div class='grp'><h3>{_esc(sym)}</h3><div class='scroll'>"
            "<table class='tbl'><thead><tr><th>Date</th><th>Close</th>"
            "<th>Posture</th><th>RSI</th><th>MACD</th><th>CCI</th>"
            "<th>ADX</th><th>Next 5d</th><th>Next 10d</th></tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table></div>{note}</div>")
    if not blocks:
        return ""
    return (
        "<section><h2>Check it yourself "
        "<span class='muted small'>&mdash; what we said, and what happened"
        "</span></h2>"
        "<div class='note'>For each watchlist stock: the posture and the key "
        "readings on each of the last twelve trading days, beside the move "
        "that actually followed. Take any row to your broker's chart and "
        "confirm it. <b>Nothing here is a trade instruction</b> — no entry, "
        "no stop, no size. It is a direction and the distance price then "
        "travelled.</div>" + "".join(blocks) + "</section>")


def build(con, date_str: str | None = None,
          out_path: Path = DEFAULT_OUT, board_n: int = 20) -> Path:
    date_str = date_str or str(con.execute(
        "SELECT max(date) FROM ta_ratings").fetchone()[0])
    groups = load_watchlist()
    sections = []

    all_rows = ranked(con, date_str, limit=10000)
    gmap = gauges_for(con, date_str)
    fmap = _fundamentals(con, date_str)
    lmap = levels_for(con, date_str)
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
        overlay.annotate(all_rows, fmap)
        sections.append(_evidence(con))
        sections.append(_gauge_evidence(con))
        sections.append(_rotation(con, date_str))
        sections.append(_confluence(all_rows, gmap))

        wl = []
        for name, syms in groups.items():
            entries = ranked(con, date_str, symbols=list(syms), limit=200)
            missing = sorted(set(syms) - {e["symbol"] for e in entries})
            note = (f"<div class='legend'>Not rated today: "
                    f"{', '.join(missing)}</div>" if missing else "")
            wl.append(f"<div class='grp'><h3>{_esc(name)}</h3>"
                      + _table(entries, gmap, fmap, lmap) + note + "</div>")
        watch = [sym for syms in groups.values() for sym in syms]
        sections.append(_verification(con, sorted(set(watch))))
        sections.append(
            "<section><h2>Your watchlist</h2><div class='note'>Ranked within "
            "each group by technical score. Open <i>why</i> on any row to see "
            "exactly which indicators voted and in which direction.</div>"
            + "".join(wl) + "</section>")

        top = all_rows[:board_n]
        bottom = ranked(con, date_str, limit=board_n, ascending=True)
        sections.append(
            "<section><h2>Quick scan</h2>"
            f"<div class='grp'><h3>Strongest {board_n}</h3>"
            + _table(top, gmap, fmap, lmap) + "</div>"
            f"<div class='grp'><h3>Weakest {board_n}</h3>"
            + _table(bottom, gmap, fmap, lmap) + "</div></section>")

        # every rated symbol on the exchange, grouped by sector
        by_sector: dict = {}
        for r in all_rows:
            by_sector.setdefault(r.get("sector") or "Unclassified", []).append(r)
        blocks = []
        for sector in sorted(by_sector):
            rows = sorted(by_sector[sector], key=lambda x: -x["score"])
            blocks.append(f"<div class='grp'><h3>{_esc(sector)} "
                          f"<span class='muted'>({len(rows)})</span></h3>"
                          + _table(rows, gmap, fmap, lmap, full=True)
                          + "</div>")
        sections.append(
            f"<section><h2>Every listed security &mdash; {len(all_rows)} "
            "rated today</h2><div class='note'>Complete coverage of the "
            "exchange, grouped by sector and ranked by technical score within "
            "each. Open <i>why</i> on any row for the full per-stock review: "
            "all 26 indicator votes with their reasons, plus that "
            "company&rsquo;s fundamentals.</div>"
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
  <b>A technical posture is a mechanical summary of indicator states</b> —
  not a forecast, and <b>not investment advice</b>. Every row carries two
  independent reads: the six-family score (MACD, RSI, Bollinger, MA cross,
  SuperTrend, candlesticks) and the selected indicator set, split into
  a moving-average trend gauge and an oscillator gauge. Each band's measured
  historical edge is printed below so you can judge it on evidence.
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
