"""Obsidian vault generator v1 (spec §7): Journal, signal Prediction notes,
Company notes, Home dashboard. Machine content lives strictly between the
markers; anything a human writes outside them survives regeneration
byte-identical (tested). Notes use [[wiki-links]] so Obsidian's graph view
is the knowledge graph."""
from pathlib import Path

from vectora.collect import dse_company
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


def _fmt(v, spec="{:.2f}", dash="-"):
    return dash if v is None else spec.format(v)


def _write_companies(con, date_str: str, vault_dir: Path,
                     extra: set | None = None) -> int:
    """One note per rated symbol, carrying the whole day's analysis.

    `extra` guarantees a note for symbols that signalled or had an event
    today even if they carry no technical rating — a signal without a note
    would be a hole in exactly the place a reader looks first.
    """
    rows = con.execute(
        """
        SELECT r.symbol, s.sector, s.category, r.band, r.score, r.rsi,
               r.st_dir, g.summary_band, g.ma_band, g.ma_buy, g.ma_sell,
               g.osc_band, g.osc_buy, g.osc_sell, l.pivot_point,
               l.nearest_res, l.nearest_sup, l.room_up, l.room_dn,
               l.hi_252d, l.lo_252d
        FROM ta_ratings r
        LEFT JOIN symbols s ON s.symbol = r.symbol
        LEFT JOIN ta_gauges g ON g.symbol = r.symbol AND g.date = r.date
        LEFT JOIN ta_levels l ON l.symbol = r.symbol AND l.date = r.date
        WHERE r.date = ?
        """, [date_str]).fetchall()
    rated = {r[0] for r in rows}
    for sym in sorted((extra or set()) - rated):
        meta = con.execute(
            "SELECT sector, category FROM symbols WHERE symbol = ?",
            [sym]).fetchone() or (None, None)
        rows.append((sym, meta[0], meta[1], None, None, None, None,
                     None, None, None, None, None, None, None,
                     None, None, None, None, None, None, None))
    if not rows:
        return 0

    funds = {r[0]: r for r in con.execute(
        """
        WITH latest AS (
            SELECT symbol, max(as_of) AS as_of FROM fundamentals GROUP BY 1)
        SELECT f.symbol, f.trailing_pe, f.latest_dividend_pct,
               f.dividend_year, f.market_cap_mn, f.free_float_mcap_mn
        FROM fundamentals f JOIN latest l
          ON l.symbol = f.symbol AND l.as_of = f.as_of
        """).fetchall()}
    scores = {r[0]: (r[1], r[2]) for r in con.execute(
        """
        SELECT p.symbol, count(*), sum(CASE WHEN o.hit THEN 1 ELSE 0 END)
        FROM predictions p JOIN outcomes o ON o.prediction_id = p.id
        GROUP BY 1
        """).fetchall()}

    written = 0
    for row in rows:
        (sym, sector, cat, band, score, rsi, st_dir, summary, ma_band,
         ma_buy, ma_sell, osc_band, osc_buy, osc_sell, pivot, res, sup,
         room_up, room_dn, hi252, lo252) = row
        trend = "up" if (st_dir or 0) > 0 else (
            "down" if (st_dir or 0) < 0 else "flat")
        body = [f"# {sym}", "",
                f"**{sector or 'Unclassified'}** | category {cat or '?'} | "
                f"as of [[Journal/{date_str}|{date_str}]]", ""]
        if sector:
            body += [f"Sector view: [[Sectors/{sector}|{sector}]]", ""]
        body += ["## Technical posture", "",
                 "| gauge | verdict | detail |", "|---|---|---|",
                 f"| Summary (26 indicators) | **{summary or '-'}** | |",
                 f"| Moving averages | {ma_band or '-'} | "
                 f"{ma_buy or 0} up / {ma_sell or 0} down |",
                 f"| Oscillators | {osc_band or '-'} | "
                 f"{osc_buy or 0} up / {osc_sell or 0} down |",
                 f"| Six-family score | {band or '-'} | "
                 f"{'-' if score is None else format(score, '+d')} |", "",
                 f"RSI {_fmt(rsi, '{:.0f}')} | SuperTrend {trend}", "",
                 "## Levels", "",
                 f"- Monthly pivot: {_fmt(pivot)}",
                 f"- Nearest resistance: {_fmt(res)}"
                 + (f" ({room_up * 100:+.1f}% away)"
                    if room_up is not None else ""),
                 f"- Nearest support: {_fmt(sup, dash='none below price')}"
                 + (f" ({room_dn * 100:.1f}% away)"
                    if room_dn is not None else ""),
                 f"- 52-week range: {_fmt(lo252)} to {_fmt(hi252)}", ""]

        f = funds.get(sym)
        if f:
            pe, div, div_yr, mcap, ff = f[1], f[2], f[3], f[4], f[5]
            float_pct = (ff / mcap * 100) if (ff and mcap) else None
            thin = (float_pct is not None and float_pct <= 20)
            body += ["## Fundamentals", "",
                     f"- Trailing P/E: {_fmt(pe)}",
                     f"- Latest cash dividend: {_fmt(div)}% of face"
                     + (f" for {div_yr}" if div_yr else ""),
                     f"- Market cap: {_fmt(mcap, '{:,.0f}')} mn",
                     f"- Free float: {_fmt(float_pct, '{:.0f}')}% of market cap"
                     + ("  **thin float - easier to move, easier to trap**"
                        if thin else ""), ""]

        resolved, hits = scores.get(sym, (0, 0))
        body += ["## Track record", "",
                 f"{hits or 0}/{resolved or 0} resolved predictions reached "
                 "their target", ""]
        _write_machine(vault_dir / "Companies" / f"{sym}.md", "\n".join(body))
        written += 1
    return written


def _write_indexes(con, date_str: str, vault_dir: Path) -> int:
    """Index notes: the entry points that make the vault navigable.

    Without these the only way into 449 company notes is the search bar.
    Each index is a live list for one question a reader actually asks —
    what is rated Strong Buy, what passes the value screen, what carries a
    float small enough to be dangerous.
    """
    from vectora.ta import overlay

    rows = con.execute(
        """
        SELECT g.symbol, g.summary_band, f.trailing_pe, f.market_cap_mn,
               f.free_float_mcap_mn, f.reserve_surplus_mn,
               c.paid_up_capital_mn, f.latest_dividend_pct, f.face_value,
               p.close
        FROM ta_gauges g
        LEFT JOIN (
            SELECT f.* FROM fundamentals f JOIN (
                SELECT symbol, max(as_of) AS as_of FROM fundamentals
                GROUP BY 1) l
              ON l.symbol = f.symbol AND l.as_of = f.as_of) f
          ON f.symbol = g.symbol
        LEFT JOIN (
            SELECT c.* FROM company_snapshot c JOIN (
                SELECT symbol, max(as_of) AS as_of FROM company_snapshot
                GROUP BY 1) l
              ON l.symbol = c.symbol AND l.as_of = c.as_of) c
          ON c.symbol = g.symbol
        LEFT JOIN prices p ON p.symbol = g.symbol AND p.date = g.date
        WHERE g.date = ?
        """, [date_str]).fetchall()
    if not rows:
        return 0

    buckets: dict = {}
    postures: dict = {}
    for (sym, band, pe, mcap, ff, reserve, paid_up, div_pct, face,
         close) in rows:
        postures.setdefault(band or "Unrated", []).append(sym)
        fund = {"trailing_pe": pe, "market_cap_mn": mcap,
                "free_float_mcap_mn": ff, "reserve_surplus_mn": reserve,
                "paid_up_capital_mn": paid_up,
                "latest_dividend_pct": div_pct, "face_value": face}
        fund.update(dse_company.derive_metrics(fund, close))
        for flag in overlay._screen({"symbol": sym, "fundamentals": fund}):
            buckets.setdefault(flag["screen"], []).append((sym, flag["detail"]))

    blurbs = {
        "Value": "Trailing P/E under 15 on genuinely positive earnings. A "
                 "negative P/E is a loss, not a bargain, and is excluded.",
        "Income": "Cash dividend yield of 4% or better at today's price. "
                  "Bonus issues are excluded — stock is not income.",
        "Quality": "Reserves above 3x paid-up capital. The textbook bar of "
                   "1x is the DSE median, so it selects half the exchange "
                   "and means nothing.",
        "Thin float": "20% or less of the company can actually trade. This "
                      "is a RISK list, never an opportunity list — a small "
                      "float is what makes a price easy to push around.",
        "Impaired": "Carrying accumulated losses instead of reserves.",
    }
    written = 0
    for screen, members in buckets.items():
        body = [f"# {screen}", "",
                f"{len(members)} companies | as of "
                f"[[Journal/{date_str}|{date_str}]]", "",
                blurbs.get(screen, ""), "",
                "| company | why |", "|---|---|"]
        body += [f"| [[{s}]] | {d} |" for s, d in sorted(members)]
        _write_machine(vault_dir / "Screens" / f"{screen}.md",
                       "\n".join(body))
        written += 1

    for band in ("Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"):
        members = sorted(postures.get(band, []))
        if not members:
            continue
        body = [f"# {band}", "",
                f"{len(members)} securities | as of "
                f"[[Journal/{date_str}|{date_str}]]", "",
                "Summary gauge across all 26 indicators. See "
                "[[Evidence]] for what this posture has historically been "
                "worth.", "",
                " · ".join(f"[[{m}]]" for m in members), ""]
        _write_machine(vault_dir / "Postures" / f"{band}.md", "\n".join(body))
        written += 1
    return written


def _write_evidence(con, vault_dir: Path) -> int:
    """The measured edge of every posture, as a note rather than a webpage."""
    rows = con.execute(
        "SELECT gauge, band, n, hit_rate, base_rate FROM ta_gauge_stats "
        "WHERE horizon = 10 ORDER BY gauge, hit_rate DESC").fetchall()
    if not rows:
        return 0
    labels = {"summary": "Summary (all 26)",
              "moving_averages": "Moving averages (15)",
              "oscillators": "Oscillators (11)"}
    body = ["# Evidence", "",
            "Every posture replayed across 13 years of DSE history and "
            "scored against what actually happened: did the stock gain 5% "
            "within 10 trading days? **Edge** is the band's hit rate minus "
            "the market's own base rate over the same windows.", "",
            "| gauge | posture | n | hit | base | edge |",
            "|---|---|---|---|---|---|"]
    body += [f"| {labels.get(g, g)} | {b} | {n:,} | {h:.1%} | {base:.1%} | "
             f"{(h - base) * 100:+.1f} pp |" for g, b, n, h, base in rows]
    body += ["", "> This measures whether price *touched* +5% at any point "
             "in the window. It is not a profitable round trip: it ignores "
             "commission, spread and exit timing. Both oscillator extremes "
             "beat the base rate, which is a volatility artefact rather "
             "than directional skill."]
    _write_machine(vault_dir / "Evidence.md", "\n".join(body))
    return 1


def _write_sectors(con, date_str: str, vault_dir: Path) -> int:
    """One note per sector: rotation phase plus its members.

    These are what give the graph its shape. Without them every company is
    an isolated node hanging off a journal entry; with them the vault has
    the structure the analysis actually has.
    """
    rows = con.execute(
        """
        SELECT sector, n_symbols, ret_21d, rs_21d, rs_momentum, quadrant,
               ret_63d
        FROM sector_rs WHERE date = ? ORDER BY rs_21d DESC
        """, [date_str]).fetchall()
    if not rows:
        return 0
    written = 0
    for sector, n_sym, r21, rs21, mom, quad, r63 in rows:
        members = [m[0] for m in con.execute(
            """
            SELECT r.symbol FROM ta_ratings r
            JOIN symbols s ON s.symbol = r.symbol
            WHERE r.date = ? AND s.sector = ? ORDER BY r.score DESC
            """, [date_str, sector]).fetchall()]
        body = [f"# {sector}", "",
                f"Phase: **{quad}** | as of [[Journal/{date_str}|{date_str}]]",
                "", "| measure | value |", "|---|---|",
                f"| 21-day return | {_fmt(r21, '{:+.2%}')} |",
                f"| vs market, 21 days | {_fmt(rs21, '{:+.2%}')} |",
                f"| momentum | {_fmt(mom, '{:+.2%}')} |",
                f"| 63-day return | {_fmt(r63, '{:+.2%}')} |",
                f"| constituents | {n_sym} |", ""]
        if members:
            body += ["## Members, strongest first", "",
                     " · ".join(f"[[{m}]]" for m in members), ""]
        _write_machine(vault_dir / "Sectors" / f"{sector}.md", "\n".join(body))
        written += 1
    return written


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
        SELECT e.symbol, e.title, coalesce(l.event_type, 'unclassified'),
               max(coalesce(l.materiality, 1)) AS m
        FROM events e LEFT JOIN event_labels l ON l.event_id = e.id
        WHERE e.post_date = ? AND e.symbol IS NOT NULL
          AND coalesce(l.materiality, 1) >= 1
        -- the same announcement is often filed more than once with a
        -- slightly different body, so the sha256 id does not collapse them;
        -- group on what a reader actually sees
        GROUP BY 1, 2, 3
        ORDER BY m DESC
        LIMIT 20
        """, [date_str]).fetchall()
    zwatch = con.execute(
        "SELECT symbol, kind, score, phase FROM zwatch WHERE date = ? "
        "ORDER BY kind, score DESC", [date_str]).fetchall()

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
                  for sym, title, etype, _m in events]
    if zwatch:
        lines.append("")
        lines.append("## Z-watch")
        for sym, kind, score, phase in zwatch:
            tag = (f"pump {score:.0f} ({phase})" if kind == "pump"
                   else f"pre-announcement footprint (vol_z {score})")
            lines.append(f"- [[{sym}]] {tag}")

    # sector phases and the day's extremes: these are what make a journal
    # entry worth opening a month later, and they are what wire the graph
    # together — without them a journal note links only to a few signals
    sectors = con.execute(
        "SELECT sector, quadrant, rs_21d FROM sector_rs WHERE date = ? "
        "ORDER BY rs_21d DESC", [date_str]).fetchall()
    if sectors:
        lines += ["", "## Sector phases", ""]
        lines += [f"- [[Sectors/{s}|{s}]] — **{q}** "
                  f"({rs * 100:+.1f} pp vs market)"
                  for s, q, rs in sectors if rs is not None]
    movers = con.execute(
        "SELECT symbol, band, score FROM ta_ratings WHERE date = ? "
        "ORDER BY score DESC LIMIT 10", [date_str]).fetchall()
    laggards = con.execute(
        "SELECT symbol, band, score FROM ta_ratings WHERE date = ? "
        "ORDER BY score ASC LIMIT 10", [date_str]).fetchall()
    if movers:
        lines += ["", "## Strongest technical postures", "",
                  " · ".join(f"[[{s}]] ({sc:+d})" for s, _b, sc in movers),
                  "", "## Weakest", "",
                  " · ".join(f"[[{s}]] ({sc:+d})" for s, _b, sc in laggards)]
    if not preds:
        lines += ["", "> No predictions were generated on this date. Market "
                  "data was collected, but the forecasting pipeline was not "
                  "running. Predictions are never backfilled — a forecast "
                  "made now with a model trained on later data would be "
                  "hindsight, not a record."]
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

    # Company and sector notes ------------------------------------------------
    # Every symbol rated today, not only those that produced a signal. The
    # screener does the full analysis on 400+ securities daily; writing four
    # lines for the handful that signalled discarded the rest and left the
    # graph far sparser than the work standing behind it.
    #
    # These are CURRENT-STATE notes: one per company describing where it
    # stands now. Backfilling an older journal must not rewrite them with
    # that day's stale posture, so they are written only when generating for
    # the most recent date that has ratings.
    latest = con.execute("SELECT max(date) FROM ta_ratings").fetchone()[0]
    if latest is None or str(latest) == str(date_str):
        touched = {sig[1] for sig in signals} | {e[0] for e in events}
        n += _write_companies(con, date_str, vault_dir, extra=touched)
        n += _write_sectors(con, date_str, vault_dir)
        n += _write_indexes(con, date_str, vault_dir)
        n += _write_evidence(con, vault_dir)

    # Home dashboard ----------------------------------------------------------
    total = con.execute("SELECT count(*) FROM predictions").fetchone()[0]
    resolved = con.execute("SELECT count(*) FROM outcomes").fetchone()[0]
    home = ["# Vectora", "",
            f"Latest run: [[Journal/{date_str}|{date_str}]] | "
            f"{len(signals)} signal(s)", "",
            f"Lifetime: {total} predictions, {resolved} resolved", "",
            "## Start here", "",
            "- [[Postures/Strong Buy|Strong Buy today]] · "
            "[[Postures/Strong Sell|Strong Sell today]]",
            "- [[Screens/Value|Value]] · [[Screens/Income|Income]] · "
            "[[Screens/Quality|Quality]]",
            "- [[Screens/Thin float|Thin float]] — risk list, not an "
            "opportunity list",
            "- [[Evidence]] — what each posture has historically been worth",
            "",
            "## Reading the graph", "",
            "Colours are live search queries, so the graph is a market view "
            "rather than decoration:", "",
            "| colour | meaning |", "|---|---|",
            "| green | company currently Strong Buy on the summary gauge |",
            "| red | currently Strong Sell |",
            "| amber | thin float — risk overrides posture here |",
            "| teal | sector hubs |",
            "| violet | signal prediction notes |",
            "| slate | daily journal entries |", "",
            "Set the local graph to **depth 2** (its settings cog) — depth 1 "
            "shows a company and its sector, depth 2 shows the sector's "
            "other members, which is the comparison worth having.", "",
            "_Research tool, not investment advice._"]
    _write_machine(vault_dir / "Home.md", "\n".join(home))
    n += 1
    return {"notes": n, "signals": len(signals)}
