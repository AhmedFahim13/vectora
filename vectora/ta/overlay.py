"""Fundamental overlay on the technical screen (spec: Phase 6D).

This deliberately does NOT feed the model. The fundamentals table holds a
single snapshot; attaching today's P/E to a 2015 training row would let the
model see which companies later turned out profitable — textbook lookahead,
and the kind that produces a beautiful backtest and loses real money. These
screens rank the PRESENT, where no such leak exists, and the snapshots
accumulate over time so point-in-time fundamentals become legitimately
available later.

Four screens, each answering a different question:

  value    is it cheap relative to earnings, and are those earnings real
  income   does it actually pay cash, and how much against today's price
  quality  has it built reserves against its paid-up capital
  float    how much of it can actually trade — the manipulation-risk read

The float screen matters more here than on most exchanges. A small free
float is what makes DSE pump patterns mechanically possible, so it is
reported as a RISK flag, never as an opportunity.
"""

_MIN_PE, _MAX_PE = 0.0, 15.0
_MIN_YIELD = 0.04
_LOW_FLOAT = 0.20
# Thresholds are set from the exchange's own distribution, not from habit.
# "Reserves exceed paid-up capital" is the textbook quality test, but on the
# DSE the MEDIAN listed company sits at 1.08x — that bar selects half the
# board and means nothing. Measured across 720 equities: p25 0.18, p50 1.08,
# p75 2.79. The bar is set at 3.0x, roughly the top quartile.
_MIN_RESERVE_RATIO = 3.0


def _screen(entry: dict) -> list[dict]:
    """Flags a single symbol qualifies for, with the number behind each."""
    f, flags = entry.get("fundamentals") or {}, []
    pe = f.get("trailing_pe")
    if pe is not None and _MIN_PE < pe <= _MAX_PE:
        flags.append({"screen": "Value", "detail":
                      f"trailing P/E {pe:.1f} on positive earnings"})
    y = f.get("dividend_yield")
    if y is not None and y >= _MIN_YIELD:
        flags.append({"screen": "Income", "detail":
                      f"cash yield {y * 100:.1f}% at today's price"})
    reserve, cap = f.get("reserve_surplus_mn"), f.get("paid_up_capital_mn")
    if reserve is not None and cap:
        ratio = reserve / cap
        if ratio >= _MIN_RESERVE_RATIO:
            flags.append({"screen": "Quality", "detail":
                          f"reserves are {ratio:.1f}x paid-up capital "
                          "(exchange median is 1.1x)"})
        elif reserve < 0:
            flags.append({"screen": "Impaired", "detail":
                          f"accumulated losses of {abs(reserve):,.0f} mn"})
    free, mcap = f.get("free_float_mcap_mn"), f.get("market_cap_mn")
    # A reported free float of exactly zero is DSE saying "not applicable",
    # not "nothing can trade": every one of the 35 mutual funds and 22 bonds
    # reports 0. Flagging those as thin float produced the nonsense line
    # "only 0% of the company can trade" on instruments the screen was never
    # about. Unknown is not the same as dangerous.
    if free and mcap:
        share = free / mcap
        if share <= _LOW_FLOAT:
            flags.append({"screen": "Thin float", "detail":
                          f"only {share * 100:.0f}% of the company can trade "
                          "— easier to move, easier to trap"})
    return flags


def annotate(entries: list[dict], fmap: dict) -> list[dict]:
    """Attaches fundamentals and screen flags to each screener row."""
    for e in entries:
        e["fundamentals"] = fmap.get(e["symbol"]) or {}
        e["screens"] = _screen(e)
    return entries


def confluence(entries: list[dict], gmap: dict) -> list[dict]:
    """Rows where a bullish technical posture meets a supporting fundamental.

    This is the combination worth a reader's attention and the one no single
    column shows: technically strong AND cheap, or technically strong AND
    paying real cash. A technical Strong Buy on a company with accumulated
    losses and a thin float is a different proposition entirely, and it is
    excluded here rather than quietly ranked alongside.
    """
    out = []
    for e in entries:
        g = gmap.get(e["symbol"]) or {}
        bullish = (g.get("summary_band") in ("Buy", "Strong Buy")
                   or e.get("band") in ("Buy", "Strong Buy"))
        names = {s["screen"] for s in e.get("screens", [])}
        supporting = names & {"Value", "Income", "Quality"}
        disqualifying = names & {"Impaired", "Thin float"}
        if bullish and supporting and not disqualifying:
            out.append(e)
    return sorted(out, key=lambda x: -len(x["screens"]))


def counts(entries: list[dict]) -> dict:
    tally: dict = {}
    for e in entries:
        for s in e.get("screens", []):
            tally[s["screen"]] = tally.get(s["screen"], 0) + 1
    return tally
