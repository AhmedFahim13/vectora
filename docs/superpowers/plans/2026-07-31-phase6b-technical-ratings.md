# Vectora Phase 6B: Technical Rating Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A TradingView-style 5-band technical posture (Strong Buy → Strong Sell) for every DSE symbol, built from MACD / Bollinger / SuperTrend / MA-cross / RSI / candlestick patterns, with per-indicator rationale — and, unlike any premium site, **each band carrying its measured historical hit rate** so the rating is evidence rather than opinion.

**Architecture:** `vectora/ta/indicators.py` computes the six indicator families as pure polars expressions over the existing price panel. `vectora/ta/rating.py` converts each into a vote in {-2,-1,0,+1,+2}, sums to a raw score, maps to a 5-band posture, and returns a per-indicator breakdown (the drop-down rationale). `vectora/ta/validate.py` replays the rating over all 13 years of history and measures what each band *actually did* over the next 5/10 days — turning "Strong Buy" from a claim into a statistic. `vectora/ta/screener.py` ranks the whole board plus the named watchlist, feeding a new dashboard page.

**Framing decision (deliberate, documented):** the bands keep their conventional names because that is the product the client recognises — but the UI never presents them bare. Every band is shown with its historical base rate and n, the page states plainly that a technical posture is a mechanical indicator summary and not advice, and Vectora's calibrated model probability sits beside it as the separately-validated number. If validation shows a band has no edge over the base rate, the page says so rather than hiding it.

**Tech Stack:** existing only (polars, DuckDB). No new dependencies — no TA-Lib (C build, breaks zero-cost CI); every indicator is ~5 lines of polars.

**Existing contracts:** `features/base.load_panel(con)` → symbol/date/open/high/low/close/ycp/trades/value_mn/volume/ret sorted by symbol,date (already corporate-action-adjusted). `labels.make_labels(panel, thresholds, horizons, continuous=True)` → `y_gX_hH`, `fwdmax_hH`, `fwdmin_hH`. `universe.tradable_universe(con, as_of, min_median_value_mn)`. `symbols(symbol, sector, category, instrument_type)`. `dashboard._esc/_kpi` + `_TEMPLATE`. Bulk-seed rule for synthetic prices in tests. Fast tests: `uv run pytest -m "not slow"` (currently 208). Branch `phase-6b-ta` off main.

**Watchlist mapping (verified against the live symbol master 2026-07-31 — three fuzzy matches were wrong and are corrected here):**

```
Govt/SOE : TITASGAS DESCO POWERGRID PADMAOIL MPETROLEUM BSC BSCPLC
           EASTRNLUB ECABLES NTLTUBES ICB
Insurance: EIL ICICL MERCINS NORTHRNINS PIONEERINS PURABIGEN RUPALIINS
           SONARBAINS STANDARINS
Life ins : CLICL DELTALIFE POPULARLIF PRIMELIFE SANDHANINS
Mutual   : 1STPRIMFMF ABB1STMF AIBL1STIMF CAPITECGBF CAPMBDBLMF DBH1STMF
           EBLNRBMF FBFIF GLDNJMF GREENDELMF PF1STMF
Textile  : ARGONDENIM DSSL ENVOYTEX ETL FEKDIL HRTEX MATINSPINN MHSML PTL
           SHEPHERD SIMTEX SAIHAMCOT SAIHAMTEX KDSALTD
Pharma   : ACMELAB ACMEPL BEACONPHAR BXPHARMA TECHNODRUG
IT       : BDCOM EGEN GENEXIL ITC
Cement   : CONFIDCEM LHB RAKCERAMIC MONNOCERA SPCERAMICS
```

Corrected: EasternLub→**EASTRNLUB** (not EASTERNINS, an insurer) · Mercantile Ins→**MERCINS** (not MERCANBANK, a bank) · Meghna Petroleum→**MPETROLEUM** (not MEGHNAPET, a plastics maker).

**File structure:**

```
vectora/ta/__init__.py, indicators.py, rating.py, validate.py, screener.py
vectora/config/watchlist.yaml
vectora/db.py                       # + ta_ratings, ta_band_stats tables
vectora/dashboard.py                # + screener page link/section
vectora/__main__.py                 # + ta stage
docs/dashboard/screener.html        # generated
tests/ta/…
```

---

### Task 1: Indicator library

**Files:**
- Create: `vectora/ta/__init__.py` (empty), `vectora/ta/indicators.py`
- Test: `tests/ta/__init__.py` (empty), `tests/ta/test_indicators.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/ta/test_indicators.py
import numpy as np
import polars as pl

from vectora.ta import indicators as ind


def _series(closes, highs=None, lows=None, opens=None):
    n = len(closes)
    return pl.DataFrame({
        "symbol": ["X"] * n,
        "date": [f"2026-01-{i + 1:02d}" for i in range(n)],
        "open": opens or closes,
        "high": highs or [c * 1.01 for c in closes],
        "low": lows or [c * 0.99 for c in closes],
        "close": closes,
    }).with_columns(pl.col("date").str.to_date())


def test_macd_positive_on_uptrend():
    up = [100 * (1.01 ** i) for i in range(60)]
    out = ind.add_all(_series(up))
    last = out.tail(1).row(0, named=True)
    assert last["macd_hist"] > 0        # fast EMA above slow, rising
    assert last["macd"] > last["macd_signal"]


def test_macd_negative_on_downtrend():
    down = [100 * (0.99 ** i) for i in range(60)]
    out = ind.add_all(_series(down))
    assert out.tail(1).row(0, named=True)["macd_hist"] < 0


def test_rsi_extremes():
    up = [100 * (1.02 ** i) for i in range(40)]
    down = [100 * (0.98 ** i) for i in range(40)]
    assert ind.add_all(_series(up)).tail(1).row(0, named=True)["rsi14"] > 70
    assert ind.add_all(_series(down)).tail(1).row(0, named=True)["rsi14"] < 30


def test_bollinger_position_and_width():
    flat = [100.0] * 30 + [130.0]
    out = ind.add_all(_series(flat))
    last = out.tail(1).row(0, named=True)
    assert last["bb_pos"] > 1.0          # closed above the upper band
    assert last["bb_width"] >= 0


def test_ma_cross_state_and_recency():
    # 40 flat then a sharp rally: fast MA crosses above slow
    closes = [100.0] * 40 + [100 * (1.03 ** i) for i in range(1, 30)]
    out = ind.add_all(_series(closes))
    last = out.tail(1).row(0, named=True)
    assert last["ma_fast"] > last["ma_slow"]
    assert last["ma_cross_up"] == 1


def test_supertrend_flips_direction():
    closes = [100 * (1.02 ** i) for i in range(40)] + \
             [100 * (1.02 ** 39) * (0.97 ** i) for i in range(1, 30)]
    out = ind.add_all(_series(closes))
    dirs = out["st_dir"].to_list()
    assert dirs[38] == 1                 # uptrend during the rally
    assert dirs[-1] == -1                # flipped down after the slide


def test_candles_detect_bullish_engulfing():
    # prior red candle, then a larger green candle engulfing it
    o = [100, 99, 96]
    c = [99, 96, 101]
    out = ind.add_all(_series(c, opens=o,
                              highs=[max(a, b) * 1.005 for a, b in zip(o, c, strict=True)],
                              lows=[min(a, b) * 0.995 for a, b in zip(o, c, strict=True)]))
    assert out.tail(1).row(0, named=True)["candle_bull"] == 1


def test_no_lookahead_in_any_indicator():
    """Indicator values for early rows must not change when later rows differ."""
    base = [100 + i * 0.5 for i in range(60)]
    a = ind.add_all(_series(base + [100.0] * 20))
    b = ind.add_all(_series(base + [400.0] * 20))
    cols = [c for c in a.columns if c not in ("symbol", "date", "open",
                                              "high", "low", "close")]
    ha, hb = a.head(60), b.head(60)
    for col in cols:
        va, vb = ha[col].to_list(), hb[col].to_list()
        for x, y in zip(va, vb, strict=True):
            if x is None and y is None:
                continue
            assert x == y or abs(x - y) < 1e-9, f"LOOKAHEAD in {col}"
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/ta -v` → FAIL (module missing).

- [ ] **Step 3: Implement `vectora/ta/indicators.py`**

```python
"""Classical technical indicators as polars expressions (spec: Phase 6B).

Every indicator is trailing-only — a value at row t uses rows <= t, which
the lookahead test pins. No TA-Lib: a C dependency would break the
zero-cost CI, and each of these is a handful of lines.

Columns added by add_all():
  macd, macd_signal, macd_hist      MACD(12,26,9) on close
  rsi14                             Wilder-style RSI, 14
  bb_mid, bb_up, bb_lo, bb_pos,     Bollinger(20, 2sd); bb_pos is 0 at the
  bb_width                          lower band, 1 at the upper
  ma_fast, ma_slow, ma_cross_up,    SMA(20)/SMA(50) + fresh-cross flags
  ma_cross_dn
  st_dir, st_line                   SuperTrend(10, 3) direction and stop
  candle_bull, candle_bear          engulfing / hammer / shooting-star flags
"""
import polars as pl

MACD_FAST, MACD_SLOW, MACD_SIG = 12, 26, 9
RSI_N = 14
BB_N, BB_SD = 20, 2.0
MA_FAST, MA_SLOW = 20, 50
ST_N, ST_MULT = 10, 3.0
CROSS_LOOKBACK = 5      # a cross counts as "fresh" for this many days


def _ema(col: str, span: int) -> pl.Expr:
    return pl.col(col).ewm_mean(span=span, adjust=False).over("symbol")


def add_all(df: pl.DataFrame) -> pl.DataFrame:
    d = df.sort(["symbol", "date"])

    # --- MACD ---
    d = d.with_columns(
        (_ema("close", MACD_FAST) - _ema("close", MACD_SLOW)).alias("macd"))
    d = d.with_columns(
        pl.col("macd").ewm_mean(span=MACD_SIG, adjust=False).over("symbol")
        .alias("macd_signal"))
    d = d.with_columns((pl.col("macd") - pl.col("macd_signal"))
                       .alias("macd_hist"))

    # --- RSI (Wilder smoothing via ewm alpha=1/n) ---
    chg = pl.col("close").diff().over("symbol")
    gain = pl.when(chg > 0).then(chg).otherwise(0.0)
    loss = pl.when(chg < 0).then(-chg).otherwise(0.0)
    d = d.with_columns(
        gain.ewm_mean(alpha=1 / RSI_N, adjust=False).over("symbol").alias("_g"),
        loss.ewm_mean(alpha=1 / RSI_N, adjust=False).over("symbol").alias("_l"))
    d = d.with_columns(
        (100 - 100 / (1 + pl.col("_g") / (pl.col("_l") + 1e-12))).alias("rsi14"))

    # --- Bollinger ---
    mid = pl.col("close").rolling_mean(BB_N).over("symbol")
    sd = pl.col("close").rolling_std(BB_N).over("symbol")
    d = d.with_columns(mid.alias("bb_mid"), sd.alias("_sd"))
    d = d.with_columns(
        (pl.col("bb_mid") + BB_SD * pl.col("_sd")).alias("bb_up"),
        (pl.col("bb_mid") - BB_SD * pl.col("_sd")).alias("bb_lo"))
    d = d.with_columns(
        ((pl.col("close") - pl.col("bb_lo"))
         / (pl.col("bb_up") - pl.col("bb_lo") + 1e-12)).alias("bb_pos"),
        ((pl.col("bb_up") - pl.col("bb_lo"))
         / (pl.col("bb_mid") + 1e-12)).alias("bb_width"))

    # --- MA cross ---
    d = d.with_columns(
        pl.col("close").rolling_mean(MA_FAST).over("symbol").alias("ma_fast"),
        pl.col("close").rolling_mean(MA_SLOW).over("symbol").alias("ma_slow"))
    above = (pl.col("ma_fast") > pl.col("ma_slow")).cast(pl.Int8)
    d = d.with_columns(above.alias("_above"))
    prev = pl.col("_above").shift(1).over("symbol")
    d = d.with_columns(
        ((pl.col("_above") == 1) & (prev == 0)).cast(pl.Int8).alias("_xup"),
        ((pl.col("_above") == 0) & (prev == 1)).cast(pl.Int8).alias("_xdn"))
    d = d.with_columns(
        (pl.col("_xup").rolling_sum(CROSS_LOOKBACK).over("symbol") > 0)
        .cast(pl.Int8).alias("ma_cross_up"),
        (pl.col("_xdn").rolling_sum(CROSS_LOOKBACK).over("symbol") > 0)
        .cast(pl.Int8).alias("ma_cross_dn"))

    # --- SuperTrend(10,3): ATR bands, direction flips on close breach ---
    prev_close = pl.col("close").shift(1).over("symbol")
    tr = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - prev_close).abs(),
        (pl.col("low") - prev_close).abs())
    d = d.with_columns(
        tr.ewm_mean(alpha=1 / ST_N, adjust=False).over("symbol").alias("_atr"))
    hl2 = (pl.col("high") + pl.col("low")) / 2
    d = d.with_columns(
        (hl2 - ST_MULT * pl.col("_atr")).alias("_st_lo"),
        (hl2 + ST_MULT * pl.col("_atr")).alias("_st_up"))
    # vectorised approximation of the recursive SuperTrend: direction is up
    # while close holds above the lower band, down while below the upper.
    d = d.with_columns(
        pl.when(pl.col("close") > pl.col("_st_up")).then(1)
        .when(pl.col("close") < pl.col("_st_lo")).then(-1)
        .otherwise(None).alias("_raw_dir"))
    d = d.with_columns(
        pl.col("_raw_dir").forward_fill().over("symbol").fill_null(0)
        .cast(pl.Int8).alias("st_dir"))
    d = d.with_columns(
        pl.when(pl.col("st_dir") == 1).then(pl.col("_st_lo"))
        .otherwise(pl.col("_st_up")).alias("st_line"))

    # --- candlesticks (engulfing, hammer, shooting star) ---
    body = (pl.col("close") - pl.col("open"))
    rng = (pl.col("high") - pl.col("low")) + 1e-12
    p_open = pl.col("open").shift(1).over("symbol")
    p_close = pl.col("close").shift(1).over("symbol")
    bull_engulf = ((p_close < p_open) & (pl.col("close") > pl.col("open"))
                   & (pl.col("close") >= p_open) & (pl.col("open") <= p_close))
    bear_engulf = ((p_close > p_open) & (pl.col("close") < pl.col("open"))
                   & (pl.col("close") <= p_open) & (pl.col("open") >= p_close))
    lower_wick = pl.min_horizontal(pl.col("open"), pl.col("close")) - pl.col("low")
    upper_wick = pl.col("high") - pl.max_horizontal(pl.col("open"), pl.col("close"))
    hammer = (lower_wick > 2 * body.abs()) & (upper_wick < body.abs()) & (body > 0)
    shooting = (upper_wick > 2 * body.abs()) & (lower_wick < body.abs()) & (body < 0)
    d = d.with_columns(
        ((bull_engulf | hammer).cast(pl.Int8)).alias("candle_bull"),
        ((bear_engulf | shooting).cast(pl.Int8)).alias("candle_bear"),
        (body / rng).alias("candle_body_frac"))

    drop = [c for c in d.columns if c.startswith("_")]
    return d.drop(drop)
```

Debug note: if `ewm_mean(...).over("symbol")` is rejected by the installed polars, compute per-symbol with `group_by("symbol").map_groups(...)` — adjust the IMPLEMENTATION, never the tests. The lookahead test is the contract that matters most; if it fails, the indicator is wrong, not the test.

- [ ] **Step 4: Run tests** — `uv run pytest tests/ta -v` → 8 passed; fast suite; ruff.

- [ ] **Step 5: Commit**

```bash
git add vectora/ta tests/ta
git commit -m "feat: technical indicator library (MACD, RSI, Bollinger, MA cross, SuperTrend, candles)"
```

---

### Task 2: Rating engine with per-indicator rationale

**Files:**
- Modify: `vectora/db.py` (SCHEMA)
- Create: `vectora/ta/rating.py`
- Test: `tests/ta/test_rating.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/ta/test_rating.py
import polars as pl

from vectora.ta import rating


def _row(**kw):
    base = dict(macd=0.0, macd_signal=0.0, macd_hist=0.0, rsi14=50.0,
                bb_pos=0.5, bb_width=0.1, ma_fast=100.0, ma_slow=100.0,
                ma_cross_up=0, ma_cross_dn=0, st_dir=0, candle_bull=0,
                candle_bear=0, close=100.0)
    return {**base, **kw}


def test_all_bullish_gives_strong_buy():
    r = rating.score_row(_row(macd_hist=1.0, macd=1.0, rsi14=58,
                              ma_fast=110, ma_slow=100, ma_cross_up=1,
                              st_dir=1, candle_bull=1, bb_pos=0.7))
    assert r["band"] == "Strong Buy"
    assert r["score"] > 0
    assert len(r["votes"]) == 6            # one per indicator family


def test_all_bearish_gives_strong_sell():
    r = rating.score_row(_row(macd_hist=-1.0, macd=-1.0, rsi14=42,
                              ma_fast=90, ma_slow=100, ma_cross_dn=1,
                              st_dir=-1, candle_bear=1, bb_pos=0.3))
    assert r["band"] == "Strong Sell"
    assert r["score"] < 0


def test_neutral_gives_hold():
    assert rating.score_row(_row())["band"] == "Hold"


def test_overbought_rsi_counts_bearish_not_bullish():
    """RSI > 70 is exhaustion, not strength — the classic novice error."""
    hot = rating.score_row(_row(rsi14=82))
    rsi_vote = next(v for v in hot["votes"] if v["indicator"] == "RSI")
    assert rsi_vote["vote"] < 0
    assert "overbought" in rsi_vote["reason"].lower()


def test_votes_carry_human_readable_reasons():
    r = rating.score_row(_row(macd_hist=0.8, macd=0.5, st_dir=1))
    for v in r["votes"]:
        assert set(v) == {"indicator", "vote", "reason"}
        assert -2 <= v["vote"] <= 2
        assert len(v["reason"]) > 12
    macd = next(v for v in r["votes"] if v["indicator"] == "MACD")
    assert "signal line" in macd["reason"].lower()


def test_bands_are_ordered_and_exhaustive():
    scores = [rating.band_for(s) for s in range(-8, 9)]
    assert scores[0] == "Strong Sell" and scores[-1] == "Strong Buy"
    assert set(scores) == {"Strong Sell", "Sell", "Hold", "Buy", "Strong Buy"}
    # monotone: never improves as score falls
    order = ["Strong Sell", "Sell", "Hold", "Buy", "Strong Buy"]
    idx = [order.index(b) for b in scores]
    assert idx == sorted(idx)


def test_rate_frame_adds_columns():
    df = pl.DataFrame([_row(macd_hist=1.0, st_dir=1) | {"symbol": "A"},
                       _row(macd_hist=-1.0, st_dir=-1) | {"symbol": "B"}])
    out = rating.rate_frame(df)
    assert {"ta_score", "ta_band", "ta_votes"} <= set(out.columns)
    assert out.filter(pl.col("symbol") == "A")["ta_score"][0] > 0
```

- [ ] **Step 2: Run to verify failure** — FAIL.

- [ ] **Step 3: Implement**

Append to `SCHEMA` in `vectora/db.py`:

```sql
CREATE TABLE IF NOT EXISTS ta_ratings (
    date DATE, symbol TEXT, score INTEGER, band TEXT,
    votes TEXT,                     -- JSON list of {indicator, vote, reason}
    rsi DOUBLE, macd_hist DOUBLE, bb_pos DOUBLE, st_dir INTEGER,
    PRIMARY KEY (date, symbol)
);
CREATE TABLE IF NOT EXISTS ta_band_stats (
    band TEXT, horizon INTEGER, n INTEGER,
    hit_rate DOUBLE, base_rate DOUBLE, mean_fwd DOUBLE,
    computed_at TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (band, horizon)
);
```

Create `vectora/ta/rating.py`:

```python
"""Six indicator families -> votes -> 5-band technical posture.

Each family returns a vote in [-2, +2] with a plain-English reason; the
sum maps to a band. This is a MECHANICAL SUMMARY of indicator states, not
a forecast and not advice — vectora/ta/validate.py measures what each band
historically did, and the UI always shows that base rate alongside.

RSI is scored the way practitioners actually read it: mid-range strength
is constructive, but >70 is exhaustion (bearish) and <30 is washed-out
(bullish) — the opposite of the naive 'high RSI = strong' reading.
"""
import json

import polars as pl

BANDS = ["Strong Sell", "Sell", "Hold", "Buy", "Strong Buy"]
# score thresholds: <=-5 SS, -4..-2 S, -1..1 H, 2..4 B, >=5 SB
_CUTS = [(-99, -5, "Strong Sell"), (-5, -1, "Sell"), (-1, 2, "Hold"),
         (2, 5, "Buy"), (5, 99, "Strong Buy")]


def band_for(score: int) -> str:
    for lo, hi, name in _CUTS:
        if lo <= score < hi if name != "Strong Sell" else score < hi:
            return name
    return "Strong Buy"


def _macd(r) -> dict:
    h = r.get("macd_hist") or 0.0
    if h > 0:
        v, why = (2 if (r.get("macd") or 0) > 0 else 1), \
            "MACD is above its signal line and the histogram is positive"
    elif h < 0:
        v, why = (-2 if (r.get("macd") or 0) < 0 else -1), \
            "MACD is below its signal line and the histogram is negative"
    else:
        v, why = 0, "MACD is sitting on its signal line"
    return {"indicator": "MACD", "vote": v, "reason": why}


def _rsi(r) -> dict:
    x = r.get("rsi14")
    if x is None:
        return {"indicator": "RSI", "vote": 0, "reason": "RSI not yet available"}
    if x >= 70:
        return {"indicator": "RSI", "vote": -1,
                "reason": f"RSI {x:.0f} is overbought — stretched, prone to mean reversion"}
    if x <= 30:
        return {"indicator": "RSI", "vote": 1,
                "reason": f"RSI {x:.0f} is oversold — washed out, prone to a bounce"}
    if x >= 55:
        return {"indicator": "RSI", "vote": 1,
                "reason": f"RSI {x:.0f} shows constructive momentum without being stretched"}
    if x <= 45:
        return {"indicator": "RSI", "vote": -1,
                "reason": f"RSI {x:.0f} shows fading momentum"}
    return {"indicator": "RSI", "vote": 0, "reason": f"RSI {x:.0f} is neutral"}


def _bollinger(r) -> dict:
    p = r.get("bb_pos")
    if p is None:
        return {"indicator": "Bollinger", "vote": 0,
                "reason": "Bollinger bands not yet available"}
    if p > 1:
        return {"indicator": "Bollinger", "vote": -1,
                "reason": "price closed above the upper band — extended"}
    if p < 0:
        return {"indicator": "Bollinger", "vote": 1,
                "reason": "price closed below the lower band — capitulation zone"}
    if p > 0.65:
        return {"indicator": "Bollinger", "vote": 1,
                "reason": "price is riding the upper half of the band"}
    if p < 0.35:
        return {"indicator": "Bollinger", "vote": -1,
                "reason": "price is pinned to the lower half of the band"}
    return {"indicator": "Bollinger", "vote": 0,
            "reason": "price is mid-band, no stretch either way"}


def _ma(r) -> dict:
    f, s = r.get("ma_fast"), r.get("ma_slow")
    if f is None or s is None:
        return {"indicator": "MA cross", "vote": 0,
                "reason": "moving averages not yet available"}
    if r.get("ma_cross_up"):
        return {"indicator": "MA cross", "vote": 2,
                "reason": "the 20-day average has just crossed above the 50-day (golden cross)"}
    if r.get("ma_cross_dn"):
        return {"indicator": "MA cross", "vote": -2,
                "reason": "the 20-day average has just crossed below the 50-day (death cross)"}
    if f > s:
        return {"indicator": "MA cross", "vote": 1,
                "reason": "the 20-day average holds above the 50-day"}
    if f < s:
        return {"indicator": "MA cross", "vote": -1,
                "reason": "the 20-day average sits below the 50-day"}
    return {"indicator": "MA cross", "vote": 0, "reason": "moving averages are entwined"}


def _supertrend(r) -> dict:
    d = r.get("st_dir") or 0
    if d > 0:
        return {"indicator": "SuperTrend", "vote": 2,
                "reason": "SuperTrend is in an up-phase; its stop sits below price"}
    if d < 0:
        return {"indicator": "SuperTrend", "vote": -2,
                "reason": "SuperTrend is in a down-phase; its stop sits above price"}
    return {"indicator": "SuperTrend", "vote": 0,
            "reason": "SuperTrend has not established a direction"}


def _candle(r) -> dict:
    if r.get("candle_bull"):
        return {"indicator": "Candlestick", "vote": 1,
                "reason": "a bullish reversal candle printed (engulfing or hammer)"}
    if r.get("candle_bear"):
        return {"indicator": "Candlestick", "vote": -1,
                "reason": "a bearish reversal candle printed (engulfing or shooting star)"}
    return {"indicator": "Candlestick", "vote": 0,
            "reason": "no notable candlestick pattern"}


_FAMILIES = (_macd, _rsi, _bollinger, _ma, _supertrend, _candle)


def score_row(r: dict) -> dict:
    votes = [f(r) for f in _FAMILIES]
    score = sum(v["vote"] for v in votes)
    return {"score": score, "band": band_for(score), "votes": votes}


def rate_frame(df: pl.DataFrame) -> pl.DataFrame:
    results = [score_row(r) for r in df.iter_rows(named=True)]
    return df.with_columns(
        pl.Series("ta_score", [x["score"] for x in results]),
        pl.Series("ta_band", [x["band"] for x in results]),
        pl.Series("ta_votes", [json.dumps(x["votes"]) for x in results]))
```

Careful with `band_for`: the inline conditional in the loop is deliberately awkward; if the ordering test fails, rewrite it as a plain ladder of `if score <= -5 … elif score < -1 …` — the TEST is the contract (bands must be monotone in score and cover -8..+8).

- [ ] **Step 4: Run tests** — 7 passed; fast suite; ruff.

- [ ] **Step 5: Commit**

```bash
git add vectora/db.py vectora/ta/rating.py tests/ta/test_rating.py
git commit -m "feat: 5-band technical rating with per-indicator plain-English rationale"
```

---

### Task 3: Historical validation — what each band actually did

**Files:**
- Create: `vectora/ta/validate.py`
- Test: `tests/ta/test_validate.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/ta/test_validate.py
import datetime as dt

import numpy as np
import polars as pl

from vectora import db as vdb
from vectora.ta import validate


def _seed(con, n_days=400, n_syms=25, seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    d0 = dt.date(2024, 1, 1)
    px = {f"S{i:02d}": 100.0 for i in range(n_syms)}
    for day in range(n_days):
        d = d0 + dt.timedelta(days=day)
        for sym in px:
            px[sym] *= float(np.exp(rng.normal(0.0004, 0.02)))
            p = round(max(px[sym], 1.0), 2)
            rows.append(dict(symbol=sym, date=d, open=p, high=p * 1.02,
                             low=p * 0.98, close=p, ltp=p, ycp=p, trades=30,
                             value_mn=4.0, volume=9000, source="dse_eod"))
    df = pl.DataFrame(rows)  # noqa: F841
    con.execute("INSERT INTO prices_raw SELECT * FROM df")


def test_validation_produces_band_stats(test_db):
    _seed(test_db)
    result = validate.run(test_db, horizons=(5, 10), threshold=0.05)
    assert result["rows_scored"] > 1000
    stats = test_db.execute(
        "SELECT band, horizon, n, hit_rate, base_rate FROM ta_band_stats"
    ).fetchall()
    assert stats
    for band, h, n, hit, base in stats:
        assert band in ("Strong Sell", "Sell", "Hold", "Buy", "Strong Buy")
        assert h in (5, 10)
        assert n > 0
        assert 0.0 <= hit <= 1.0
        assert 0.0 <= base <= 1.0


def test_base_rate_is_shared_across_bands(test_db):
    _seed(test_db)
    validate.run(test_db, horizons=(10,), threshold=0.05)
    bases = {r[0] for r in test_db.execute(
        "SELECT DISTINCT base_rate FROM ta_band_stats WHERE horizon = 10"
    ).fetchall()}
    assert len(bases) == 1      # one market-wide base rate per horizon


def test_edge_is_reported_not_assumed(test_db):
    """On random-walk data no band should show a large edge; the function
    must report that honestly rather than manufacturing one."""
    _seed(test_db)
    validate.run(test_db, horizons=(10,), threshold=0.05)
    rows = test_db.execute(
        "SELECT band, hit_rate, base_rate FROM ta_band_stats WHERE horizon=10"
    ).fetchall()
    for _band, hit, base in rows:
        assert abs(hit - base) < 0.25
```

- [ ] **Step 2: Run to verify failure** — FAIL.

- [ ] **Step 3: Implement**

```python
# vectora/ta/validate.py
"""Measure what each technical band HISTORICALLY did (spec: Phase 6B).

Replays the rating across all available history, joins each rating to the
forward outcome the models are trained on (did price gain X% within H
trading rows), and records hit rate per band beside the market-wide base
rate. This is the honesty layer: a band is only worth showing if it beats
the base rate, and the dashboard prints both numbers.
"""
from vectora import db as vdb
from vectora import labels as lab
from vectora.features import base
from vectora.ta import indicators, rating


def run(con, horizons=(5, 10), threshold: float = 0.05,
        min_history: int = 60) -> dict:
    panel = base.load_panel(con).select(
        ["symbol", "date", "open", "high", "low", "close"])
    ind = indicators.add_all(panel)
    # a rating is only meaningful once the slowest indicator has warmed up
    ind = ind.filter(ind["ma_slow"].is_not_null())
    rated = rating.rate_frame(ind)

    labeled = lab.make_labels(
        panel, thresholds=(threshold,), horizons=tuple(horizons),
        continuous=True)
    pct = round(threshold * 100)

    out_rows = []
    for h in horizons:
        col = f"y_g{pct}_h{h}"
        joined = (rated.select(["symbol", "date", "ta_band"])
                  .join(labeled.select(["symbol", "date", col]),
                        on=["symbol", "date"], how="inner")
                  .filter(labeled[col].is_not_null()
                          if False else __import__("polars").col(col).is_not_null()))
        if joined.height == 0:
            continue
        base_rate = float(joined[col].mean())
        for band in ("Strong Sell", "Sell", "Hold", "Buy", "Strong Buy"):
            sub = joined.filter(__import__("polars").col("ta_band") == band)
            if sub.height == 0:
                continue
            out_rows.append({
                "band": band, "horizon": h, "n": int(sub.height),
                "hit_rate": float(sub[col].mean()),
                "base_rate": base_rate,
                "mean_fwd": float(sub[col].mean()) - base_rate,
            })
    if out_rows:
        vdb.upsert(con, "ta_band_stats", out_rows)
    return {"rows_scored": rated.height, "bands": len(out_rows)}
```

Cleanup required before committing: the `__import__("polars")` calls and the `if False else` are sketch artifacts — add `import polars as pl` at the top and use `pl.col(...)` normally. They are left visible here only so the intent (filter on the label column, filter by band) is unambiguous.

- [ ] **Step 4: Run tests** — 3 passed; fast suite; ruff.

- [ ] **Step 5: Commit**

```bash
git add vectora/ta/validate.py tests/ta/test_validate.py
git commit -m "feat: historical validation of technical bands against forward outcomes"
```

---

### Task 4: Screener — rate the board, rank it, persist it

**Files:**
- Create: `vectora/ta/screener.py`, `vectora/config/watchlist.yaml`
- Modify: `vectora/__main__.py` (add `ta` stage)
- Test: `tests/ta/test_screener.py`

- [ ] **Step 1: Write `vectora/config/watchlist.yaml`** — the verified mapping:

```yaml
# Verified against the live symbol master 2026-07-31.
# Three names resolved to the WRONG company by fuzzy match and are fixed here:
#   EasternLub -> EASTRNLUB (not EASTERNINS, an insurer)
#   Mercantile Ins -> MERCINS (not MERCANBANK, a bank)
#   Meghna Petroleum -> MPETROLEUM (not MEGHNAPET, a plastics maker)
groups:
  Govt/SOE: [TITASGAS, DESCO, POWERGRID, PADMAOIL, MPETROLEUM, BSC, BSCPLC,
             EASTRNLUB, ECABLES, NTLTUBES, ICB]
  Insurance: [EIL, ICICL, MERCINS, NORTHRNINS, PIONEERINS, PURABIGEN,
              RUPALIINS, SONARBAINS, STANDARINS]
  Life insurance: [CLICL, DELTALIFE, POPULARLIF, PRIMELIFE, SANDHANINS]
  Mutual funds: [1STPRIMFMF, ABB1STMF, AIBL1STIMF, CAPITECGBF, CAPMBDBLMF,
                 DBH1STMF, EBLNRBMF, FBFIF, GLDNJMF, GREENDELMF, PF1STMF]
  Textile: [ARGONDENIM, DSSL, ENVOYTEX, ETL, FEKDIL, HRTEX, MATINSPINN,
            MHSML, PTL, SHEPHERD, SIMTEX, SAIHAMCOT, SAIHAMTEX, KDSALTD]
  Pharma: [ACMELAB, ACMEPL, BEACONPHAR, BXPHARMA, TECHNODRUG]
  IT: [BDCOM, EGEN, GENEXIL, ITC]
  Cement & ceramics: [CONFIDCEM, LHB, RAKCERAMIC, MONNOCERA, SPCERAMICS]
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/ta/test_screener.py
import datetime as dt

import numpy as np
import polars as pl

from vectora import db as vdb
from vectora.ta import screener


def _seed(con, n_days=200, n_syms=12, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    d0 = dt.date(2025, 6, 1)
    px = {f"S{i:02d}": 100.0 for i in range(n_syms)}
    for day in range(n_days):
        d = d0 + dt.timedelta(days=day)
        for sym in px:
            drift = 0.01 if sym == "S00" else (-0.01 if sym == "S01" else 0.0)
            px[sym] *= float(np.exp(rng.normal(drift, 0.012)))
            p = round(max(px[sym], 1.0), 2)
            rows.append(dict(symbol=sym, date=d, open=p, high=p * 1.01,
                             low=p * 0.99, close=p, ltp=p, ycp=p, trades=30,
                             value_mn=5.0, volume=9000, source="dse_eod"))
    df = pl.DataFrame(rows)  # noqa: F841
    con.execute("INSERT INTO prices_raw SELECT * FROM df")
    vdb.upsert(con, "symbols", [
        dict(symbol=s, name=None, sector="Bank", instrument_type="Equity",
             category="A", listing_status="active", first_seen="2020-01-01",
             last_seen="2026-12-31") for s in px])
    return (d0 + dt.timedelta(days=n_days - 1)).isoformat()


def test_screener_rates_every_symbol_and_persists(test_db):
    last = _seed(test_db)
    result = screener.run(test_db, date_str=last)
    assert result["rated"] == 12
    rows = test_db.execute(
        "SELECT symbol, band, score, votes FROM ta_ratings WHERE date = ?",
        [last]).fetchall()
    assert len(rows) == 12
    import json
    for _s, band, _sc, votes in rows:
        assert band in ("Strong Sell", "Sell", "Hold", "Buy", "Strong Buy")
        assert len(json.loads(votes)) == 6


def test_uptrend_outranks_downtrend(test_db):
    last = _seed(test_db)
    screener.run(test_db, date_str=last)
    up = test_db.execute(
        "SELECT score FROM ta_ratings WHERE date=? AND symbol='S00'",
        [last]).fetchone()[0]
    dn = test_db.execute(
        "SELECT score FROM ta_ratings WHERE date=? AND symbol='S01'",
        [last]).fetchone()[0]
    assert up > dn


def test_rerun_is_idempotent(test_db):
    last = _seed(test_db)
    screener.run(test_db, date_str=last)
    screener.run(test_db, date_str=last)
    assert test_db.execute(
        "SELECT count(*) FROM ta_ratings WHERE date = ?", [last]
    ).fetchone()[0] == 12


def test_watchlist_loads_and_maps():
    groups = screener.load_watchlist()
    assert "Pharma" in groups
    assert "BXPHARMA" in groups["Pharma"]
    assert "MPETROLEUM" in groups["Govt/SOE"]
    # the three corrected tickers must never regress to the wrong company
    flat = {s for v in groups.values() for s in v}
    assert "EASTRNLUB" in flat and "EASTERNINS" not in flat
    assert "MERCINS" in flat and "MERCANBANK" not in flat
    assert "MEGHNAPET" not in flat
```

- [ ] **Step 3: Run to verify failure** — FAIL.

- [ ] **Step 4: Implement `vectora/ta/screener.py`**

```python
"""Rate the whole board for one date and persist to ta_ratings."""
import json
from pathlib import Path

import polars as pl
import yaml

from vectora import db as vdb
from vectora.features import base
from vectora.ta import indicators, rating

WATCHLIST_PATH = (Path(__file__).resolve().parent.parent / "config"
                  / "watchlist.yaml")


def load_watchlist(path: Path = WATCHLIST_PATH) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))["groups"]


def run(con, date_str: str | None = None) -> dict:
    panel = base.load_panel(con).select(
        ["symbol", "date", "open", "high", "low", "close"])
    ind = indicators.add_all(panel)
    run_date = date_str or str(ind["date"].max())
    today = ind.filter(pl.col("date") == pl.lit(run_date).str.to_date())
    if today.height == 0:
        return {"date": run_date, "rated": 0}
    rated = rating.rate_frame(today)
    rows = [{
        "date": run_date, "symbol": r["symbol"], "score": int(r["ta_score"]),
        "band": r["ta_band"], "votes": r["ta_votes"],
        "rsi": r.get("rsi14"), "macd_hist": r.get("macd_hist"),
        "bb_pos": r.get("bb_pos"), "st_dir": int(r.get("st_dir") or 0),
    } for r in rated.iter_rows(named=True)]
    vdb.upsert(con, "ta_ratings", rows)
    bands = {}
    for r in rows:
        bands[r["band"]] = bands.get(r["band"], 0) + 1
    return {"date": run_date, "rated": len(rows), "bands": bands,
            "watchlist_groups": len(load_watchlist())}


def ranked(con, date_str: str, symbols: list[str] | None = None,
           limit: int = 50) -> list[dict]:
    q = ("SELECT symbol, score, band, votes, rsi, macd_hist, bb_pos, st_dir "
         "FROM ta_ratings WHERE date = ?")
    params: list = [date_str]
    if symbols:
        q += " AND symbol IN (" + ",".join("?" * len(symbols)) + ")"
        params += symbols
    q += " ORDER BY score DESC, symbol LIMIT ?"
    params.append(limit)
    return [{"symbol": s, "score": sc, "band": b, "votes": json.loads(v),
             "rsi": rsi, "macd_hist": mh, "bb_pos": bp, "st_dir": sd}
            for s, sc, b, v, rsi, mh, bp, sd in con.execute(q, params).fetchall()]
```

Add the `ta` CLI stage in `vectora/__main__.py` (stage choices gain `"ta"`):

```python
    if args.command == "run" and args.stage == "ta":
        from vectora import db as vdb
        from vectora.settings import DB_PATH
        from vectora.ta import screener, validate
        con = vdb.connect(DB_PATH)
        try:
            vdb.init_schema(con)
            result = screener.run(con, date_str=args.date)
            result["validation"] = validate.run(con)
        finally:
            con.close()
        print(json.dumps(result, indent=1, default=str))
        return 0
```

- [ ] **Step 5: Run tests** — 4 passed; fast suite; ruff.

- [ ] **Step 6: Commit**

```bash
git add vectora/ta/screener.py vectora/config/watchlist.yaml vectora/__main__.py tests/ta/test_screener.py
git commit -m "feat: board-wide technical screener with verified watchlist mapping"
```

---

### Task 5: Screener page + real run + merge

**Files:**
- Create: `vectora/ta/page.py`
- Modify: `vectora/dashboard.py` (link), `.github/workflows/eod-pipeline.yml`
- Test: `tests/ta/test_page.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/ta/test_page.py
import json

from vectora import db as vdb
from vectora.ta import page


def _seed(con, d="2026-07-30"):
    votes = json.dumps([
        {"indicator": "MACD", "vote": 2, "reason": "MACD is above its signal line"},
        {"indicator": "RSI", "vote": -1, "reason": "RSI 78 is overbought"},
    ])
    vdb.upsert(con, "ta_ratings", [
        dict(date=d, symbol="AAA", score=6, band="Strong Buy", votes=votes,
             rsi=61.0, macd_hist=0.4, bb_pos=0.8, st_dir=1),
        dict(date=d, symbol="BBB", score=-6, band="Strong Sell", votes=votes,
             rsi=28.0, macd_hist=-0.3, bb_pos=0.1, st_dir=-1)])
    vdb.upsert(con, "ta_band_stats", [
        dict(band="Strong Buy", horizon=10, n=8400, hit_rate=0.34,
             base_rate=0.28, mean_fwd=0.06),
        dict(band="Strong Sell", horizon=10, n=7100, hit_rate=0.21,
             base_rate=0.28, mean_fwd=-0.07)])
    vdb.upsert(con, "symbols", [
        dict(symbol=s, name=None, sector="Bank", instrument_type="Equity",
             category="A", listing_status="active", first_seen="2020-01-01",
             last_seen="2026-12-31") for s in ("AAA", "BBB")])
    return d


def test_page_renders_bands_rationale_and_evidence(test_db, tmp_path):
    d = _seed(test_db)
    out = page.build(test_db, d, out_path=tmp_path / "screener.html")
    html = out.read_text(encoding="utf-8")
    assert "Strong Buy" in html and "AAA" in html
    # drop-down rationale is a real <details> element per symbol
    assert html.count("<details") >= 2
    assert "overbought" in html
    # the honesty layer: measured hit rate vs base rate must be present
    assert "34%" in html and "28%" in html
    assert "not investment advice" in html.lower()
    assert "mechanical" in html.lower()


def test_page_handles_empty_day(test_db, tmp_path):
    out = page.build(test_db, "2026-07-30", out_path=tmp_path / "s.html")
    assert "No technical ratings" in out.read_text(encoding="utf-8")
```

- [ ] **Step 2: Run to verify failure** — FAIL.

- [ ] **Step 3: Implement `vectora/ta/page.py`**

Render a standalone page reusing the dashboard's token CSS (import `_TEMPLATE`-style variables by copying the `<style>` block from `vectora/dashboard.py` via a shared constant — extract `dashboard._STYLE` into a module-level constant in dashboard.py and import it here, so both pages stay visually identical). Structure:

1. Header + "as of" date, link back to `index.html`.
2. An evidence strip: for each band, `hit_rate` vs `base_rate` and n at h10, rendered as a small table with the sentence *"A band is only worth acting on if its hit rate beats the base rate. These numbers are measured on DSE history, not asserted."*
3. Watchlist groups: one section per group from `load_watchlist()`, each a table of symbol / band pill / score / RSI / MACD / trend, sorted by score.
4. Whole-board top 25 and bottom 25.
5. Per symbol, a `<details><summary>` drop-down listing every indicator vote with its reason, so a reader can see exactly why the band came out where it did.
6. Footer: mechanical-summary disclaimer, not investment advice, plus a pointer that Vectora's calibrated probability is the separately-validated number.

Band pill colours reuse the existing semantic tokens: Strong Buy `--good`, Buy `--sig`, Hold `--muted`, Sell `--warning`, Strong Sell `--critical`.

- [ ] **Step 4: Wire the pipeline** — in `.github/workflows/eod-pipeline.yml` insert before the Dashboard step:

```yaml
      - name: Technical ratings
        continue-on-error: true
        run: uv run python -m vectora run ta
```

and add `docs/dashboard/screener.html` to the committed paths (already covered by `docs/dashboard`).

- [ ] **Step 5: Real run**

```bash
uv run python -m vectora run ta
uv run python -m vectora run dashboard
```

Report: band distribution across the board, the measured hit-rate-vs-base-rate table, and the watchlist standings. **Judgment gate:** if no band beats the base rate by more than ~2 percentage points, say so plainly in the summary — that is a real finding about technical analysis on the DSE, and the page must show it rather than bury it.

- [ ] **Step 6: Fast suite + ruff, commit, merge, push**

```bash
uv run pytest -m "not slow" && uv run ruff check .
git add -A
git commit -m "feat: technical screener page with measured band evidence"
git checkout main && git pull
git merge --no-ff phase-6b-ta -m "Merge phase-6b: technical rating engine"
git push
```

---

## Execution notes

- Order 1→5. Expected fast suite ≈ 230 tests.
- The lookahead test in Task 1 and the base-rate honesty test in Task 3 are the two that matter most; never weaken either.
- Fundamentals (P/E, NAV, dividend yield) are NOT in this plan — the `fundamentals` table was specced but never built and the company parser does not extract EPS/NAV. That is the natural follow-on (Phase 6C) and the biggest remaining gap for portfolio work.
