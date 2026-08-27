"""TradingView-parity gauge tests: 26 components, two gauges, one summary."""
import datetime as dt

import polars as pl

from vectora.ta import gauges, indicators


def _row(**kw) -> dict:
    base = {"close": 100.0}
    base.update(kw)
    return base


def test_ma_gauge_matches_the_client_spec():
    """Pruned 2026-08-26 to the components the client reads."""
    names = [v["indicator"] for v in gauges.ma_votes(_row())]
    assert names == [
        "SMA(10)", "SMA(20)", "SMA(50)", "SMA(100)", "SMA(200)",
        "EMA(10)", "EMA(20)", "EMA(50)", "EMA(100)", "EMA(200)",
        "VWMA(10)", "VWMA(20)", "Ichimoku Cloud"]


def test_osc_gauge_matches_the_client_spec():
    names = [v["indicator"] for v in gauges.osc_votes(_row())]
    assert names == [
        "RSI(14)", "CCI(10)", "CCI(20)", "ADX(14)", "Awesome Oscillator",
        "MACD(12,26)", "Stochastic RSI(14)", "Ultimate Oscillator"]


def test_dropped_components_are_really_gone():
    """A component left computing but unread is dead weight that still
    costs a column in every stored vote payload."""
    names = ({v["indicator"] for v in gauges.ma_votes(_row())}
             | {v["indicator"] for v in gauges.osc_votes(_row())})
    for dropped in ("SMA(30)", "EMA(30)", "Hull MA(9)", "Stochastic(14,3,3)",
                    "Momentum(10)", "Williams %R(14)", "Bull Bear Power"):
        assert dropped not in names, dropped


def test_price_above_every_average_is_strong_buy_on_the_ma_gauge():
    r = _row(**{f"{k}{n}": 50.0 for k in ("sma", "ema")
                for n in gauges.MA_PERIODS})
    r |= {"vwma10": 50.0, "vwma20": 50.0, "ichi_a": 40.0, "ichi_b": 45.0}
    g = gauges.gauge(gauges.ma_votes(r))
    assert g["buy"] == 13 and g["sell"] == 0
    assert g["band"] == "Strong Buy"


def test_price_below_every_average_is_strong_sell():
    r = _row(**{f"{k}{n}": 150.0 for k in ("sma", "ema")
                for n in gauges.MA_PERIODS})
    r |= {"vwma10": 150.0, "vwma20": 150.0, "ichi_a": 160.0, "ichi_b": 155.0}
    g = gauges.gauge(gauges.ma_votes(r))
    assert g["sell"] == 13 and g["band"] == "Strong Sell"


def test_missing_data_votes_neutral_never_crashes():
    g = gauges.gauge(gauges.osc_votes(_row()))
    assert g["buy"] == 0 and g["sell"] == 0 and g["band"] == "Hold"


def test_oscillators_are_contrarian_at_extremes():
    """Oversold AND turning up is the buy vote — not merely 'high reading'."""
    up = gauges.osc_votes(_row(rsi14=25.0, prev_rsi14=22.0))
    assert next(v for v in up if v["indicator"] == "RSI(14)")["vote"] == 1
    # oversold but still falling is NOT yet a buy
    still = gauges.osc_votes(_row(rsi14=25.0, prev_rsi14=28.0))
    assert next(v for v in still if v["indicator"] == "RSI(14)")["vote"] == 0
    down = gauges.osc_votes(_row(rsi14=78.0, prev_rsi14=80.0))
    assert next(v for v in down if v["indicator"] == "RSI(14)")["vote"] == -1


def test_adx_below_twenty_refuses_to_vote():
    v = next(x for x in gauges.osc_votes(
        _row(adx14=12.0, di_plus=30.0, di_minus=5.0))
        if x["indicator"] == "ADX(14)")
    assert v["vote"] == 0 and "no trend" in v["reason"]


def test_ichimoku_inside_the_cloud_is_neutral():
    v = next(x for x in gauges.ma_votes(_row(ichi_a=90.0, ichi_b=110.0))
             if x["indicator"] == "Ichimoku Cloud")
    assert v["vote"] == 0


def test_band_cutpoints_match_tradingview():
    assert gauges.band_for(0.6) == "Strong Buy"
    assert gauges.band_for(0.2) == "Buy"
    assert gauges.band_for(0.0) == "Hold"
    assert gauges.band_for(-0.2) == "Sell"
    assert gauges.band_for(-0.6) == "Strong Sell"


def test_summary_weights_the_two_gauges_equally():
    """15 bullish averages must not drown out 11 bearish oscillators."""
    r = _row(**{f"{k}{n}": 50.0 for k in ("sma", "ema")
                for n in gauges.MA_PERIODS})
    r |= {"vwma10": 50.0, "vwma20": 50.0, "ichi_a": 40.0, "ichi_b": 45.0}
    out = gauges.rate_row(r)
    assert out["ma"]["mean"] == 1.0
    assert out["osc"]["mean"] == 0.0        # no oscillator inputs supplied
    assert abs(out["summary_mean"] - 0.5) < 1e-9
    assert out["summary_band"] == "Strong Buy"


def test_vote_frame_agrees_with_rate_row_on_real_shaped_data():
    """The fast vectorised path and the prose path must never disagree."""
    import numpy as np
    rng = np.random.default_rng(11)
    n = 220
    close = 100 + np.cumsum(rng.normal(0, 2, n))
    df = pl.DataFrame({
        "symbol": ["X"] * n,
        "date": pl.date_range(dt.date(2025, 1, 1), dt.date(2025, 1, 1)
                              + dt.timedelta(days=n - 1), eager=True),
        "open": close - rng.normal(0, 0.5, n),
        "high": close + np.abs(rng.normal(0, 1.5, n)),
        "low": close - np.abs(rng.normal(0, 1.5, n)),
        "close": close,
        "volume": rng.integers(1_000, 500_000, n).astype(float),
    })
    full = indicators.add_tradingview_set(indicators.add_all(df))
    fast = gauges.vote_frame(full)
    slow = gauges.rate_frame(full)
    for col in ("ma_mean", "osc_mean", "summary_mean"):
        assert np.allclose(fast[col].to_numpy(), slow[col].to_numpy()), col
    assert fast["summary_band"].to_list() == slow["summary_band"].to_list()


def test_rate_frame_adds_columns_and_prev_shift_is_per_symbol():
    df = pl.DataFrame({
        "symbol": ["A", "A", "B", "B"],
        "date": ["2026-01-01", "2026-01-02"] * 2,
        "close": [100.0, 101.0, 10.0, 11.0],
        "rsi14": [22.0, 25.0, 80.0, 78.0],
    })
    out = gauges.rate_frame(df)
    for col in ("ma_mean", "ma_band", "osc_mean", "osc_band",
                "summary_mean", "summary_band", "gauge_votes"):
        assert col in out.columns
    # row 2 (symbol B, first row) must not inherit symbol A's prior RSI
    assert out["osc_mean"][2] == 0.0


def test_abstaining_at_an_extreme_says_so():
    """A reading of 8 is not 'mid-range'. A reader who checks the number
    against the prose and finds it wrong stops trusting the whole page."""
    v = next(x for x in gauges.osc_votes(_row(stochrsi=8.0, prev_stochrsi=12.0))
             if x["indicator"] == "Stochastic RSI(14)")
    assert v["vote"] == 0
    assert "oversold but still falling" in v["reason"]

    v = next(x for x in gauges.osc_votes(_row(cci20=180.0, prev_cci20=150.0))
             if x["indicator"] == "CCI(20)")
    assert v["vote"] == 0
    assert "overbought but still rising" in v["reason"]

    v = next(x for x in gauges.osc_votes(_row(cci10=40.0, prev_cci10=20.0))
             if x["indicator"] == "CCI(10)")
    assert "mid-range" in v["reason"]
