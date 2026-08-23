"""Backtest correctness — mostly pinning the ways a backtest can lie.

Every test here exists because the opposite behaviour would silently
overstate profitability: entering at a price you could not have traded at,
assuming the target beat the stop inside a bar, ignoring the cost of being
large in a thin stock, or taking more positions than capital allows.
"""
import datetime as dt

import polars as pl

from vectora.backtest import costs as C
from vectora.backtest import metrics, simulator
from vectora.backtest.simulator import Portfolio, Rules


def _panel(bars, symbol="AAA", adv=100.0):
    """bars: list of (open, high, low, close) on consecutive days."""
    d0 = dt.date(2026, 1, 1)
    return pl.DataFrame({
        "symbol": [symbol] * len(bars),
        "date": [d0 + dt.timedelta(days=i) for i in range(len(bars))],
        "open": [b[0] for b in bars], "high": [b[1] for b in bars],
        "low": [b[2] for b in bars], "close": [b[3] for b in bars],
        "adv_mn": [adv] * len(bars),
    })


def _entry(day=0, symbol="AAA", score=1.0):
    return pl.DataFrame({"symbol": [symbol],
                         "date": [dt.date(2026, 1, 1) + dt.timedelta(days=day)],
                         "score": [score]})


def test_entry_is_the_next_open_not_the_signal_close():
    """You cannot trade at the close you are predicting from."""
    panel = _panel([(100, 100, 100, 100), (50, 60, 50, 55),
                    (55, 55, 55, 55), (55, 55, 55, 55)])
    r = simulator.run(panel, _entry(0), Rules(max_days=2), C.ZERO)
    assert r["trades"][0]["entry_px"] == 50      # day 1 open, not day 0 close


def test_close_entry_is_available_but_not_the_default():
    panel = _panel([(100, 100, 100, 100), (50, 60, 50, 55),
                    (55, 55, 55, 55)])
    r = simulator.run(panel, _entry(0),
                      Rules(max_days=2, entry="close"), C.ZERO)
    assert r["trades"][0]["entry_px"] == 100


def test_target_is_taken_when_the_bar_reaches_it():
    panel = _panel([(100, 100, 100, 100), (100, 106, 99, 105),
                    (105, 105, 105, 105)])
    t = simulator.run(panel, _entry(0), Rules(target_pct=0.05, max_days=2),
                      C.ZERO)["trades"][0]
    assert t["reason"] == "target"
    assert abs(t["gross_ret"] - 0.05) < 1e-9


def test_stop_is_taken_when_the_bar_breaks_it():
    panel = _panel([(100, 100, 100, 100), (100, 101, 94, 95),
                    (95, 95, 95, 95)])
    t = simulator.run(panel, _entry(0),
                      Rules(target_pct=0.05, stop_pct=-0.05, max_days=2),
                      C.ZERO)["trades"][0]
    assert t["reason"] == "stop"
    assert abs(t["gross_ret"] + 0.05) < 1e-9


def test_ambiguous_bar_assumes_the_stop_hit_first():
    """A day that touches both is unorderable on daily data. Assuming the
    target won is the most common way a backtest overstates itself."""
    panel = _panel([(100, 100, 100, 100), (100, 110, 90, 100),
                    (100, 100, 100, 100)])
    rules = Rules(target_pct=0.05, stop_pct=-0.05, max_days=2)
    pess = simulator.run(panel, _entry(0), rules, C.ZERO)["trades"][0]
    assert pess["reason"] == "stop"
    optimistic = simulator.run(
        panel, _entry(0),
        Rules(target_pct=0.05, stop_pct=-0.05, max_days=2,
              pessimistic_intrabar=False), C.ZERO)["trades"][0]
    assert optimistic["reason"] == "target"
    assert pess["gross_ret"] < optimistic["gross_ret"]


def test_time_exit_uses_the_close_of_the_last_day():
    panel = _panel([(100, 100, 100, 100), (100, 101, 99, 100),
                    (100, 101, 99, 102), (102, 102, 102, 102)])
    t = simulator.run(panel, _entry(0),
                      Rules(target_pct=0.50, max_days=2), C.ZERO)["trades"][0]
    assert t["reason"] == "time"


def test_costs_reduce_the_net_return():
    panel = _panel([(100, 100, 100, 100), (100, 106, 99, 105),
                    (105, 105, 105, 105)])
    model = C.CostModel(commission=0.004, regulatory=0.0005,
                        half_spread=0.0015, impact_coef=0.0)
    t = simulator.run(panel, _entry(0), Rules(max_days=2), model)["trades"][0]
    assert abs(t["cost"] - 2 * 0.006) < 1e-9
    assert abs(t["net_ret"] - (0.05 - 0.012)) < 1e-9


def test_thin_stocks_cost_more_than_liquid_ones():
    """Impact is not optional on an exchange where a third of the board
    trades under 5 mn a day."""
    model = C.CostModel(impact_coef=0.10)
    liquid = model.side_cost(notional_mn=1.0, adv_mn=100.0)
    thin = model.side_cost(notional_mn=1.0, adv_mn=2.0)
    assert thin > liquid


def test_unknown_turnover_is_charged_the_cap_not_zero():
    """99% of the price history has no turnover column. Treating unknown as
    free would flatter every deep-history backtest."""
    model = C.CostModel()
    assert model.impact(1.0, None) == model.max_impact
    assert model.impact(1.0, 0.0) == model.max_impact


def test_position_cap_is_respected():
    """Signals are not free: taking one means declining another."""
    frames = [_panel([(100, 100, 100, 100), (100, 106, 99, 105),
                      (105, 105, 105, 105)], symbol=s)
              for s in ("AAA", "BBB", "CCC")]
    panel = pl.concat(frames)
    entries = pl.concat([_entry(0, s, score=i)
                         for i, s in enumerate(("AAA", "BBB", "CCC"))])
    r = simulator.run(panel, entries, Rules(max_days=2), C.ZERO,
                      Portfolio(max_positions=2))
    assert len(r["trades"]) == 2


def test_higher_scored_candidate_wins_a_contested_slot():
    frames = [_panel([(100, 100, 100, 100), (100, 106, 99, 105),
                      (105, 105, 105, 105)], symbol=s)
              for s in ("LOW", "HIGH")]
    panel = pl.concat(frames)
    entries = pl.concat([_entry(0, "LOW", score=0.1),
                         _entry(0, "HIGH", score=0.9)])
    r = simulator.run(panel, entries, Rules(max_days=2), C.ZERO,
                      Portfolio(max_positions=1))
    assert r["trades"][0]["symbol"] == "HIGH"


def test_expectancy_is_negative_when_wins_cap_and_losses_run():
    """The shape found live on 2026-08-20: most trades win, and the
    strategy still loses. Win rate cannot detect this; expectancy can."""
    trades = ([{"net_ret": 0.04, "cost": 0.01, "reason": "target"}] * 7
              + [{"net_ret": -0.20, "cost": 0.01, "reason": "time"}] * 3)
    m = metrics.summarize({
        "trades": trades, "start_capital": 100.0, "final_equity": 100.0,
        "equity_curve": [{"date": 1, "equity": 100.0}]})
    assert m["win_rate"] == 0.7
    assert m["expectancy"] < 0
    assert m["profit_factor"] < 1


def test_no_trades_summarizes_cleanly():
    m = metrics.summarize({"trades": [], "start_capital": 100.0,
                           "final_equity": 100.0, "equity_curve": []})
    assert m["trades"] == 0
    assert "no trades" in metrics.render("empty", m)
