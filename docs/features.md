# Feature registry

44 features. Every feature documents its economic reasoning (enforced by test).

| name | family | reasoning |
|---|---|---|
| `ret_1d` | momentum | one-day return is the base momentum unit and mean-reversion input on a retail-driven exchange |
| `ret_3d` | momentum | three-day compounded return captures the start of herding runs documented in DSE pump episodes |
| `ret_5d` | momentum | one-trading-week momentum aligns with the Sun-Thu week and weekly retail cycle |
| `ret_10d` | momentum | two-week momentum is where DSE herding historically peaks before reversal risk rises |
| `ret_21d` | momentum | one-month momentum is the classic cross-sectional momentum horizon in emerging markets |
| `ret_63d` | momentum | one-quarter momentum separates persistent trends from short speculative bursts |
| `rsi_14` | momentum | retail participants trade RSI signals making them partly self-fulfilling on the DSE |
| `dist_high_63d` | momentum | distance below the quarterly high measures breakout proximity which retail flows chase |
| `dist_low_63d` | momentum | distance above the quarterly low flags capitulation levels where reversals start |
| `vol_21d` | volatility | one-month realized volatility is the base risk scale for position sizing and labels |
| `vol_63d` | volatility | quarterly volatility anchors the regime a stock trades in versus its recent burst |
| `vol_ratio_21_63` | volatility | short-over-long volatility ratio detects fresh volatility expansion preceding large moves |
| `atr_14` | volatility | average true range in price units feeds stop distance and expected-move estimates |
| `range_pct_5d` | volatility | weekly average high-low range as percent of close measures intraday heat within the band |
| `limit_lock_21d` | volatility | count of near-circuit closes is the DSE-specific heat gauge since bands truncate raw volatility |
| `value_mn_med_21d` | liquidity | median daily traded value is the tradability floor input used in the universe filter |
| `amihud_21d` | liquidity | Amihud illiquidity prices the impact per taka traded which dominates execution risk on thin books |
| `zero_vol_21d` | liquidity | count of zero-volume days flags dormant names where any print can gap the price |
| `turnover_z_21d` | liquidity | value z-score highlights unusual money inflow relative to a name's own norm |
| `volume_z_21d` | liquidity | volume z-score is the classic accumulation signal preceding price in DSE pump patterns |
| `vol_ratio_5_21` | volume | five-over-twentyone-day volume ratio measures fresh participation buildup |
| `obv_slope_21d` | volume | on-balance-volume slope captures directional flow persistence beyond raw volume |
| `updown_vol_21d` | volume | up-day versus down-day volume split separates accumulation from distribution |
| `trades_z_21d` | volume | trade-count z-score proxies breadth of participation versus a few large prints |
| `vwap_dev_5d` | volume | deviation from rolling value-weighted price shows who is paying up for inventory |
| `ret_21d_xrank` | cross_sectional | cross-sectional momentum rank is the tradable signal form robust to market-wide moves |
| `vol_21d_xrank` | cross_sectional | volatility rank positions a name within the day's risk spectrum for regime-aware gating |
| `turnover_xrank` | cross_sectional | liquidity rank distinguishes market darlings from dormant names in the same market state |
| `sector_ret_21d` | cross_sectional | sector momentum captures the rotation flows that dominate a 22-sector market |
| `ret_vs_sector_21d` | cross_sectional | return relative to own sector isolates idiosyncratic strength from rotation beta |
| `breadth_above_ma50` | cross_sectional | share of names above their 50-day average is the market-wide regime thermometer |
| `dow` | calendar | Sunday and Thursday carry systematic open-of-week and pre-weekend retail flow effects |
| `month` | calendar | June-July budget season and December closing drive seasonal flows in Dhaka |
| `days_listed` | calendar | newly listed names trade under different rules and speculative attention than seasoned ones |
| `px_level_log` | structure | low-priced shares attract disproportionate retail speculation on the DSE |
| `gap_open_1d` | structure | open-versus-yesterday-close gap measures overnight information or manipulation pressure |
| `hl_position_1d` | structure | where the close sits in the day's range reveals end-of-session buying or selling urgency |
| `ma20_dist` | structure | distance from the 20-day average is the mean-reversion anchor retail chartists watch |
| `ma50_dist` | structure | the 50-day average is the trend line separating accumulation from markdown phases |
| `ma20_above_ma50` | structure | moving-average cross state is a self-fulfilling regime flag among local technical traders |
| `ycp_gap_flag` | structure | days where ycp diverges from prior close mark corporate-action ex-dates the model must know about |
| `days_since_event` | calendar | announcement-driven drift persists for days where information diffuses slowly |
| `board_meeting_soon` | calendar | a board meeting notice under LR 16(1) means dividends or earnings land within days |
| `regime_code` | cross_sectional | the same setup carries different odds in Panic versus Bull markets per spec regime gating |
