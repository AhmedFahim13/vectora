"""What a round trip actually costs on the DSE.

This is the most important file in the backtest and the least interesting
to write. Today's measurement showed the same strategy earning +2.67% per
trade at zero cost and +1.67% at 1% — the entire question of whether this
system is profitable lives inside these numbers, not inside the model.

DEFAULTS ARE ESTIMATES, NOT FACTS. Bangladeshi brokerage commission is
negotiable and varies by broker and account size; the regulatory components
(Howla, laga, CDBL) are small but real. Override them with the rates on an
actual contract note before trusting any result computed here.

Slippage is modelled rather than assumed away, because on this exchange it
is often larger than commission. A stock trading 2 mn BDT a day cannot
absorb a 1 mn BDT order at the quoted price, and roughly a third of the
board trades that thinly.
"""
from dataclasses import dataclass

# per side unless stated
DEFAULT_COMMISSION = 0.0040      # 0.40%, a typical negotiated retail rate
DEFAULT_REGULATORY = 0.0005      # Howla + laga + CDBL, approximate
DEFAULT_HALF_SPREAD = 0.0015     # 15 bps: you cross the spread to get filled
DEFAULT_IMPACT_COEF = 0.10       # price impact per unit of daily turnover taken
MAX_IMPACT = 0.05                # a sanity cap; beyond this the trade is fiction


@dataclass(frozen=True)
class CostModel:
    commission: float = DEFAULT_COMMISSION
    regulatory: float = DEFAULT_REGULATORY
    half_spread: float = DEFAULT_HALF_SPREAD
    impact_coef: float = DEFAULT_IMPACT_COEF
    max_impact: float = MAX_IMPACT

    def fixed_side(self) -> float:
        """Costs that do not depend on order size."""
        return self.commission + self.regulatory + self.half_spread

    def impact(self, notional_mn: float, adv_mn: float | None) -> float:
        """Extra cost from being a large share of the day's turnover.

        With no turnover figure available the impact is unknown, not zero —
        99% of the price history carries no turnover column, so returning 0
        there would quietly flatter every backtest run over deep history.
        The caller decides what to do with an unknown; this returns the cap.
        """
        if adv_mn is None or adv_mn <= 0:
            return self.max_impact
        return min(self.impact_coef * (notional_mn / adv_mn), self.max_impact)

    def side_cost(self, notional_mn: float, adv_mn: float | None) -> float:
        return self.fixed_side() + self.impact(notional_mn, adv_mn)

    def round_trip(self, notional_mn: float = 0.0,
                   adv_mn: float | None = None) -> float:
        """Total fractional cost of getting in and back out."""
        return 2 * self.side_cost(notional_mn, adv_mn)


ZERO = CostModel(commission=0.0, regulatory=0.0, half_spread=0.0,
                 impact_coef=0.0, max_impact=0.0)
