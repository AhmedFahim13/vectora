"""Tradable universe: active equities passing a trailing liquidity floor.

Illiquidity is the #1 practical risk on the DSE (spec §2): signals on
names that trade a few thousand taka a day are untradable noise, so the
universe is filtered on trailing 60-trading-day median daily traded value.
"""

TRAILING_DAYS = 60


def tradable_universe(con, as_of: str, min_median_value_mn: float = 1.0) -> list[str]:
    rows = con.execute(
        """
        WITH recent AS (
            SELECT symbol, value_mn,
                   row_number() OVER (PARTITION BY symbol ORDER BY date DESC) AS rn
            FROM prices
            WHERE date <= ?
        )
        SELECT r.symbol
        FROM recent r
        JOIN symbols s USING (symbol)
        WHERE r.rn <= ?
          AND s.instrument_type = 'Equity'
          AND s.listing_status = 'active'
        GROUP BY r.symbol
        HAVING median(r.value_mn) >= ?
        ORDER BY r.symbol
        """,
        [as_of, TRAILING_DAYS, min_median_value_mn],
    ).fetchall()
    return [r[0] for r in rows]
