"""Record live dsebd.org pages as test fixtures. Run manually, rarely.

Usage: uv run python tools/record_fixtures.py [YYYY-MM-DD]
Dates default to the most recent Sun-Thu weekday before today.
"""
import sys
from datetime import date, timedelta
from pathlib import Path

from vectora.http import PoliteSession
from vectora.settings import DSE_BASE

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def last_trading_weekday(today: date) -> date:
    d = today - timedelta(days=1)
    while d.weekday() in (4, 5):  # Fri=4, Sat=5 closed
        d -= timedelta(days=1)
    return d


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    d = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else last_trading_weekday(date.today())
    ds = d.isoformat()
    s = PoliteSession()
    pages = {
        "day_end_archive.html": (
            f"{DSE_BASE}/day_end_archive.php",
            {"startDate": ds, "endDate": ds, "inst": "All Instrument", "archive": "data"},
        ),
        "old_news.html": (
            f"{DSE_BASE}/old_news.php",
            {"startDate": ds, "endDate": ds, "criteria": "4", "archive": "news"},
        ),
        "homepage.html": (f"{DSE_BASE}/", None),
        "company_GP.html": (f"{DSE_BASE}/displayCompany.php", {"name": "GP"}),
    }
    for fname, (url, params) in pages.items():
        html = s.get(url, params=params)
        (FIXTURES / fname).write_text(html, encoding="utf-8")
        print(f"{fname}: {len(html):,} bytes")


if __name__ == "__main__":
    main()
