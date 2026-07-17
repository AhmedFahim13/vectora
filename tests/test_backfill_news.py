# tests/test_backfill_news.py
from tools import backfill_news as bn


def test_month_windows():
    wins = bn.month_windows("2025-11-01", "2026-02-10")
    assert wins == [("2025-11-01", "2025-11-30"), ("2025-12-01", "2025-12-31"),
                    ("2026-01-01", "2026-01-31"), ("2026-02-01", "2026-02-10")]


NEWS_HTML = """
<table class="table-news">
<tr><th>Trading Code:</th><td>GP</td></tr>
<tr><th>News Title:</th><td>GP: Dividend Declaration</td></tr>
<tr><th>News:</th><td>Cash dividend 125%.</td></tr>
<tr><th>Post Date:</th><td>2025-11-05</td></tr>
</table>"""


class FakeSession:
    def __init__(self):
        self.calls = []

    def get(self, url, params=None):
        self.calls.append(params)
        return NEWS_HTML


def test_backfill_range_fetches_each_month_and_upserts(test_db, tmp_path):
    s = FakeSession()
    result = bn.backfill(test_db, s, "2025-11-01", "2026-01-15",
                         raw_dir=tmp_path)
    assert len(s.calls) == 3                       # three month windows
    assert s.calls[0]["startDate"] == "2025-11-01"
    assert result["months"] == 3
    # same canned HTML each month -> same event id -> one row (idempotent)
    assert test_db.execute("SELECT count(*) FROM events").fetchone()[0] == 1
    assert len(list(tmp_path.rglob("*.html.gz"))) == 3
