"""Where the scrapers point.

The 2026-09-24 outage: DSE moved its public site to www.dse.com.bd. From
then on www.dsebd.org 301s its homepage to the new site and answers every
legacy page (day_end_archive.php, old_news.php, latest_share_price_scroll_l.php)
with 404, later 410 Gone. The new site loads its data client-side from /api,
which its robots.txt disallows. The legacy pages still live, unchanged, on
old.dsebd.org (verified 2026-11-03), so the parsers keep working there.
"""
from vectora import health, settings
from vectora.collect import dse_eod, dse_indices, dse_intraday, dse_news

LEGACY = "https://old.dsebd.org"


class RecordingSession:
    def __init__(self):
        self.urls: list[str] = []

    def get(self, url, params=None):
        self.urls.append(url)
        return ""


def test_retry_budget_rides_out_the_legacy_hosts_500s():
    """old.dsebd.org answered 500 to about 3 requests in 10, sometimes several
    in a row (measured 2026-11-03); three attempts lost whole days."""
    from vectora.http import PoliteSession
    assert PoliteSession().max_retries >= 6


def test_base_is_the_legacy_host():
    assert settings.DSE_BASE == LEGACY


def test_archive_scrapers_use_the_legacy_host():
    assert dse_eod.URL == f"{LEGACY}/day_end_archive.php"
    assert dse_news.URL == f"{LEGACY}/old_news.php"
    assert dse_intraday.URL == f"{LEGACY}/latest_share_price_scroll_l.php"


def test_homepage_is_fetched_from_the_legacy_host():
    s = RecordingSession()
    dse_indices.fetch_homepage(s)
    assert s.urls == [f"{LEGACY}/"]


def test_health_canary_probes_the_legacy_host(test_db):
    s = RecordingSession()
    health.check(test_db, holidays=set(), session=s)
    assert s.urls and all(u.startswith(LEGACY) for u in s.urls)
