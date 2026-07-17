# tests/events/test_classifier.py
from vectora import db as vdb
from vectora.events import classifier as cls

# (title, expected_type, expected_materiality)
CASES = [
    ("GLDNJMF: Daily NAV", "daily_nav", 0),
    ("DSE NEWS: Daily Turnover of Main Board", "market_stats", 0),
    ("DSE NEWS: Awareness Message for Investors", "admin_notice", 0),
    ("BSEC NEWS: Awareness Message for Investors", "admin_notice", 0),
    ("DSE NEWS: Withdrawal of Authorized Representative", "admin_notice", 0),
    ("UNIONCAP: Board Meeting schedule under LR 16(1)", "board_meeting", 2),
    ("NATLIFEINS: Reschedule of Board Meeting under LR 16(1)", "board_meeting", 2),
    ("TB10Y0132: Record date for entitlement of coupon payment",
     "record_date", 1),
    ("PRIMELIFE: Resumption after Record Date", "trading_resume", 1),
    ("XYZ: Suspension for Record Date", "trading_suspension", 1),
    ("ABC: Halt of trading of the company", "trading_halt", 3),
    ("ABC: Price Limit Open", "price_limit_change", 2),
    ("MERCANBANK: Credit Rating Result", "credit_rating", 1),
    ("LINDEBD: Dividend Disbursement", "dividend_disbursement", 1),
    ("SQURPHARMA: Dividend Declaration", "dividend_declared", 3),
    ("ACI: Declaration of Interim Dividend and Audited Q2 Financials",
     "dividend_declared", 3),
    ("GP: Q2 Financials", "earnings_release", 3),
    ("GP: Q1 Financials", "earnings_release", 3),
    ("FIRSTFIN: Spot News", "spot_market", 2),
    ("ABC: Record Date and key features of the rights issuance",
     "rights_issue", 3),
    ("ABC: Query Response", "query_response", 2),
    ("ABC: Clarification on the news published in the online news",
     "query_response", 2),
    ("ABC: Signing of Selling & Distribution Agreement", "business_update", 2),
    ("ABC: Something entirely novel here", "unclassified", 1),
]


def test_taxonomy_on_observed_titles():
    for title, etype, mat in CASES:
        got_type, got_mat = cls.classify_title(title)
        assert got_type == etype, f"{title!r}: {got_type} != {etype}"
        assert got_mat == mat, f"{title!r}: materiality {got_mat} != {mat}"


def test_classify_new_writes_labels_and_is_incremental(test_db):
    vdb.upsert(test_db, "events", [
        dict(id="e1", post_date="2026-07-16", symbol="GP",
             title="GP: Q2 Financials", body="EPS 5.2", source="dse_news"),
        dict(id="e2", post_date="2026-07-16", symbol=None,
             title="DSE NEWS: Greetings Message", body="", source="dse_news"),
    ])
    result = cls.classify_new(test_db)
    assert result == {"classified": 2}
    rows = dict(test_db.execute(
        "SELECT event_id, event_type FROM event_labels").fetchall())
    assert rows["e1"] == "earnings_release"
    assert rows["e2"] == "admin_notice"
    assert cls.classify_new(test_db) == {"classified": 0}   # incremental


def test_event_labels_table_exists(test_db):
    tables = {r[0] for r in test_db.execute("SHOW TABLES").fetchall()}
    assert {"event_labels", "event_studies"} <= tables
