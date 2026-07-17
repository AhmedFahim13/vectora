"""Announcement taxonomy (spec §12.1): ordered regex rules over titles,
grounded in observed DSE title shapes (see plan doc). First match wins.
Materiality: 0 routine noise, 1 mechanical/administrative, 2 informative,
3 price-sensitive. Labels are append-only; raw events are never mutated."""
import re

from vectora import db as vdb

# (compiled regex, event_type, materiality) — ordered, first match wins.
RULES = [
    (r"daily nav", "daily_nav", 0),
    (r"daily turnover|market statistics", "market_stats", 0),
    (r"awareness message|greetings message|withdrawal of authorized|"
     r"lodging investor complaints", "admin_notice", 0),
    (r"halt of trading", "trading_halt", 3),
    (r"price limit", "price_limit_change", 2),
    (r"suspension for record date", "trading_suspension", 1),
    (r"resumption after record date", "trading_resume", 1),
    (r"record date.*rights|rights issu", "rights_issue", 3),
    (r"record date", "record_date", 1),
    (r"board meeting", "board_meeting", 2),
    (r"declaration of.*dividend|dividend declaration", "dividend_declared", 3),
    (r"dividend disbursement", "dividend_disbursement", 1),
    (r"q[1-4] financials|quarterly financ|audited financ|earnings",
     "earnings_release", 3),
    (r"credit rating", "credit_rating", 1),
    (r"spot news|spot market", "spot_market", 2),
    (r"query response|clarification on the news", "query_response", 2),
    (r"agreement|acquisition|new plant|expansion|contract",
     "business_update", 2),
    (r"agm|annual general meeting", "agm_notice", 1),
    (r"category", "category_change", 3),
]
_COMPILED = [(re.compile(p, re.IGNORECASE), t, m) for p, t, m in RULES]

# dividend_declared beats earnings_release for combined announcements
# ("Declaration of Interim Dividend and Audited Q2 Financials") because
# the dividend rule sits earlier in RULES — order is load-bearing.


def classify_title(title: str) -> tuple[str, int]:
    text = title.strip()
    for rx, etype, mat in _COMPILED:
        if rx.search(text):
            return etype, mat
    return "unclassified", 1


def classify_new(con) -> dict:
    rows = con.execute(
        """
        SELECT e.id, e.title FROM events e
        LEFT JOIN event_labels l ON l.event_id = e.id
        WHERE l.event_id IS NULL
        """).fetchall()
    labels = []
    for event_id, title in rows:
        etype, mat = classify_title(title or "")
        labels.append({"event_id": event_id, "event_type": etype,
                       "materiality": mat})
    if labels:
        vdb.upsert(con, "event_labels", labels)
    return {"classified": len(labels)}
