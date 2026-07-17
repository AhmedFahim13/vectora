"""Daily digest (spec §16, Phase 3B slice 1): one email per trading day
summarizing signals, suppressions, and quality. Degrades gracefully — with
no GMAIL_APP_PASSWORD in the environment the digest is written to reports/
instead of sent, so the pipeline never fails on a missing secret.

Urgent single alerts, urgency tiers, and throttling arrive with the rest
of Phase 3B; today there is exactly one message class.
"""
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path

from vectora.settings import REPORTS_DIR

EMAIL_ADDRESS = "ahmed.fahim.official.bd@gmail.com"  # sender == recipient


def build(con, date_str: str) -> str:
    n = con.execute(
        "SELECT count(*) FROM predictions WHERE date = ?", [date_str]
    ).fetchone()[0]
    if n == 0:
        return (f"# Vectora digest {date_str}\n\n"
                "No predictions for this date (holiday, pipeline failure, "
                "or predict stage not yet run).\n")
    quality = con.execute(
        "SELECT score FROM data_quality WHERE date = ? AND source = 'dse_eod'",
        [date_str]).fetchone()
    signals = con.execute(
        """
        SELECT p.symbol, p.target, p.probability, e.rendered
        FROM predictions p LEFT JOIN explanations e ON e.prediction_id = p.id
        WHERE p.date = ? AND p.is_signal
        ORDER BY p.probability DESC
        """, [date_str]).fetchall()
    suppressed = con.execute(
        """
        SELECT suppressed_reason, count(*) FROM predictions
        WHERE date = ? AND NOT is_signal GROUP BY 1 ORDER BY 2 DESC
        """, [date_str]).fetchall()

    regime = con.execute(
        "SELECT regime FROM regimes WHERE date = ?", [date_str]).fetchone()
    q = quality[0] if quality else "unknown"
    reg = regime[0] if regime else "unclassified"
    lines = [
        f"# Vectora digest {date_str}",
        "",
        f"{n} predictions | {len(signals)} signal(s) | data quality {q} | "
        f"regime {reg}",
        "",
    ]
    if signals:
        lines.append("## Signals")
        for sym, target, p, rendered in signals:
            lines += ["", f"### {sym} ({target}, {p:.0%})", "",
                      rendered or "(no explanation stored)"]
    else:
        lines.append("No signals today - no setup cleared the admission gates.")
    lines += ["", "## Suppressions"]
    lines += [f"- {reason}: {cnt}" for reason, cnt in suppressed]
    lines += ["", "_Research tool, not investment advice._", ""]
    return "\n".join(lines)


def send_or_save(subject: str, body: str,
                 reports_dir: Path = REPORTS_DIR) -> dict:
    password = os.environ.get("GMAIL_APP_PASSWORD")
    if not password:
        reports_dir.mkdir(parents=True, exist_ok=True)
        safe = subject.replace(" ", "_").replace(":", "")
        path = reports_dir / f"digest_{safe}.md"
        path.write_text(body, encoding="utf-8")
        return {"sent": False, "path": str(path)}
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = EMAIL_ADDRESS
    msg.set_content(body)
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(EMAIL_ADDRESS, password)
        smtp.send_message(msg)
    return {"sent": True, "path": None}
