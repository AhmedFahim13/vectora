"""CLI: python -m vectora run eod [--date YYYY-MM-DD]

Without --date, gap-fills every trading day since the last successful
collection. Exit codes: 0 = all runs clean (or clean skip); 1 = any run had
stage errors, a crashed validation, or quality below settings.MIN_QUALITY_SCORE.
CI treats exit 1 as "investigate now" — data is still committed either way.
"""
import argparse
import json
import sys

from vectora import orchestrator
from vectora.settings import MIN_QUALITY_SCORE


def _run_failed(summary: dict) -> bool:
    if summary.get("skipped"):
        return False
    quality = summary.get("quality_score")
    return (
        quality is None            # validation crashed or never ran
        or quality < MIN_QUALITY_SCORE
        or bool(summary.get("errors"))
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vectora")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run a pipeline stage")
    run.add_argument("stage",
                     choices=["eod", "train", "predict", "digest", "outcomes",
                              "vault", "regime", "events", "zscan", "intraday", "evaluate",
                              "health"])
    run.add_argument("--date", default=None,
                     help="YYYY-MM-DD (default: gap-fill up to today)")
    run.add_argument("--target", default="g5_h10",
                     help="label target for train, e.g. g5_h10")
    args = parser.parse_args(argv)

    if args.command == "run" and args.stage == "eod":
        summaries = orchestrator.run_eod_live(args.date)
        print(json.dumps(summaries, indent=1))
        return 1 if any(_run_failed(s) for s in summaries) else 0

    if args.command == "run" and args.stage == "train":
        from vectora import db as vdb
        from vectora.settings import DB_PATH
        from vectora.train import trainer
        con = vdb.connect(DB_PATH)
        try:
            vdb.init_schema(con)
            result = trainer.run(con, target=args.target)
        finally:
            con.close()
        print(json.dumps(result, indent=1))
        return 0 if result["lgbm_brier"] < result["logistic_brier"] else 1

    if args.command == "run" and args.stage == "predict":
        from vectora import db as vdb
        from vectora.predict import engine as pengine
        from vectora.settings import DB_PATH
        con = vdb.connect(DB_PATH)
        try:
            vdb.init_schema(con)
            result = pengine.run_predict(con, date_str=args.date)
        finally:
            con.close()
        print(json.dumps(result, indent=1))
        return 0

    if args.command == "run" and args.stage == "digest":
        from vectora import db as vdb
        from vectora.alerts import digest
        from vectora.settings import DB_PATH
        con = vdb.connect(DB_PATH)
        try:
            vdb.init_schema(con)
            date_str = args.date or str(con.execute(
                "SELECT max(date) FROM predictions").fetchone()[0])
            body = digest.build(con, date_str)
            from vectora.alerts import signals as sig
            new_symbols = sig.log_signal_alerts(con, date_str)
        finally:
            con.close()
        n_signals = body.count("### ")
        prefix = f"[{len(new_symbols)} NEW] " if new_symbols else ""
        subject = f"{prefix}Vectora digest {date_str} - {n_signals} signal(s)"
        result = digest.send_or_save(subject, body)
        print(json.dumps(result, indent=1))
        return 0

    if args.command == "run" and args.stage == "outcomes":
        from vectora import db as vdb
        from vectora.outcomes import resolver
        from vectora.settings import DB_PATH
        con = vdb.connect(DB_PATH)
        try:
            vdb.init_schema(con)
            result = resolver.resolve(con)
        finally:
            con.close()
        print(json.dumps(result, indent=1))
        return 0

    if args.command == "run" and args.stage == "vault":
        from vectora import db as vdb
        from vectora.settings import DB_PATH
        from vectora.vault import generator
        con = vdb.connect(DB_PATH)
        try:
            vdb.init_schema(con)
            date_str = args.date or str(con.execute(
                "SELECT max(date) FROM predictions").fetchone()[0])
            result = generator.generate(con, date_str)
        finally:
            con.close()
        print(json.dumps(result, indent=1))
        return 0

    if args.command == "run" and args.stage == "regime":
        from vectora import db as vdb
        from vectora.regime import rules
        from vectora.settings import DB_PATH
        con = vdb.connect(DB_PATH)
        try:
            vdb.init_schema(con)
            result = rules.classify_history(con)
        finally:
            con.close()
        print(json.dumps(result, indent=1))
        return 0

    if args.command == "run" and args.stage == "events":
        from vectora import db as vdb
        from vectora.events import classifier
        from vectora.settings import DB_PATH
        con = vdb.connect(DB_PATH)
        try:
            vdb.init_schema(con)
            result = classifier.classify_new(con)
        finally:
            con.close()
        print(json.dumps(result, indent=1))
        return 0

    if args.command == "run" and args.stage == "zscan":
        from vectora import db as vdb
        from vectora.settings import DB_PATH
        from vectora.zmod import scan
        con = vdb.connect(DB_PATH)
        try:
            vdb.init_schema(con)
            result = scan.run_zscan(con, date_str=args.date)
        finally:
            con.close()
        print(json.dumps(result, indent=1))
        return 0

    if args.command == "run" and args.stage == "intraday":
        from vectora import db as vdb
        from vectora.alerts import intraday
        from vectora.http import PoliteSession
        from vectora.settings import DB_PATH
        con = vdb.connect(DB_PATH)
        try:
            vdb.init_schema(con)
            result = intraday.run_intraday(con, PoliteSession())
        finally:
            con.close()
        print(json.dumps(result, indent=1))
        return 0

    if args.command == "run" and args.stage == "evaluate":
        from vectora import db as vdb
        from vectora.evaluate import report
        from vectora.settings import DB_PATH
        con = vdb.connect(DB_PATH)
        try:
            vdb.init_schema(con)
            result = report.evaluate(con)
        finally:
            con.close()
        print(json.dumps({k: v for k, v in result.items()
                          if k != "targets"} | {
            "targets": {t: {k2: v2 for k2, v2 in m.items()
                            if k2 != "reliability"}
                        for t, m in result.get("targets", {}).items()}},
              indent=1, default=str))
        return 0

    if args.command == "run" and args.stage == "health":
        from vectora import db as vdb
        from vectora import health
        from vectora.alerts.digest import send_or_save
        from vectora.http import PoliteSession
        from vectora.settings import DB_PATH
        con = vdb.connect(DB_PATH)
        try:
            vdb.init_schema(con)
            result = health.check(con, session=PoliteSession())
        finally:
            con.close()
        print(json.dumps(result, indent=1, default=str))
        if not result["ok"]:
            failing = [c for c in result["checks"] if not c["ok"]]
            body = "\n".join(
                f"- {c['name']}: {c['detail']}" for c in failing)
            send_or_save(f"[HEALTH] Vectora: {len(failing)} check(s) failing",
                         body)
            return 1
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
