"""CLI: python -m vectora run eod [--date YYYY-MM-DD]

Exit codes: 0 = success or clean skip; 1 = unusable day (quality 0) or
unknown command. Stage errors surface in output but only zero-quality
days fail the run — CI treats exit 1 as "investigate now".
"""
import argparse
import json
import sys

from vectora import orchestrator


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vectora")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run a pipeline stage")
    run.add_argument("stage", choices=["eod"])
    run.add_argument("--date", default=None, help="YYYY-MM-DD (default: today)")
    args = parser.parse_args(argv)

    if args.command == "run" and args.stage == "eod":
        summary = orchestrator.run_eod_live(args.date)
        print(json.dumps(summary, indent=1))
        if summary.get("skipped"):
            return 0
        return 1 if summary.get("quality_score") == 0 else 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
