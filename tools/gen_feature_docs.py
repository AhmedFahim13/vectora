"""Generate docs/features.md from the feature registry (spec §22)."""
from pathlib import Path

from vectora.features import registry


def main() -> int:
    specs = registry.load()
    lines = ["# Feature registry", "",
             f"{len(specs)} features. Every feature documents its economic "
             "reasoning (enforced by test).", "",
             "| name | family | reasoning |", "|---|---|---|"]
    lines += [f"| `{s.name}` | {s.family} | {s.reasoning} |" for s in specs]
    Path("docs/features.md").write_text("\n".join(lines) + "\n",
                                        encoding="utf-8")
    print(f"docs/features.md: {len(specs)} features")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
