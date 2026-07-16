"""Feature registry: every feature is declared in features.yaml with its
computation function, params, family, and documented economic reasoning
(spec §8 requires reasoning; a test enforces it)."""
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config" / "features.yaml"

KNOWN_FAMILIES = {
    "momentum", "volatility", "liquidity", "volume",
    "cross_sectional", "calendar", "structure",
}


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    family: str
    fn: str
    params: dict
    reasoning: str


def load(path: Path = DEFAULT_PATH) -> list[FeatureSpec]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    specs = []
    for item in raw["features"]:
        spec = FeatureSpec(
            name=item["name"], family=item["family"], fn=item["fn"],
            params=item.get("params") or {}, reasoning=item["reasoning"],
        )
        if spec.family not in KNOWN_FAMILIES:
            raise ValueError(f"unknown family '{spec.family}' for {spec.name}")
        specs.append(spec)
    names = [s.name for s in specs]
    if len(names) != len(set(names)):
        raise ValueError("duplicate feature names in registry")
    return specs
