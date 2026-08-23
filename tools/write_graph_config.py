"""Write the Obsidian graph configuration for the Vectora vault.

The default graph is a grey hairball: no colour groups, every label drawn at
every zoom level, and forces tuned for a vault an order of magnitude smaller
than this one. This is generated rather than hand-edited so it survives a
vault rebuild and stays in version control with the rest of the system.

The colour groups are the point. They are search queries evaluated live by
Obsidian, so the graph becomes a market state view rather than decoration:
a company note currently carrying a Strong Buy summary is green tonight and
red next month if the posture flips, with no regeneration.

The three company queries are mutually exclusive by construction — the
posture groups exclude "thin float" — so the result does not depend on
which group Obsidian happens to evaluate first. Risk wins over posture on
purpose: a Strong Buy you cannot safely exit should not read as green.
"""
import json
from pathlib import Path

from vectora.settings import VAULT_DIR


def _rgb(hex_colour: str) -> int:
    return int(hex_colour.lstrip("#"), 16)


COLOR_GROUPS = [
    # risk first in the file for readability; the queries are exclusive
    {"query": 'path:Companies "thin float"', "color": "#F59E0B"},
    {"query": 'path:Companies "**Strong Buy**" -"thin float"',
     "color": "#22C55E"},
    {"query": 'path:Companies "**Strong Sell**" -"thin float"',
     "color": "#EF4444"},
    {"query": "path:Sectors", "color": "#14B8A6"},
    {"query": "path:Predictions", "color": "#A855F7"},
    {"query": "path:Journal", "color": "#64748B"},
    {"query": "path:Evaluations", "color": "#3B82F6"},
]

GRAPH = {
    "collapse-filter": True,
    "search": "",
    "showTags": False,
    "showAttachments": False,
    "hideUnresolved": True,      # unresolved links are noise, not structure
    "showOrphans": False,        # 500+ nodes: an orphan adds nothing
    "collapse-color-groups": False,
    "colorGroups": [
        {"query": g["query"], "color": {"a": 1, "rgb": _rgb(g["color"])}}
        for g in COLOR_GROUPS
    ],
    "collapse-display": False,
    "showArrow": False,
    # -1.4 keeps labels hidden until you zoom in. At 0 every one of 500+
    # labels draws at once, which is what made the graph unreadable.
    "textFadeMultiplier": -1.4,
    "nodeSizeMultiplier": 1.25,   # let sector and journal hubs read as hubs
    "lineSizeMultiplier": 0.55,   # thousands of links; thin them down
    "collapse-forces": False,
    # Tuned for ~520 nodes with dense sector membership. Weaker centring and
    # weaker links let sectors separate into their own clusters instead of
    # collapsing into one ball; stronger repulsion keeps names legible.
    "centerStrength": 0.32,
    "repelStrength": 14.5,
    "linkStrength": 0.55,
    "linkDistance": 105,
    "scale": 0.42,
    "close": False,
}

# Local graph: depth 2 is the useful setting here. Depth 1 shows a company
# and its sector; depth 2 shows the sector's other members, which is the
# comparison a reader actually wants. Obsidian stores this per-view, so it
# is written here as a default and repeated in the vault README.
LOCAL_GRAPH = {
    "collapse-filter": True, "search": "", "showTags": False,
    "showAttachments": False, "hideUnresolved": True, "showOrphans": False,
    "collapse-color-groups": False,
    "colorGroups": GRAPH["colorGroups"],
    "collapse-display": False, "showArrow": False,
    "textFadeMultiplier": -0.6, "nodeSizeMultiplier": 1.4,
    "lineSizeMultiplier": 1, "collapse-forces": False,
    "centerStrength": 0.5, "repelStrength": 11, "linkStrength": 1,
    "linkDistance": 90, "scale": 1,
    "localJumps": 2, "localBacklinks": True, "localForelinks": True,
    "localInterlinks": True, "close": False,
}


def write(vault_dir: Path = VAULT_DIR) -> dict:
    cfg = Path(vault_dir) / ".obsidian"
    cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "graph.json").write_text(
        json.dumps(GRAPH, indent=2), encoding="utf-8")
    (cfg / "graph-local.json").write_text(
        json.dumps(LOCAL_GRAPH, indent=2), encoding="utf-8")
    return {"color_groups": len(GRAPH["colorGroups"]),
            "path": str(cfg / "graph.json")}


if __name__ == "__main__":
    print(json.dumps(write(), indent=1))
