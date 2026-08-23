"""Write the Obsidian graph configuration for the Vectora vault.

The default graph is a grey hairball: no colour groups, every label drawn at
every zoom level, and forces tuned for a vault an order of magnitude smaller
than this one. This is generated rather than hand-edited so it survives a
vault rebuild and stays in version control with the rest of the system.

The colour groups are the point. They are tag queries evaluated live by
Obsidian, so the graph becomes a market state view rather than decoration:
a company tagged #posture/strong-buy tonight is green, and red next month
when the posture flips, with no regeneration of anything.

Tags rather than note text is a correctness choice, not a style one. A
content query such as '"**Strong Buy**"' depends on how the search index
handles markdown emphasis and quietly matched nothing; a tag is a
first-class token and matches exactly.

The company queries are mutually exclusive by construction — every posture
group excludes #risk/thin-float — so the result does not depend on which
group Obsidian evaluates first. Risk wins over posture on purpose: a Strong
Buy you cannot safely exit should not read as green.
"""
import json
from pathlib import Path

from vectora.settings import VAULT_DIR


def _rgb(hex_colour: str) -> int:
    return int(hex_colour.lstrip("#"), 16)


COLOR_GROUPS = [
    # Tags, not note text. A content query like `"**Strong Buy**"` depends on
    # how the search index treats markdown emphasis; a tag is a first-class
    # token that matches exactly, every time.
    #
    # Risk is listed first and deliberately wins over posture: a Strong Buy
    # you could not safely exit should not read as green.
    {"query": "tag:#risk/thin-float", "color": "#F59E0B"},
    {"query": "tag:#posture/strong-buy -tag:#risk/thin-float",
     "color": "#22C55E"},
    {"query": "tag:#posture/buy -tag:#risk/thin-float", "color": "#4ADE80"},
    {"query": "tag:#posture/strong-sell -tag:#risk/thin-float",
     "color": "#EF4444"},
    {"query": "tag:#posture/sell -tag:#risk/thin-float", "color": "#FB7185"},
    {"query": "tag:#posture/hold -tag:#risk/thin-float", "color": "#94A3B8"},
    {"query": "tag:#sector-note", "color": "#22D3EE"},
    {"query": "tag:#signal", "color": "#C084FC"},
    {"query": "tag:#journal", "color": "#475569"},
    {"query": "path:Screens OR path:Postures OR path:Evaluations",
     "color": "#FBBF24"},
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
    # negative keeps labels hidden until you zoom in. At 0, every one of
    # 540 labels draws at once, which is what made the graph a grey wall.
    "textFadeMultiplier": -1.1,
    "nodeSizeMultiplier": 1.45,   # let sector and journal hubs read as hubs
    "lineSizeMultiplier": 0.35,   # thousands of links; thin them down
    "collapse-forces": False,
    # Tuned for ~540 nodes with dense sector membership. The first attempt
    # (centre 0.32, repel 14.5, link 0.55, distance 105) still produced a
    # single ball: centring and link strength were both pulling inward
    # faster than repulsion could push out. Near-zero centring, weak links
    # and a long rest length let each sector settle into its own lobe.
    "centerStrength": 0.10,
    "repelStrength": 20,
    "linkStrength": 0.25,
    "linkDistance": 320,
    "scale": 0.30,
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
