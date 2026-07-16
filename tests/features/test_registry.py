# tests/features/test_registry.py
import pytest

from vectora.features import registry


def test_load_registry_returns_specs():
    specs = registry.load()
    assert len(specs) >= 35
    names = [s.name for s in specs]
    assert len(names) == len(set(names))          # unique
    assert "ret_21d" in names and "amihud_21d" in names


def test_every_feature_documents_reasoning():
    for s in registry.load():
        assert len(s.reasoning) >= 20, f"{s.name} lacks documented reasoning"
        assert s.family in registry.KNOWN_FAMILIES


def test_unknown_family_rejected(tmp_path):
    bad = tmp_path / "f.yaml"
    bad.write_text(
        "features:\n  - name: x\n    family: nonsense\n"
        "    fn: ret_nd\n    params: {days: 1}\n"
        "    reasoning: twenty characters of reasoning here\n",
        encoding="utf-8")
    with pytest.raises(ValueError, match="nonsense"):
        registry.load(bad)
