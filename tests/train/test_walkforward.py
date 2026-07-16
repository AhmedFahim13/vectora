# tests/train/test_walkforward.py
import datetime as dt

from vectora.train import walkforward as wf


def _dates(n):
    d0 = dt.date(2020, 1, 1)
    return [d0 + dt.timedelta(days=i) for i in range(n)]


def test_expanding_splits_with_embargo():
    splits = wf.splits(_dates(1500), min_train_days=750, test_days=126,
                       embargo_days=30)
    assert len(splits) >= 4
    for s in splits:
        assert s.train_start < s.train_end < s.test_start <= s.test_end
        # embargo: gap between train end and test start
        assert (s.test_start - s.train_end).days >= 30
    # expanding: every train window starts at the beginning
    assert len({s.train_start for s in splits}) == 1
    # test windows are consecutive and non-overlapping
    for a, b in zip(splits, splits[1:], strict=False):
        assert b.test_start > a.test_end


def test_no_split_when_history_too_short():
    assert wf.splits(_dates(400), min_train_days=750, test_days=126,
                     embargo_days=30) == []


def test_assign_rows():
    splits = wf.splits(_dates(1500), min_train_days=750, test_days=126,
                       embargo_days=30)
    s = splits[0]
    assert wf.role(s, s.train_start) == "train"
    assert wf.role(s, s.train_end + dt.timedelta(days=1)) == "embargo"
    assert wf.role(s, s.test_start) == "test"
    assert wf.role(s, s.test_end + dt.timedelta(days=1)) == "future"
