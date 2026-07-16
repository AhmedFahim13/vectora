# vectora/train/walkforward.py
"""Expanding-window walk-forward splits with an embargo gap (spec §10.2).

The embargo (>= max label horizon, default 30 days) sits between train end
and test start so no training label's forward window overlaps test data.
Never use random k-fold on time series — this module is the only sanctioned
splitter."""
import datetime as dt
from dataclasses import dataclass


@dataclass(frozen=True)
class Split:
    train_start: dt.date
    train_end: dt.date
    test_start: dt.date
    test_end: dt.date


def splits(dates: list[dt.date], min_train_days: int = 750,
           test_days: int = 126, embargo_days: int = 30) -> list[Split]:
    if not dates:
        return []
    dates = sorted(set(dates))
    start = dates[0]
    out = []
    train_end_i = min_train_days - 1
    while True:
        test_start_i = train_end_i + embargo_days + 1
        test_end_i = test_start_i + test_days - 1
        if test_end_i >= len(dates):
            break
        out.append(Split(
            train_start=start,
            train_end=dates[train_end_i],
            test_start=dates[test_start_i],
            test_end=dates[test_end_i],
        ))
        train_end_i = test_end_i - embargo_days  # next train absorbs this test
    return out


def role(split: Split, d: dt.date) -> str:
    if d <= split.train_end:
        return "train"
    if d < split.test_start:
        return "embargo"
    if d <= split.test_end:
        return "test"
    return "future"
