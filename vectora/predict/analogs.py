# vectora/predict/analogs.py
"""Historical-analog retrieval (spec §15): k nearest labeled situations in
standardized feature space, summarized by their realized outcomes. This is
both the explanation ingredient ("12 of 20 similar setups hit the target")
and the risk engine's empirical move-size estimate."""
import numpy as np
import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class AnalogIndex:
    def __init__(self, nn, imputer, scaler, outcomes: np.ndarray):
        self._nn = nn
        self._imputer = imputer
        self._scaler = scaler
        # outcomes columns: label, fwdmax, fwdmin
        self._outcomes = outcomes
        self.n_rows = len(outcomes)

    @classmethod
    def fit(cls, history: pl.DataFrame, feature_names: list[str],
            label_col: str, fwdmax_col: str, fwdmin_col: str) -> "AnalogIndex":
        usable = history.filter(
            pl.col(label_col).is_not_null()
            & pl.col(fwdmax_col).is_not_null()
            & pl.col(fwdmin_col).is_not_null())
        X = usable.select(feature_names).to_numpy().astype(np.float64)
        imputer = SimpleImputer(strategy="median").fit(X)
        scaler = StandardScaler().fit(imputer.transform(X))
        Xs = scaler.transform(imputer.transform(X))
        nn = NearestNeighbors(n_neighbors=50).fit(Xs)
        outcomes = usable.select(
            [label_col, fwdmax_col, fwdmin_col]).to_numpy().astype(np.float64)
        return cls(nn, imputer, scaler, outcomes)

    def query(self, x: np.ndarray, k: int = 20) -> dict:
        k = min(k, self.n_rows)
        xs = self._scaler.transform(
            self._imputer.transform(x.reshape(1, -1)))
        _, idx = self._nn.kneighbors(xs, n_neighbors=k)
        o = self._outcomes[idx[0]]
        return {
            "hit_rate": float(o[:, 0].mean()),
            "median_up": float(np.median(o[:, 1])),
            "median_down": float(np.median(o[:, 2])),
            "max_drawdown": float(o[:, 2].min()),
            "n": int(k),
        }
