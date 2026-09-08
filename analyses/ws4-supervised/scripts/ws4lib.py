"""
ws4lib.py — shared helpers for the WS4 supervised workstream.

Paths, split loading, the DEV-only label loader, candidate-feature
construction from tweet-level caches (so any tweet subset can be
featurized), and the ridge / GBM fitting primitives used by scripts 02–07.

Truth discipline: `load_dev_labels()` is the only label accessor used by
scripts 01–06. `sealed_truth.parquet` is touched exclusively by
07_unseal_evaluate.py.
"""

from __future__ import annotations

import json
import re
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WS4 = HERE.parent
REPO = WS4.parent.parent
WS0 = REPO / "ws0-harness"
INTER = WS4 / "intermediate"
OUT = WS4 / "outputs"
FIG = WS4 / "figures"
for d in (INTER, OUT, FIG):
    d.mkdir(exist_ok=True)

sys.path.insert(0, str(WS0))
import metrics  # noqa: E402  (ws0-harness/metrics.py)

SEED = 20260908
DIM = 100
SPACES = ["tfidf", "w2v", "doc2vec", "m2v", "behav"]
CONTENT = ["tfidf", "doc2vec"]
STYLE = ["w2v", "m2v", "behav"]

# ---- tokenizer: identical to ws0-harness/02_freeze_baselines.py ----
URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@\w+")
TOKEN_RE = re.compile(r"[a-z][a-z']+")


def tokenize(text: str) -> list[str]:
    t = text.lower()
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    t = t.replace("#", " ")
    return TOKEN_RE.findall(t)


def det_hash(s: str) -> int:
    return zlib.crc32(s.encode("utf-8"))


# ---- data loaders ----
def load_blind() -> pd.DataFrame:
    df = pd.read_parquet(WS0 / "blind_corpus.parquet")
    ab = pd.read_parquet(WS0 / "tweet_split_ab.parquet")[["tweet_id", "split"]]
    df = df.merge(ab, on="tweet_id", how="left")
    return df.sort_values(["candidate_id", "tweet_id"]).reset_index(drop=True)


def load_meta() -> pd.DataFrame:
    return pd.read_csv(WS0 / "baselines" / "candidate_metadata.csv").sort_values("candidate_id").reset_index(drop=True)


def load_splits() -> dict:
    return json.loads((WS0 / "ws4_splits.json").read_text())


def load_dev_labels() -> pd.Series:
    """DEV-only true_ideology, indexed by candidate_id. The sole pre-unseal label source."""
    lab = pd.read_parquet(WS0 / "ws4_dev_labels.parquet")
    return lab.set_index("candidate_id")["true_ideology"]


def load_org_ideologies() -> dict:
    return json.loads((REPO / "analyses" / "ws3-llm-scaling" / "outputs" / "org_ideologies.json").read_text())


def cand_order() -> list[str]:
    return load_meta()["candidate_id"].tolist()


# ---- tweet-level caches (written by 01_features.py) ----
def load_tweet_cache(space: str) -> np.ndarray:
    return np.load(INTER / f"tweet_{space}.npy", mmap_mode="r")


def load_full_features() -> dict[str, np.ndarray]:
    z = np.load(OUT / "features_full.npz", allow_pickle=True)
    return {k: z[k] for k in z.files}


# ---- candidate features from an arbitrary tweet subset ----
def _mean_rows(cache: np.ndarray, rows: np.ndarray) -> np.ndarray:
    if len(rows) == 0:
        return np.zeros(cache.shape[1], dtype=np.float32)
    return np.asarray(cache[np.sort(rows)]).mean(axis=0)


def behav_features(sub: pd.DataFrame, orgs: dict) -> np.ndarray:
    """sub: tweets of ONE candidate (already restricted to split A if desired).
    Returns [mean_source_ideology, has_retweet, rt_share, log1p_n, 20 source shares]."""
    org_names = list(orgs.keys())
    n = len(sub)
    rt = sub[sub.is_retweet.astype(bool)]
    shares = np.zeros(len(org_names), dtype=np.float32)
    if len(rt):
        vc = rt.retweeted_handle.value_counts()
        for j, o in enumerate(org_names):
            shares[j] = vc.get(o, 0) / len(rt)
        src_mean = float(np.mean([orgs.get(h, 0.0) for h in rt.retweeted_handle]))
        has = 1.0
    else:
        src_mean, has = 0.0, 0.0
    head = np.array([src_mean, has, len(rt) / max(n, 1), np.log1p(n)], dtype=np.float32)
    return np.concatenate([head, shares])


class Featurizer:
    """Builds candidate x feature matrices for any tweet subset, from caches."""

    def __init__(self, blind: pd.DataFrame | None = None):
        self.blind = blind if blind is not None else load_blind()
        self.orgs = load_org_ideologies()
        self.cands = cand_order()
        self.cache = {s: load_tweet_cache(s) for s in ("tfidf", "w2v", "m2v", "m2v_strip")}
        self.tokens = None  # lazy, for doc2vec inference
        self._d2v = None
        self.row_of_cand = {c: np.flatnonzero(self.blind.candidate_id.to_numpy() == c) for c in self.cands}

    @property
    def d2v(self):
        if self._d2v is None:
            from gensim.models import Doc2Vec
            self._d2v = Doc2Vec.load(str(INTER / "doc2vec.model"))
        return self._d2v

    def subset_rows(self, cid: str, k: int | None, rng: np.random.Generator,
                    originals_only: bool = False) -> np.ndarray:
        rows = self.row_of_cand[cid]
        if originals_only:
            rows = rows[~self.blind.is_retweet.to_numpy()[rows].astype(bool)]
        if k is not None and len(rows) > k:
            rows = rng.choice(rows, size=k, replace=False)
        return rows

    def features(self, space: str, rows_by_cand: dict[str, np.ndarray],
                 behav_split_a_only: bool = True) -> np.ndarray:
        if space in ("tfidf", "w2v", "m2v", "m2v_strip"):
            return np.vstack([_mean_rows(self.cache[space], rows_by_cand[c]) for c in self.cands])
        if space == "doc2vec":
            if self.tokens is None:
                self.tokens = self.blind.text.astype(str).map(tokenize).tolist()
            vecs = []
            for c in self.cands:
                toks = [t for r in rows_by_cand[c] for t in self.tokens[r]]
                self.d2v.random.seed(SEED)  # deterministic inference
                vecs.append(self.d2v.infer_vector(toks, epochs=50))
            return np.vstack(vecs).astype(np.float32)
        if space == "behav":
            is_a = (self.blind.split.to_numpy() == "A")
            out = []
            for c in self.cands:
                rows = rows_by_cand[c]
                if behav_split_a_only:
                    rows = rows[is_a[rows]]
                out.append(behav_features(self.blind.iloc[rows], self.orgs))
            return np.vstack(out)
        raise ValueError(space)


# ---- modelling primitives ----
ALPHAS = np.logspace(-3, 3, 13)


def fit_ridge(X: np.ndarray, y: np.ndarray, folds: np.ndarray | None = None):
    """Standardize + RidgeCV with inner folds (or 5-fold KFold if folds is None)."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold, PredefinedSplit
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    cv = PredefinedSplit(folds) if folds is not None else KFold(5, shuffle=True, random_state=SEED)
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS, cv=cv))
    return model.fit(X, y)


def fit_gbm(X: np.ndarray, y: np.ndarray):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(early_stopping=True, random_state=SEED).fit(X, y)


def oof_predictions(X: np.ndarray, y: np.ndarray, folds: np.ndarray, learner: str = "ridge") -> np.ndarray:
    """Out-of-fold predictions over the given fold labels (DEV inner folds)."""
    pred = np.full(len(y), np.nan)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        if learner == "ridge":
            m = fit_ridge(X[tr], y[tr])
        else:
            m = fit_gbm(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def style_projected(X: np.ndarray, rt_share: np.ndarray) -> np.ndarray:
    """Project out the PC most correlated with retweet share (confound gate)."""
    from sklearn.decomposition import PCA
    Xc = X - X.mean(axis=0, keepdims=True)
    pca = PCA(n_components=min(10, X.shape[1]), random_state=SEED).fit(Xc)
    P = pca.transform(Xc)
    r = np.array([abs(np.corrcoef(P[:, j], rt_share)[0, 1]) for j in range(P.shape[1])])
    k = int(np.nanargmax(r))
    return metrics.project_out(Xc, pca.components_[k:k + 1]), k, float(r[k])


def scores(pred: np.ndarray, truth: np.ndarray) -> dict:
    from scipy.stats import pearsonr, spearmanr
    return {
        "r": float(pearsonr(pred, truth)[0]),
        "rho": float(spearmanr(pred, truth)[0]),
        "rmse": float(np.sqrt(np.mean((pred - truth) ** 2))),
    }


def paired_bootstrap_dr(pred_a: np.ndarray, pred_b: np.ndarray, truth: np.ndarray,
                        n_boot: int = 2000, seed: int = SEED) -> dict:
    """Δr = r(a) − r(b) with percentile 95% CI over candidate resamples."""
    rng = np.random.default_rng(seed)
    n = len(truth)
    d = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ra = np.corrcoef(pred_a[idx], truth[idx])[0, 1]
        rb = np.corrcoef(pred_b[idx], truth[idx])[0, 1]
        d.append(ra - rb)
    d = np.array(d)
    return {"dr": float(np.corrcoef(pred_a, truth)[0, 1] - np.corrcoef(pred_b, truth)[0, 1]),
            "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5))}
