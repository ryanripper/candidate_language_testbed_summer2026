"""
04_text_scarcity.py — E4.3 text scarcity (DEV only; TEST untouched)
-------------------------------------------------------------------
For k in {5, 10, 25, 50, 100, all} tweets per candidate and two regimes
(mixed = retweets included; originals = retweets excluded), draw a random
k-subset of every candidate's tweets (DEV *and* TEST — no labels involved),
recompute every feature space from the subset, and score the primary ridge
(and secondary GBM) out-of-fold on the DEV inner folds. 20 draws per k
(1 for k = all). Seed 20260908.

The subset schedule is frozen to intermediate/e43_schedule.npz (row indices
into blind_corpus order per candidate per draw) and the per-draw doc2vec
inference (the only slow featurization) is cached to intermediate/e43_doc2vec/.
07_unseal_evaluate.py refeaturizes from the same schedule + cache and refits
the full-DEV models — deterministic, so the models are frozen by the schedule.

Reads labels only via ws4lib.load_dev_labels().
Outputs: outputs/e43_draws.csv (regime x k x draw x space x learner: DEV OOF r/rho/rmse,
         n candidates with < k tweets), outputs/e43_summary.csv (median / IQR).
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import ws4lib as L

K_GRID = [5, 10, 25, 50, 100, "all"]
N_DRAWS = 20
REGIMES = ("mixed", "originals")
SPACES = L.SPACES + ["m2v_strip"]
LEARNERS = ("ridge", "gbm")
SCHED = L.INTER / "e43_schedule.npz"
D2V_DIR = L.INTER / "e43_doc2vec"
D2V_DIR.mkdir(exist_ok=True)


def key(regime, k, draw) -> str:
    return f"{regime}|{k}|{draw}"


def build_schedule(F: L.Featurizer) -> dict[str, dict[str, np.ndarray]]:
    """Frozen: {key: {candidate_id: rows}}. One rng stream, fixed loop order."""
    rng = np.random.default_rng(L.SEED)
    sched = {}
    for regime in REGIMES:
        orig = regime == "originals"
        for k in K_GRID:
            for d in range(1 if k == "all" else N_DRAWS):
                sched[key(regime, k, d)] = {c: F.subset_rows(c, None if k == "all" else k, rng, originals_only=orig)
                                            for c in F.cands}
    return sched


def save_schedule(sched: dict) -> None:
    flat = {}
    for kk, rows_by in sched.items():
        lens = np.array([len(rows_by[c]) for c in sorted(rows_by)], dtype=np.int32)
        flat[kk + "|rows"] = np.concatenate([rows_by[c] for c in sorted(rows_by)]).astype(np.int32)
        flat[kk + "|lens"] = lens
    np.savez_compressed(SCHED, cands=np.array(sorted(next(iter(sched.values())))), **flat)


def load_schedule() -> dict[str, dict[str, np.ndarray]]:
    z = np.load(SCHED)
    cands = list(z["cands"])
    out = {}
    for name in z.files:
        if name.endswith("|rows"):
            kk = name[:-5]
            rows, lens = z[name], z[kk + "|lens"]
            offs = np.concatenate([[0], np.cumsum(lens)])
            out[kk] = {c: rows[offs[i]:offs[i + 1]] for i, c in enumerate(cands)}
    return out


def featurize(F: L.Featurizer, kk: str, rows_by: dict, full: dict | None = None) -> dict[str, np.ndarray]:
    """All spaces for one draw (910 rows, metadata order). mixed|all == E4.1 features."""
    if kk == key("mixed", "all", 0) and full is not None:
        return {s: full[s] for s in SPACES}
    X = {s: F.features(s, rows_by) for s in ("tfidf", "w2v", "m2v", "m2v_strip", "behav")}
    p = D2V_DIR / (kk.replace("|", "_") + ".npy")
    if p.exists():
        X["doc2vec"] = np.load(p)
    else:
        X["doc2vec"] = F.features("doc2vec", rows_by, n_threads=4)
        np.save(p, X["doc2vec"])
    return X


def _oof(Xd, y, folds, learner):
    pred = L.oof_predictions(Xd, y, folds, learner)
    return L.scores(pred, y)


def main() -> None:
    F = L.Featurizer()
    full = L.load_full_features()
    cands = list(full["candidate_ids"])
    assert cands == F.cands
    splits = L.load_splits()
    labels = L.load_dev_labels()
    meta = L.load_meta().set_index("candidate_id")

    dev_idx = np.array([i for i, c in enumerate(cands) if c in splits["inner_fold"]])
    dev_c = [cands[i] for i in dev_idx]
    folds = np.array([splits["inner_fold"][c] for c in dev_c])
    y = labels.loc[dev_c].to_numpy()

    if SCHED.exists():
        sched = load_schedule()
    else:
        sched = build_schedule(F)
        save_schedule(sched)
    # candidates with fewer than k eligible tweets keep all (reported)
    n_orig = (meta.n_tweets * (1 - meta.share_retweets)).round().astype(int)

    part = L.INTER / "e43_draws.partial.csv"
    rows = pd.read_csv(part, dtype={"k": str}).to_dict("records") if part.exists() else []
    done = {(r["regime"], str(r["k"]), r["draw"]) for r in rows}
    t0 = time.time()
    for regime in REGIMES:
        for k in K_GRID:
            for d in range(1 if k == "all" else N_DRAWS):
                if (regime, str(k), d) in done:
                    continue
                kk = key(regime, k, d)
                rows_by = sched[kk]
                X = featurize(F, kk, rows_by, full)
                short = 0 if k == "all" else int(((meta.n_tweets if regime == "mixed" else n_orig) < k).sum())
                n_empty = int(sum(len(rows_by[c]) == 0 for c in cands))
                tasks = [(s, lr) for s in SPACES for lr in LEARNERS]
                res = Parallel(n_jobs=4)(delayed(_oof)(X[s].astype(np.float64)[dev_idx], y, folds, lr) for s, lr in tasks)
                for (s, lr), sc in zip(tasks, res):
                    rows.append({"regime": regime, "k": str(k), "draw": d, "space": s, "learner": lr, **sc,
                                 "n_short": short, "n_empty": n_empty})
                pd.DataFrame(rows).to_csv(part, index=False)
                r_line = "  ".join(f"{s}={sc['r']:.3f}" for (s, lr), sc in zip(tasks, res) if lr == "ridge")
                print(f"{regime:9s} k={str(k):4s} d={d:2d}  {r_line}  short={short} [{time.time()-t0:.0f}s]", flush=True)

    draws = pd.DataFrame(rows)
    draws["k"] = pd.Categorical(draws.k.astype(str), categories=[str(k) for k in K_GRID], ordered=True)
    draws.to_csv(L.OUT / "e43_draws.csv", index=False)
    summ = (draws.groupby(["regime", "k", "space", "learner"], observed=True)
            .agg(median_r=("r", "median"), q1=("r", lambda s: s.quantile(.25)), q3=("r", lambda s: s.quantile(.75)),
                 median_rmse=("rmse", "median"), n_draws=("r", "size"), n_short=("n_short", "first"))
            .reset_index())
    summ.to_csv(L.OUT / "e43_summary.csv", index=False)
    for regime in REGIMES:
        print(f"\n{regime} — ridge, DEV OOF median r:")
        print(summ[(summ.regime == regime) & (summ.learner == "ridge")]
              .pivot(index="k", columns="space", values="median_r").round(3))


if __name__ == "__main__":
    main()
