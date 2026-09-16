"""
09_active_learning.py — E4.8 pool-based active learning (DEV only; Addendum B §B.2)
----------------------------------------------------------------------------------
Inside each frozen DEV inner fold: pool = the four training folds (~485),
evaluation = the held-out fold (~121). Spaces tfidf, doc2vec, m2v_strip, behav.
Learner = primary ridge with α by 3-fold CV inside the labeled set
(ws4lib.fit_ridge_al). Protocol: seed set n₀ = 10 stratified on party × chamber,
batch 5, stop at n = 60; 20 draws per fold → 100 curves per (space, strategy).
Strategies (fixed): random / uncertainty / diversity / hybrid (§B.2).

Seeds: rng = default_rng([SEED_B + draw, fold]) — one stream per (draw, fold),
shared by all strategies and spaces, so the seed set is paired across
strategies (the `random` control is a recomputation under SEED_B, §B.4(2)).
Diversity's k-center runs in the pool-standardized feature space (label-free).

Metrics: learning curves (median / IQR r over the 100 curves), n*_AL (smallest n
with median r within .02 of the space's full-DEV E4.1 OOF r — same definition as
E4.2), AULC over n ∈ [10, 60] with a paired bootstrap over curves vs `random`
(2,000 resamples, seed SEED_B). The n = 40 pick per (space, strategy) = the curve
whose r at n = 40 is closest to the median; its labeled candidate ids are frozen
to outputs/e48_n40_picks.json and 07 refits on them (deterministic) for one
descriptive TEST r each.

Checkpoints per (space, fold) in intermediate/ (sandbox calls are time-capped).
Reads labels only via ws4lib.load_dev_labels().
Outputs: outputs/e48_curves.csv, outputs/e48_summary.csv, outputs/e48_aulc.csv,
         outputs/e48_n40_picks.json
"""

import json
import time

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler

import ws4lib as L

N_STEPS = list(range(L.AL_N0, L.AL_NMAX + 1, L.AL_BATCH))  # 10, 15, ..., 60
TOL = 0.02


def run_draw(Xp: np.ndarray, yp: np.ndarray, Xe: np.ndarray, ye: np.ndarray, strata: np.ndarray,
             pool_ids: list[str], draw: int, fold: int) -> list[dict]:
    """All four strategies for one (draw, fold) on one space; shared seed set."""
    rng = np.random.default_rng([L.SEED_B + draw, fold])
    seed = L.stratified_seed_set(strata, np.arange(len(yp)), L.AL_N0, rng)
    Z = StandardScaler().fit_transform(Xp)  # label-free standardization for k-center
    out = []
    for strat in L.AL_STRATEGIES:
        srng = np.random.default_rng([L.SEED_B + draw, fold, L.AL_STRATEGIES.index(strat)])
        lab = np.zeros(len(yp), bool)
        lab[seed] = True
        history = [pool_ids[i] for i in seed]
        step = 0
        while True:
            n = int(lab.sum())
            m = L.fit_ridge_al(Xp[lab], yp[lab])
            r = float(np.corrcoef(m.predict(Xe), ye)[0, 1])
            out.append({"strategy": strat, "draw": draw, "fold": fold, "n": n, "r": r,
                        "alpha": float(m.named_steps["ridgecv"].alpha_),
                        "labeled": list(history) if n == 40 else None})
            if n >= L.AL_NMAX:
                break
            unl = np.flatnonzero(~lab)
            if strat == "random":
                pick = srng.choice(unl, size=L.AL_BATCH, replace=False)
            elif strat == "uncertainty" or (strat == "hybrid" and step >= L.AL_HYBRID_SWITCH):
                v = L.ridge_predictive_variance(m, Xp[unl])
                pick = unl[np.argsort(-v)[:L.AL_BATCH]]
            else:  # diversity, or hybrid's first AL_HYBRID_SWITCH batches
                pick = L.kcenter_greedy(Z, lab, L.AL_BATCH)
            lab[pick] = True
            history.extend(pool_ids[i] for i in pick)
            step += 1
    return out


def main() -> None:
    feats = L.load_full_features()
    cands = list(feats["candidate_ids"])
    splits = L.load_splits()
    labels = L.load_dev_labels()
    meta = L.load_meta().set_index("candidate_id")
    e41 = pd.read_csv(L.OUT / "e41_dev_summary.csv")
    full_r = {r.space: r.r for r in e41.itertuples() if r.variant == "raw" and r.learner == "ridge"}

    dev_c = [c for c in cands if c in splits["inner_fold"]]
    dev_idx = np.array([cands.index(c) for c in dev_c])
    folds = np.array([splits["inner_fold"][c] for c in dev_c])
    y = labels.loc[dev_c].to_numpy()
    strata_all = (meta.loc[dev_c, "party"] + "|" + meta.loc[dev_c, "chamber"]).to_numpy()

    rows = []
    t0 = time.time()
    for space in L.AL_SPACES:
        X = feats[space].astype(np.float64)[dev_idx]
        for f in np.unique(folds):
            ck = L.INTER / f"e48_partial_{space}_{f}.joblib"
            if ck.exists():
                rows.extend(joblib.load(ck))
                continue
            pool, ev = folds != f, folds == f
            pool_ids = [c for c, p in zip(dev_c, pool) if p]
            res = Parallel(n_jobs=4)(delayed(run_draw)(X[pool], y[pool], X[ev], y[ev], strata_all[pool], pool_ids, d, int(f))
                                     for d in range(L.AL_DRAWS))
            block = [dict(r, space=space) for rr in res for r in rr]
            joblib.dump(block, ck)
            rows.extend(block)
            df = pd.DataFrame(block)
            med = df[df.n.isin([15, 25, 40])].groupby(["strategy", "n"]).r.median().unstack()
            print(f"{space:10s} fold {f}  median r @n=15/25/40:\n{med.round(3)}  [{time.time()-t0:.0f}s]", flush=True)

    curves = pd.DataFrame(rows)
    picks = {}
    for (space, strat), g in curves[curves.n == 40].groupby(["space", "strategy"]):
        med = g.r.median()
        row = g.iloc[(g.r - med).abs().argmin()]
        picks[f"{space}|{strat}"] = {"draw": int(row.draw), "fold": int(row.fold), "dev_r_at_40": float(row.r),
                                     "median_r_at_40": float(med), "labeled": row.labeled}
    (L.OUT / "e48_n40_picks.json").write_text(json.dumps(picks, indent=1))
    curves.drop(columns=["labeled"]).to_csv(L.OUT / "e48_curves.csv", index=False)

    summ, aulc_rows = [], []
    rng = np.random.default_rng(L.SEED_B)
    for space in L.AL_SPACES:
        cs = curves[curves.space == space]
        # AULC per curve (trapezoid over n), indexed by (draw, fold) so strategies pair up
        au = {}
        for strat in L.AL_STRATEGIES:
            g = cs[cs.strategy == strat]
            n_star = None
            for n in N_STEPS:
                q1, med, q3 = np.percentile(g[g.n == n].r, [25, 50, 75])
                summ.append({"space": space, "strategy": strat, "n": n, "median_r": med, "q1": q1, "q3": q3,
                             "gap_to_full": full_r[space] - med})
                if n_star is None and full_r[space] - med <= TOL:
                    n_star = n
            for row in summ:
                if row["space"] == space and row["strategy"] == strat:
                    row["n_star_al"] = n_star if n_star is not None else -1
            piv = g.pivot_table(index=["draw", "fold"], columns="n", values="r").sort_index()
            au[strat] = np.trapz(piv[N_STEPS].to_numpy(), N_STEPS, axis=1) / (N_STEPS[-1] - N_STEPS[0])
        idx_all = np.arange(len(au["random"]))
        for strat in L.AL_STRATEGIES:
            d = au[strat] - au["random"]
            boots = [np.mean(d[rng.integers(0, len(d), len(d))]) for _ in range(2000)]
            aulc_rows.append({"space": space, "strategy": strat, "aulc": float(au[strat].mean()),
                              "d_vs_random": float(d.mean()), "lo": float(np.percentile(boots, 2.5)),
                              "hi": float(np.percentile(boots, 97.5)), "n_curves": len(d)})
    summ = pd.DataFrame(summ)
    summ.to_csv(L.OUT / "e48_summary.csv", index=False)
    aulc = pd.DataFrame(aulc_rows)
    aulc.to_csv(L.OUT / "e48_aulc.csv", index=False)

    print("\nn*_AL (DEV; -1 = not reached by n = 60) and AULC Δ vs random:")
    tab = summ.drop_duplicates(["space", "strategy"])[["space", "strategy", "n_star_al"]].merge(
        aulc[["space", "strategy", "aulc", "d_vs_random", "lo", "hi"]])
    print(tab.round(4).to_string(index=False))
    for space in L.AL_SPACES:
        print(f"\n{space} median r by n:")
        print(summ[summ.space == space].pivot(index="n", columns="strategy", values="median_r").round(3).T)


if __name__ == "__main__":
    main()
