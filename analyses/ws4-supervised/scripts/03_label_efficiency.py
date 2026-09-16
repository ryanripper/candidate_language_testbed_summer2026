"""
03_label_efficiency.py — E4.2 label efficiency (DEV only; TEST untouched)
------------------------------------------------------------------------
For n in {10, 20, 40, 80, 160, 320, all-DEV}: draw n DEV candidates
(stratified by party, largest-remainder allocation), fit the primary ridge
(and the secondary GBM), score the *remaining* DEV candidates (out-of-draw).
50 draws per n per space; the all-DEV point is the E4.1 inner-fold OOF r.
Seed 20260908 (preregistration §2); the draw schedule is frozen to
outputs/e42_draw_schedule.json and 07_unseal_evaluate.py refits from it —
fits are deterministic (RidgeCV over fixed KFold, GBM random_state fixed),
so freezing the schedule freezes the models.

n* (DEV, development number) = smallest n whose median out-of-draw r is
within .02 of the all-DEV OOF r for that space (preregistration §5).

Reads labels only via ws4lib.load_dev_labels().
Outputs: outputs/e42_draws.csv (one row per space x learner x n x draw),
         outputs/e42_summary.csv (median / IQR / n*),
         outputs/e42_draw_schedule.json (frozen candidate ids per draw)
"""

import json
import time

import numpy as np
import pandas as pd

import ws4lib as L

N_GRID = [10, 20, 40, 80, 160, 320]
N_DRAWS = 50
SPACES = L.SPACES + ["m2v_strip"]
LEARNERS = ("ridge", "gbm")
TOL = 0.02


def stratified_draw(strata: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """n row indices, proportional-to-stratum with largest-remainder rounding."""
    keys, counts = np.unique(strata, return_counts=True)
    quota = counts / counts.sum() * n
    take = np.floor(quota).astype(int)
    rem = n - take.sum()
    if rem > 0:
        take[np.argsort(-(quota - take))[:rem]] += 1
    out = []
    for key, t in zip(keys, take):
        idx = np.flatnonzero(strata == key)
        out.extend(rng.choice(idx, size=min(t, len(idx)), replace=False).tolist())
    return np.array(sorted(out))


def build_schedule(dev_c: list[str], party: np.ndarray) -> dict:
    """Frozen draw schedule: {n: [[candidate_id, ...] x N_DRAWS]}; same draws for every space/learner."""
    rng = np.random.default_rng(L.SEED)
    sched = {}
    for n in N_GRID:
        sched[str(n)] = [[dev_c[i] for i in stratified_draw(party, n, rng)] for _ in range(N_DRAWS)]
    return sched


def fit_e42(X: np.ndarray, y: np.ndarray, learner: str):
    """The one fit path shared with 07 (α by 5-fold KFold inside the draw; GBM seeded)."""
    return L.fit_ridge(X, y) if learner == "ridge" else L.fit_gbm(X, y)


def main() -> None:
    feats = L.load_full_features()
    cands = list(feats["candidate_ids"])
    splits = L.load_splits()
    labels = L.load_dev_labels()
    meta = L.load_meta().set_index("candidate_id")

    dev_c = [c for c in cands if c in splits["inner_fold"]]
    dev_pos = {c: i for i, c in enumerate(dev_c)}
    y = labels.loc[dev_c].to_numpy()
    party = meta.loc[dev_c, "party"].to_numpy()

    sched = build_schedule(dev_c, party)
    (L.OUT / "e42_draw_schedule.json").write_text(json.dumps(
        {"seed": L.SEED, "n_grid": N_GRID, "n_draws": N_DRAWS, "stratify": "party (largest remainder)",
         "draws": sched}, indent=0))

    e41 = pd.read_csv(L.OUT / "e41_dev_summary.csv")
    full_r = {(r.space, r.learner): r.r for r in e41.itertuples() if r.variant == "raw"}

    # resumable: per-(space, learner) blocks are appended to the partial file as they finish
    part = L.INTER / "e42_draws.partial.csv"  # gitignored checkpoint (sandbox calls are time-capped; sandbox cannot delete files)
    rows = pd.read_csv(part).to_dict("records") if part.exists() else []
    done = {(r["space"], r["learner"], r["n"]) for r in rows}
    t0 = time.time()
    for space in SPACES:
        X = feats[space].astype(np.float64)[[cands.index(c) for c in dev_c]]
        for learner in LEARNERS:
            for n in N_GRID:
                if (space, learner, n) in done:
                    continue
                for d, ids in enumerate(sched[str(n)]):
                    tr = np.array([dev_pos[c] for c in ids])
                    te = np.setdiff1d(np.arange(len(dev_c)), tr)
                    m = fit_e42(X[tr], y[tr], learner)
                    s = L.scores(m.predict(X[te]), y[te])
                    rows.append({"space": space, "learner": learner, "n": n, "draw": d, **s})
                rr = [r["r"] for r in rows if r["space"] == space and r["learner"] == learner and r["n"] == n]
                med, n_nan = np.nanmedian(rr), int(np.sum(np.isnan(rr)))
                print(f"{space:10s} {learner:5s} n={n:4d}  median r={med:.4f}  nan={n_nan:2d}  [{time.time()-t0:.0f}s]", flush=True)
                pd.DataFrame(rows).to_csv(part, index=False)
            if (space, learner, len(dev_c)) in done:
                continue
            rows.append({"space": space, "learner": learner, "n": len(dev_c), "draw": 0,
                         "r": full_r[(space, learner)], "rho": np.nan, "rmse": np.nan})
            pd.DataFrame(rows).to_csv(part, index=False)

    draws = pd.DataFrame(rows)
    draws.to_csv(L.OUT / "e42_draws.csv", index=False)

    summ = []
    for (space, learner), g in draws.groupby(["space", "learner"]):
        ref = full_r[(space, learner)]
        n_star = None
        for n in N_GRID:
            gg = g[g.n == n].r
            q1, med, q3 = np.nanpercentile(gg, [25, 50, 75])  # GBM at n <= 20 can be constant (early-stopping val split of 1–2 rows)
            summ.append({"space": space, "learner": learner, "n": n, "median_r": med, "q1": q1, "q3": q3,
                         "gap_to_full": ref - med, "n_degenerate": int(gg.isna().sum())})
            if n_star is None and ref - med <= TOL:
                n_star = n
        summ.append({"space": space, "learner": learner, "n": len(dev_c), "median_r": ref, "q1": ref, "q3": ref,
                     "gap_to_full": 0.0})
        for row in summ:
            if row["space"] == space and row["learner"] == learner:
                row["n_star_dev"] = n_star if n_star is not None else len(dev_c)
    summ = pd.DataFrame(summ)
    summ.to_csv(L.OUT / "e42_summary.csv", index=False)

    print("\nDEV n* (development numbers; TEST n* comes from 07):")
    print(summ.drop_duplicates(["space", "learner"])[["space", "learner", "n_star_dev"]].to_string(index=False))
    print(summ[summ.learner == "ridge"].pivot(index="n", columns="space", values="median_r").round(3))


if __name__ == "__main__":
    main()
