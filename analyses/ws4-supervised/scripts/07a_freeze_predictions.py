"""
07a_freeze_predictions.py — label-free TEST predictions for every frozen model
-----------------------------------------------------------------------------
Runs BEFORE the unseal and reads NO labels beyond DEV (via ws4lib.load_dev_labels
for the refits). For every experiment it (re)fits the frozen models on DEV —
from the stored models where 02/05/06 saved them, from the frozen draw schedules
where 03/04/09 froze them (fits are deterministic) — and stores their predictions
for the 303 TEST candidates in intermediate/test_predictions.joblib.
07_unseal_evaluate.py then does nothing but read TEST truth once and score.

Why a separate script (Addendum C): the sandbox caps one call at ~3 min, so 07
could not both refit ~4,000 models and stay a single uninterrupted TEST read;
freezing predictions first also means nothing about any model can change after
the truth is seen. Stages checkpoint to intermediate/e4x_pred_*.joblib.

Stages: A E4.1 probes (+ style projection refit, unsupervised refs)
        B E4.2 label-efficiency refits from outputs/e42_draw_schedule.json
        C E4.3 scarcity refits from intermediate/e43_schedule.npz (+ E4.7(a) k = 25 shift preds)
        D E4.4 shift / control models from intermediate/e44_models.joblib
        E E4.5/E4.6 stacks from intermediate/e45_meta.joblib on E4.1 base preds
        G E4.8 n = 40 picks refit from outputs/e48_n40_picks.json
"""

import importlib
import json
import time

import joblib
import numpy as np
import pandas as pd

import ws4lib as L

E42 = importlib.import_module("03_label_efficiency")
E43 = importlib.import_module("04_text_scarcity")
FINAL = L.INTER / "test_predictions.joblib"


def ck(name):
    return L.INTER / f"e4x_pred_{name}.joblib"


def main() -> None:
    t0 = time.time()
    feats = L.load_full_features()
    cands = list(feats["candidate_ids"])
    splits = L.load_splits()
    labels = L.load_dev_labels()
    meta = L.load_meta().set_index("candidate_id").loc[cands]
    dev_idx = np.array([i for i, c in enumerate(cands) if c in splits["inner_fold"]])
    test_idx = np.array([i for i, c in enumerate(cands) if c in set(splits["test"])])
    dev_c = [cands[i] for i in dev_idx]
    test_c = [cands[i] for i in test_idx]
    assert len(test_c) == 303 and not set(test_c) & set(dev_c)
    folds = np.array([splits["inner_fold"][c] for c in dev_c])
    y = labels.loc[dev_c].to_numpy()
    P = {"candidate_ids": test_c, "dev_ids": dev_c}

    # ---- A: E4.1 ----
    if ck("A").exists():
        P.update(joblib.load(ck("A")))
    else:
        A = {}
        models = joblib.load(L.INTER / "e41_models.joblib")
        for k, m in models.items():
            space = k.split("|")[0]
            A[f"e41|{k}"] = m.predict(feats[space].astype(np.float64)[test_idx])
        rt = meta.share_retweets.to_numpy()
        for space in L.SPACES + ["tfidf_doc", "m2v_strip"]:
            Xp, pc, r = L.style_projected(feats[space].astype(np.float64), rt)
            m = L.fit_ridge(Xp[dev_idx], y, folds)  # same call as 02_probes.py
            A[f"e41|{space}|styleproj|ridge"] = m.predict(Xp[test_idx])
        refs = pd.read_csv(L.OUT / "unsupervised_refs.csv").set_index("candidate_id").loc[test_c]
        A["ref|tfidf_pc1_frozen"] = refs.tfidf_pc1_frozen.to_numpy()
        A["ref|behav_splitA_source_mean"] = refs.behav_splitA_source_mean.to_numpy()
        joblib.dump(A, ck("A"))
        P.update(A)
        print(f"A done [{time.time()-t0:.0f}s]", flush=True)

    # ---- B: E4.2 refits from the frozen schedule ----
    sched = json.loads((L.OUT / "e42_draw_schedule.json").read_text())["draws"]
    dev_pos = {c: i for i, c in enumerate(dev_c)}
    for space in E42.SPACES:
        for learner in E42.LEARNERS:
            name = f"B_{space}_{learner}"
            if ck(name).exists():
                P.update(joblib.load(ck(name)))
                continue
            X = feats[space].astype(np.float64)
            Xd, Xt = X[dev_idx], X[test_idx]
            B = {}
            for n, draws in sched.items():
                for d, ids in enumerate(draws):
                    tr = np.array([dev_pos[c] for c in ids])
                    m = E42.fit_e42(Xd[tr], y[tr], learner)
                    B[f"e42|{space}|{learner}|{n}|{d}"] = m.predict(Xt)
            joblib.dump(B, ck(name))
            P.update(B)
            print(f"B {space} {learner} done [{time.time()-t0:.0f}s]", flush=True)

    # ---- C: E4.3 refits (full-DEV models on k-subset features) + E4.7(a) shift preds ----
    F = None
    sched43 = E43.load_schedule()
    e41_models = joblib.load(L.INTER / "e41_models.joblib")
    for regime in E43.REGIMES:
        for k in E43.K_GRID:
            for d in range(1 if k == "all" else E43.N_DRAWS):
                name = f"C_{regime}_{k}_{d}"  # checkpoint per draw (≈5 s each; calls are time-capped)
                if ck(name).exists():
                    P.update(joblib.load(ck(name)))
                    continue
                if F is None:
                    F = L.Featurizer()
                C = {}
                kk = E43.key(regime, k, d)
                X = E43.featurize(F, kk, sched43[kk], feats)
                for space in E43.SPACES:
                    Xs = X[space].astype(np.float64)
                    for learner in E43.LEARNERS:
                        m = L.fit_ridge(Xs[dev_idx], y, folds) if learner == "ridge" else L.fit_gbm(Xs[dev_idx], y)
                        C[f"e43|{regime}|{k}|{d}|{space}|{learner}"] = m.predict(Xs[test_idx])
                    if regime == "mixed" and k == 25:  # E4.7(a): full-volume model applied to k = 25 TEST features
                        C[f"e47a|{d}|{space}"] = e41_models[f"{space}|ridge"].predict(Xs[test_idx])
                joblib.dump(C, ck(name))
                P.update(C)
            print(f"C {regime} k={k} done [{time.time()-t0:.0f}s]", flush=True)

    # ---- D: E4.4 ----
    if ck("D").exists():
        P.update(joblib.load(ck("D")))
    else:
        D = {}
        for k, m in joblib.load(L.INTER / "e44_models.joblib").items():
            space = k.split("|")[1]
            D[f"e44|{k}"] = m.predict(feats[space].astype(np.float64)[test_idx])
        joblib.dump(D, ck("D"))
        P.update(D)
        print(f"D done [{time.time()-t0:.0f}s]", flush=True)

    # ---- E: E4.5 / E4.6 ----
    if ck("E").exists():
        P.update(joblib.load(ck("E")))
    else:
        E = {}
        base = {s: P[f"e41|{s}|ridge"] for s in L.SPACES}
        llm = pd.read_csv(L.REPO / "analyses" / "ws3-llm-scaling" / "outputs" / "scores_main.csv").set_index("candidate_id").llm_score
        for name, meta_m in joblib.load(L.INTER / "e45_meta.joblib").items():
            cols = [base[s] if s != "llm" else llm.reindex(test_c).to_numpy() for s in meta_m["inputs"]]
            Pm = np.column_stack(cols)
            pred = np.full(len(test_c), np.nan)
            ok = ~np.isnan(Pm).any(axis=1)
            pred[ok] = meta_m["model"].predict(Pm[ok])
            E[f"e45|{name}"] = pred
        E["e46|llm_alone"] = llm.reindex(test_c).to_numpy()
        joblib.dump(E, ck("E"))
        P.update(E)
        print(f"E done [{time.time()-t0:.0f}s]", flush=True)

    # ---- G: E4.8 n = 40 picks ----
    if ck("G").exists():
        P.update(joblib.load(ck("G")))
    else:
        G = {}
        picks = json.loads((L.OUT / "e48_n40_picks.json").read_text())
        for key, pk in picks.items():
            space = key.split("|")[0]
            X = feats[space].astype(np.float64)
            tr = np.array([dev_pos[c] for c in pk["labeled"]])
            assert len(tr) == 40
            m = L.fit_ridge_al(X[dev_idx][tr], y[tr])
            G[f"e48|{key}"] = m.predict(X[test_idx])
        joblib.dump(G, ck("G"))
        P.update(G)
        print(f"G done [{time.time()-t0:.0f}s]", flush=True)

    P["frozen_on"] = pd.Timestamp.now().isoformat()
    joblib.dump(P, FINAL)
    print(f"frozen {len(P)} prediction arrays → {FINAL.name}")


if __name__ == "__main__":
    main()
