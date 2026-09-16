"""
06_ensemble.py — E4.5 cross-family stacking + E4.6 LLM-as-feature (DEV only)
----------------------------------------------------------------------------
Meta-learner: non-negative ridge (`Ridge(positive=True, fit_intercept=True)`),
α on the §2 grid chosen by the frozen DEV inner folds, over the out-of-fold
single-space ridge predictions from 02_probes.py (outputs/e41_dev_oof.csv).
Three stacks (preregistration §2): content {tfidf, doc2vec}; style/behavioral
{w2v, m2v, behav}; cross-family {all five}.

DEV number = second-level OOF over the same inner folds (meta fit on four
folds' base-OOF predictions, applied to the fifth). The base predictions are
themselves OOF, so this is the usual stacking estimate — mildly optimistic
because fold-f base predictions came from models that saw the other four
folds' labels; stated, not corrected. The cross-family stack's OOF
predictions are also the E4.7 calibration scores for `stack_all`
(Addendum B §B.1 / §B.4).

E4.6 (secondary, n = 150 support): on DEV ∩ subsample_150 (n = 98) the
cross-family stack is refit with and without the WS3 `llm_score` as a sixth
input, inner-CV within the 98 using the frozen fold assignment; the LLM
score alone is reported too. TEST ∩ 150 (n = 52) is descriptive in 07.

Final meta models (fit on all DEV) → intermediate/e45_meta.joblib; 07 feeds
them the full-DEV base models' TEST predictions from e41_models.joblib.
Reads labels only via ws4lib.load_dev_labels().
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

import ws4lib as L

STACKS = {"stack_content": ["tfidf", "doc2vec"], "stack_style": ["w2v", "m2v", "behav"], "stack_all": L.SPACES}


def fit_meta(P: np.ndarray, y: np.ndarray, folds: np.ndarray):
    """Non-negative ridge with α by CV over the given folds (RidgeCV has no `positive`)."""
    best = None
    for a in L.ALPHAS:
        err = 0.0
        for f in np.unique(folds):
            tr, te = folds != f, folds == f
            m = Ridge(alpha=a, positive=True, fit_intercept=True).fit(P[tr], y[tr])
            err += np.sum((m.predict(P[te]) - y[te]) ** 2)
        if best is None or err < best[0]:
            best = (err, a)
    m = Ridge(alpha=best[1], positive=True, fit_intercept=True).fit(P, y)
    m.alpha_ = best[1]
    return m


def meta_oof(P: np.ndarray, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    pred = np.full(len(y), np.nan)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        m = fit_meta(P[tr], y[tr], folds[tr])
        pred[te] = m.predict(P[te])
    return pred


def main() -> None:
    oof = pd.read_csv(L.OUT / "e41_dev_oof.csv")
    y = oof.true_ideology.to_numpy()
    folds = oof.inner_fold.to_numpy()
    base = {s: oof[f"{s}|raw|ridge"].to_numpy() for s in L.SPACES}

    rows, metas, weights = [], {}, {}
    stack_oof = oof[["candidate_id", "true_ideology", "inner_fold"]].copy()
    for name, spaces in STACKS.items():
        P = np.column_stack([base[s] for s in spaces])
        pred = meta_oof(P, y, folds)
        stack_oof[name] = pred
        s = L.scores(pred, y)
        m = fit_meta(P, y, folds)
        metas[name] = {"model": m, "inputs": spaces}
        weights[name] = {"alpha": float(m.alpha_), "intercept": float(m.intercept_),
                         **{sp: float(w) for sp, w in zip(spaces, m.coef_)}}
        rows.append({"stack": name, "support": "DEV", "n": len(y), **s})
        print(f"{name:14s} DEV meta-OOF r={s['r']:.4f} rho={s['rho']:.4f} rmse={s['rmse']:.3f}  weights="
              + " ".join(f"{sp}:{w:.2f}" for sp, w in zip(spaces, m.coef_)) + f"  α={m.alpha_:g}", flush=True)
    for s_ in L.SPACES:
        rows.append({"stack": s_, "support": "DEV", "n": len(y), **L.scores(base[s_], y)})

    # ---- E4.6: LLM score as a sixth input, n = 150 support ----
    ss = json.loads((L.WS0 / "splits.json").read_text())["subsample_150"]["candidate_ids"]
    llm = pd.read_csv(L.REPO / "analyses" / "ws3-llm-scaling" / "outputs" / "scores_main.csv").set_index("candidate_id").llm_score
    in150 = oof.candidate_id.isin(ss).to_numpy()
    y1, f1 = y[in150], folds[in150]
    P5 = np.column_stack([base[s] for s in L.SPACES])[in150]
    P6 = np.column_stack([P5, llm.loc[oof.candidate_id[in150]].to_numpy()])
    for name, P in (("stack_all@150", P5), ("stack_all+llm@150", P6)):
        pred = meta_oof(P, y1, f1)
        m = fit_meta(P, y1, f1)
        inputs = L.SPACES + (["llm"] if P.shape[1] == 6 else [])
        metas[name] = {"model": m, "inputs": inputs}
        weights[name] = {"alpha": float(m.alpha_), "intercept": float(m.intercept_),
                         **{sp: float(w) for sp, w in zip(inputs, m.coef_)}}
        s = L.scores(pred, y1)
        rows.append({"stack": name, "support": "DEV∩150", "n": int(in150.sum()), **s})
        print(f"{name:18s} DEV∩150 meta-OOF r={s['r']:.4f}  weights=" + " ".join(f"{sp}:{w:.2f}" for sp, w in zip(inputs, m.coef_)))
    s = L.scores(llm.loc[oof.candidate_id[in150]].to_numpy(), y1)
    rows.append({"stack": "llm_alone@150", "support": "DEV∩150", "n": int(in150.sum()), **s})
    print(f"{'llm_alone@150':18s} DEV∩150 r={s['r']:.4f}  (WS3 pilot on all 150 was .970)")
    for s_ in L.SPACES:
        rows.append({"stack": f"{s_}@150", "support": "DEV∩150", "n": int(in150.sum()), **L.scores(base[s_][in150], y1)})

    pd.DataFrame(rows).to_csv(L.OUT / "e45_dev_summary.csv", index=False)
    stack_oof.to_csv(L.OUT / "e45_dev_stack_oof.csv", index=False)
    (L.OUT / "e45_meta_weights.json").write_text(json.dumps(weights, indent=1))
    joblib.dump(metas, L.INTER / "e45_meta.joblib")


if __name__ == "__main__":
    main()
