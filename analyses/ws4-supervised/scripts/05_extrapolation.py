"""
05_extrapolation.py — E4.4 incumbent→challenger (and House→Senate) shift (DEV only)
----------------------------------------------------------------------------------
Shift-trained models: ridge (primary) and GBM (secondary) fit on DEV incumbents
only (n = 246; α by inner CV restricted to incumbents) and on DEV House only
(n = 561). Controls: 50 random DEV subsets of the same size (seed 20260908),
same learners. On DEV the shift-trained model is scored on DEV challengers
(resp. DEV senators) and each control on the DEV candidates it did not train
on — development numbers. The preregistered Δr = r(control) − r(shift-trained)
with paired-bootstrap CI is computed on TEST challengers / senators in 07,
using the models frozen here (intermediate/e44_models.joblib) and the control
subsets frozen in outputs/e44_control_schedule.json.

Also writes outputs/e44_incumbent_oof.csv: inner-fold OOF ridge predictions
within DEV incumbents — the calibration residuals Addendum B §B.4(1) assigns
to E4.7(b) (incumbent-calibrated q̂ applied to TEST challengers).

Independents (all challengers) are reported separately and never trained on
alone. Reads labels only via ws4lib.load_dev_labels().
"""

import json
import time

import joblib
import numpy as np
import pandas as pd

import ws4lib as L

N_CTRL = 50
SPACES = L.SPACES + ["m2v_strip"]
LEARNERS = ("ridge", "gbm")
AXES = {"incumbency": ("incumbent", True, "challengers"), "chamber": ("chamber", "House", "senators")}


def fit(X, y, learner, folds=None):
    return L.fit_ridge(X, y, folds) if learner == "ridge" else L.fit_gbm(X, y)


def control_delta_r(pred_shift: np.ndarray, pred_ctrls: np.ndarray, truth: np.ndarray,
                    n_boot: int = 2000, seed: int = L.SEED) -> dict:
    """Δr = median_c r(control_c) − r(shift-trained) on the same candidates, with a
    paired percentile-bootstrap 95 % CI over candidates (the median over controls is
    recomputed inside every resample). pred_ctrls: (n_controls, n_candidates)."""
    rng = np.random.default_rng(seed)
    n = len(truth)

    def stat(idx):
        t = truth[idx]
        rs = np.array([np.corrcoef(p[idx], t)[0, 1] for p in pred_ctrls])
        return float(np.median(rs) - np.corrcoef(pred_shift[idx], t)[0, 1])

    point = stat(np.arange(n))
    boots = np.array([stat(rng.integers(0, n, n)) for _ in range(n_boot)])
    return {"dr": point, "lo": float(np.percentile(boots, 2.5)), "hi": float(np.percentile(boots, 97.5))}


def main() -> None:
    feats = L.load_full_features()
    cands = list(feats["candidate_ids"])
    splits = L.load_splits()
    labels = L.load_dev_labels()
    meta = L.load_meta().set_index("candidate_id")

    dev_c = [c for c in cands if c in splits["inner_fold"]]
    dev_idx = np.array([cands.index(c) for c in dev_c])
    folds = np.array([splits["inner_fold"][c] for c in dev_c])
    y = labels.loc[dev_c].to_numpy()
    md = meta.loc[dev_c]
    is_ind = (md.party == "I").to_numpy()

    rng = np.random.default_rng(L.SEED)
    schedule, models, rows = {}, {}, []
    oof_inc = pd.DataFrame({"candidate_id": np.array(dev_c)[(md.incumbent == True).to_numpy()]})
    t0 = time.time()
    # checkpoint per (axis, space) — sandbox calls are time-capped: intermediate/e44_partial_{axis}_{space}.joblib
    for axis, (col, train_val, eval_name) in AXES.items():
        train_mask = (md[col] == train_val).to_numpy()
        n_tr = int(train_mask.sum())
        ctrl_sets = [np.sort(rng.choice(len(dev_c), size=n_tr, replace=False)) for _ in range(N_CTRL)]  # rng stream fixed regardless of resume
        eval_mask = ~train_mask
        schedule[axis] = {"train_on": f"{col} == {train_val}", "n_train": n_tr,
                          "controls": [[dev_c[i] for i in s] for s in ctrl_sets]}
        for space in SPACES:
            ck = L.INTER / f"e44_partial_{axis}_{space}.joblib"
            if ck.exists():
                part = joblib.load(ck)
                rows.extend(part["rows"]); models.update(part["models"])
                for c, v in part["oof_inc"].items():
                    oof_inc[c] = v
                continue
            rows_ax, models_ax, oof_ax = [], {}, {}
            X = feats[space].astype(np.float64)[dev_idx]
            for learner in LEARNERS:
                m = fit(X[train_mask], y[train_mask], learner, folds[train_mask] if learner == "ridge" else None)
                models_ax[f"{axis}|{space}|{learner}"] = m
                p = m.predict(X[eval_mask])
                s = L.scores(p, y[eval_mask])
                rows_ax.append({"axis": axis, "space": space, "learner": learner, "model": "shift", "draw": -1,
                                "n_train": n_tr, "n_eval": int(eval_mask.sum()), **s})
                if axis == "incumbency":
                    ni = eval_mask & ~is_ind
                    rows_ax.append({"axis": axis, "space": space, "learner": learner, "model": "shift_nonind", "draw": -1,
                                    "n_train": n_tr, "n_eval": int(ni.sum()), **L.scores(m.predict(X[ni]), y[ni])})
                    ii = eval_mask & is_ind
                    rows_ax.append({"axis": axis, "space": space, "learner": learner, "model": "shift_ind", "draw": -1,
                                    "n_train": n_tr, "n_eval": int(ii.sum()), **L.scores(m.predict(X[ii]), y[ii])})
                    if learner == "ridge":  # E4.7(b) calibration residuals: OOF within DEV incumbents
                        oof_ax[f"{space}|ridge"] = L.oof_predictions(X[train_mask], y[train_mask], folds[train_mask])
                for d, s_idx in enumerate(ctrl_sets):
                    tr = np.zeros(len(dev_c), bool)
                    tr[s_idx] = True
                    mc = fit(X[tr], y[tr], learner)
                    models_ax[f"{axis}|{space}|{learner}|ctrl{d}"] = mc
                    ev = ~tr
                    rows_ax.append({"axis": axis, "space": space, "learner": learner, "model": "control", "draw": d,
                                    "n_train": n_tr, "n_eval": int(ev.sum()), **L.scores(mc.predict(X[ev]), y[ev])})
                rc = [r["r"] for r in rows_ax if r["learner"] == learner and r["model"] == "control"]
                print(f"{axis:10s} {space:10s} {learner:5s}  shift→{eval_name} r={s['r']:.4f}   "
                      f"control (held-out DEV) median r={np.median(rc):.4f}  [{time.time()-t0:.0f}s]", flush=True)
            rows.extend(rows_ax); models.update(models_ax)
            for c, v in oof_ax.items():
                oof_inc[c] = v
            joblib.dump({"rows": rows_ax, "models": models_ax, "oof_inc": oof_ax}, ck)

    oof_inc["true_ideology"] = y[(md.incumbent == True).to_numpy()]
    oof_inc.to_csv(L.OUT / "e44_incumbent_oof.csv", index=False)
    pd.DataFrame(rows).to_csv(L.OUT / "e44_dev.csv", index=False)
    (L.OUT / "e44_control_schedule.json").write_text(json.dumps({"seed": L.SEED, "n_controls": N_CTRL, **schedule}, indent=0))
    joblib.dump(models, L.INTER / "e44_models.joblib")

    df = pd.DataFrame(rows)
    print("\nDEV development numbers (ridge). TEST Δr with CI comes from 07.")
    for axis in AXES:
        sub = df[(df.axis == axis) & (df.learner == "ridge")]
        tab = sub[sub.model == "shift"].set_index("space").r.rename("shift_r").to_frame()
        tab["ctrl_median_r"] = sub[sub.model == "control"].groupby("space").r.median()
        if axis == "incumbency":
            tab["ind_r"] = sub[sub.model == "shift_ind"].set_index("space").r
        print(f"\n{axis}:\n{tab.round(3)}")


if __name__ == "__main__":
    main()
