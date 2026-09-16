"""
08_conformal.py — E4.7 split-conformal calibration (DEV only; Addendum B §B.1)
-----------------------------------------------------------------------------
Calibration scores = frozen inner-fold OOF absolute residuals |y − ŷ| on DEV:
  * single spaces (tfidf, doc2vec, w2v, m2v, m2v_strip, behav): outputs/e41_dev_oof.csv
    columns `{space}|raw|ridge` (02_probes.py, unchanged since 2026-09-08);
  * cross-family stack: outputs/e45_dev_stack_oof.csv column `stack_all` (06);
  * E4.7(b) incumbent-calibrated variant (§B.4(1)): outputs/e44_incumbent_oof.csv —
    inner-fold OOF residuals of the incumbent-trained ridge within DEV incumbents.
q̂_α = ⌈(n+1)(1−α)⌉/n empirical quantile, α ∈ {.10, .05}. The q̂ table is
frozen to outputs/e47_qhat.json; 07 applies it to TEST predictions of the
full-DEV models (e41_models / e45_meta / e44_models) — ŷ ± q̂.

DEV sanity check (not the bar): coverage of the OOF intervals on DEV itself,
which is in-sample for the quantile and therefore ≈ 1 − α by construction —
reported only to catch bugs. Also reports the leave-one-fold-out DEV coverage
(q̂ from four folds, coverage on the fifth), which is a fair pre-unseal
estimate. Seed SEED_B is not needed here (no draws); reserved for 09.

Reads labels only via ws4lib.load_dev_labels() (through the OOF files).
"""

import json

import numpy as np
import pandas as pd

import ws4lib as L


def main() -> None:
    oof = pd.read_csv(L.OUT / "e41_dev_oof.csv")
    stack = pd.read_csv(L.OUT / "e45_dev_stack_oof.csv")
    assert (oof.candidate_id == stack.candidate_id).all()
    inc = pd.read_csv(L.OUT / "e44_incumbent_oof.csv")
    y = oof.true_ideology.to_numpy()
    folds = oof.inner_fold.to_numpy()
    sd_y = float(np.std(y))

    preds = {s: oof[f"{s}|raw|ridge"].to_numpy() for s in L.CONFORMAL_SPACES}
    preds["stack_all"] = stack["stack_all"].to_numpy()

    qhat, rows = {}, []
    for name, p in preds.items():
        resid = np.abs(y - p)
        qhat[name] = {}
        for a in L.CONFORMAL_ALPHAS:
            q = L.conformal_quantile(resid, a)
            qhat[name][str(a)] = q
            hit_in = int((resid <= q).sum())
            # leave-one-fold-out coverage: q̂ from the other four folds' residuals
            lofo = []
            for f in np.unique(folds):
                qf = L.conformal_quantile(resid[folds != f], a)
                lofo.append((resid[folds == f] <= qf).mean())
            lo, hi = L.coverage_ci(int(round(np.mean(lofo) * len(y))), len(y))
            rows.append({"model": name, "alpha": a, "qhat": q, "width_rel_sd": 2 * q / sd_y,
                         "dev_insample_coverage": hit_in / len(y),
                         "dev_lofo_coverage": float(np.mean(lofo)), "lofo_ci_lo": lo, "lofo_ci_hi": hi,
                         "n_calib": len(y)})
            print(f"{name:10s} α={a:.2f}  q̂={q:.3f}  width/SD={2*q/sd_y:.2f}  DEV in-sample {hit_in/len(y):.3f}  "
                  f"LOFO {np.mean(lofo):.3f}", flush=True)

    # E4.7(b): incumbent-trained ridge, calibrated on DEV-incumbent OOF residuals (§B.4(1))
    yi = inc.true_ideology.to_numpy()
    qhat_inc = {}
    for s in L.CONFORMAL_SPACES:
        resid = np.abs(yi - inc[f"{s}|ridge"].to_numpy())
        qhat_inc[s] = {str(a): L.conformal_quantile(resid, a) for a in L.CONFORMAL_ALPHAS}
        for a in L.CONFORMAL_ALPHAS:
            q = qhat_inc[s][str(a)]
            rows.append({"model": f"{s}@incumbent", "alpha": a, "qhat": q, "width_rel_sd": 2 * q / sd_y,
                         "dev_insample_coverage": float((resid <= q).mean()), "n_calib": len(yi)})
        print(f"{s:10s} incumbent-calibrated q̂: " + "  ".join(f"α={a}: {qhat_inc[s][str(a)]:.3f}" for a in L.CONFORMAL_ALPHAS))

    (L.OUT / "e47_qhat.json").write_text(json.dumps({
        "method": "split conformal, calibration = DEV inner-fold OOF |resid|, finite-sample quantile (Addendum B §B.1)",
        "alphas": list(L.CONFORMAL_ALPHAS), "n_calib_all_dev": len(y), "n_calib_incumbents": len(yi),
        "sd_true_ideology_dev": sd_y, "qhat": qhat, "qhat_incumbent_calibrated": qhat_inc}, indent=1))
    pd.DataFrame(rows).to_csv(L.OUT / "e47_dev_calibration.csv", index=False)


if __name__ == "__main__":
    main()
