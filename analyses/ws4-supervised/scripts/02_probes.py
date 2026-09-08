"""
02_probes.py — E4.1 development run (DEV inner folds only; TEST untouched)
--------------------------------------------------------------------------
Out-of-fold predictions on the 607 DEV candidates for every space x learner,
with and without the style-PC projection. Fits the final full-DEV models
too and stores them for 07_unseal_evaluate.py, so no model is re-chosen
after the unseal. Also reports the unsupervised references on DEV.

Reads labels from ws4_dev_labels.parquet only.
Outputs: outputs/e41_dev_oof.csv (per-candidate OOF preds),
         outputs/e41_dev_summary.csv, intermediate/e41_models.joblib
"""

import joblib
import numpy as np
import pandas as pd

import ws4lib as L


def main() -> None:
    feats = L.load_full_features()
    cands = list(feats["candidate_ids"])
    meta = L.load_meta().set_index("candidate_id").loc[cands]
    splits = L.load_splits()
    labels = L.load_dev_labels()

    dev_mask = np.array([c in splits["inner_fold"] for c in cands])
    dev_c = [c for c in cands if c in splits["inner_fold"]]
    folds = np.array([splits["inner_fold"][c] for c in dev_c])
    y = labels.loc[dev_c].to_numpy()
    rt_share = meta.share_retweets.to_numpy()

    oof = pd.DataFrame({"candidate_id": dev_c, "true_ideology": y, "inner_fold": folds})
    rows, models = [], {}
    for space in L.SPACES + ["tfidf_doc", "m2v_strip"]:
        X = feats[space].astype(np.float64)
        variants = {"raw": (X, None, None)}
        if space != "behav":
            Xp, k, r = L.style_projected(X, rt_share)
            variants["styleproj"] = (Xp, k, r)
        for vname, (Xv, k, r) in variants.items():
            Xd = Xv[dev_mask]
            for learner in ("ridge", "gbm"):
                if learner == "gbm" and vname == "styleproj":
                    continue  # projection is a linear-space diagnostic
                pred = L.oof_predictions(Xd, y, folds, learner)
                s = L.scores(pred, y)
                rows.append({"space": space, "variant": vname, "learner": learner, **s,
                             "style_pc": None if k is None else k + 1, "style_pc_r_rt": r})
                oof[f"{space}|{vname}|{learner}"] = pred
                if vname == "raw":
                    m = L.fit_ridge(Xd, y, folds) if learner == "ridge" else L.fit_gbm(Xd, y)
                    models[f"{space}|{learner}"] = m
                print(f"{space:10s} {vname:9s} {learner:5s}  r={s['r']:.4f} rho={s['rho']:.4f} rmse={s['rmse']:.3f}")

    # unsupervised references on DEV (sign-oriented by observable party, already done in WS0)
    refs = pd.read_csv(L.OUT / "unsupervised_refs.csv").set_index("candidate_id").loc[dev_c]
    for col in ("tfidf_pc1_frozen", "behav_splitA_source_mean"):
        s = L.scores(refs[col].to_numpy(), y)
        rows.append({"space": col, "variant": "unsupervised", "learner": "-", **s})
        print(f"{col:26s} unsupervised r={s['r']:.4f}")

    # residual correlations among single-space ridge probes (synthesis error-family check, DEV)
    cols = [f"{s}|raw|ridge" for s in L.SPACES]
    resid = oof[cols].sub(y, axis=0)
    resid.columns = L.SPACES
    resid.corr().round(3).to_csv(L.OUT / "e41_dev_residual_corr.csv")

    pd.DataFrame(rows).to_csv(L.OUT / "e41_dev_summary.csv", index=False)
    oof.to_csv(L.OUT / "e41_dev_oof.csv", index=False)
    joblib.dump(models, L.INTER / "e41_models.joblib")
    print("\nresidual correlations (ridge, DEV OOF):")
    print(resid.corr().round(2))


if __name__ == "__main__":
    main()
