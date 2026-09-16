"""
07_unseal_evaluate.py — THE single TEST read for WS4
----------------------------------------------------
Reads ws0-harness/sealed_truth.parquet exactly once (candidate-level
true_ideology for the 303 TEST candidates), scores every prediction array
frozen label-free by 07a_freeze_predictions.py, applies the preregistered
bars (§5, Addendum B) and writes:

  outputs/e41_test.csv                 certified probes + unsupervised refs + Δr CIs
  outputs/e42_test_draws.csv / e42_test_summary.csv   label efficiency, n* on TEST
  outputs/e43_test_draws.csv / e43_test_summary.csv   text scarcity
  outputs/e44_test.csv                 shift penalties Δr with paired-bootstrap CI; Independents
  outputs/e45_test.csv / e45_test_residual_corr.csv   stacks, Δr CIs, residual families; E4.6
  outputs/e47_test_coverage.csv / e47_test_conditional.csv   conformal coverage
  outputs/e48_test_n40.csv             the frozen n = 40 AL models
  outputs/decision.json                every bar, met / not met, with the numbers
  ws0-harness/seal_manifest.json       gets a `ws4_unseal` entry (date, sha256 check)

Modes:
  --dry-run   never opens sealed_truth: scores the same pipeline against DEV
              (DEV predictions are in-sample for most models, so the numbers are
              meaningless — this only exercises the code); writes to
              intermediate/dryrun/ instead of outputs/.
  --unseal    the real thing; refuses to run twice (outputs/decision.json exists)
              and refuses without intermediate/test_predictions.joblib.

Nothing here fits a model. Any analysis after this run is exploratory (D9).
"""

import argparse
import hashlib
import json
import sys
from datetime import date

import joblib
import numpy as np
import pandas as pd

import ws4lib as L

BOOT_SEED = L.SEED  # §3


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_truth_once(test_ids: list[str]) -> np.ndarray:
    """Read #2 of sealed_truth.parquet (preregistration §7)."""
    man = json.loads((L.WS0 / "seal_manifest.json").read_text())
    path = L.WS0 / "sealed_truth.parquet"
    assert sha256(path) == man["sealed_truth"]["sha256"], "sealed_truth.parquet hash mismatch"
    t = pd.read_parquet(path, columns=["candidate_id", "true_ideology"])
    c = t.groupby("candidate_id").true_ideology.agg(["first", "nunique"])
    assert (c["nunique"] == 1).all()
    return c["first"].reindex(test_ids).to_numpy()


def dr(pa, pb, y):
    return L.paired_bootstrap_dr(pa, pb, y, seed=BOOT_SEED)


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--unseal", action="store_true")
    args = ap.parse_args()
    dry = args.dry_run

    P = joblib.load(L.INTER / "test_predictions.joblib")
    out_dir = L.INTER / "dryrun" if dry else L.OUT
    out_dir.mkdir(exist_ok=True)
    if not dry and (L.OUT / "decision.json").exists():
        sys.exit("decision.json exists — the unseal has already happened; further TEST work is exploratory (D9).")

    if dry:
        # exercise the code on DEV with DEV labels: predictions for DEV come from the OOF files
        # where they exist (E4.1, stacks); everything else is scored on a DEV-shaped stand-in.
        ids = P["dev_ids"]
        y = L.load_dev_labels().reindex(ids).to_numpy()
        oof = pd.read_csv(L.OUT / "e41_dev_oof.csv").set_index("candidate_id").loc[ids]
        rng = np.random.default_rng(0)
        n = len(ids)

        def get(key):
            base = key.split("|")
            if key.startswith("e41|") and len(base) == 3 and f"{base[1]}|raw|{base[2]}" in oof:
                return oof[f"{base[1]}|raw|{base[2]}"].to_numpy()
            # stand-in: resample the TEST array to DEV length (numbers meaningless, shapes right)
            v = np.asarray(P[key], dtype=float)
            return v[rng.integers(0, len(v), n)]
        print("DRY RUN — sealed_truth.parquet NOT opened; numbers below are not results.")
    else:
        ids = P["candidate_ids"]
        y = read_truth_once(ids)

        def get(key):
            return np.asarray(P[key], dtype=float)

    meta = L.load_meta().set_index("candidate_id").loc[ids]
    n = len(ids)
    decision = {"mode": "dry-run" if dry else "unseal", "date": date.today().isoformat(), "n_test": n, "bars": {}}

    # ================= E4.1 =================
    rows = []
    for space in L.SPACES + ["tfidf_doc", "m2v_strip"]:
        for learner in ("ridge", "gbm"):
            rows.append({"space": space, "variant": "raw", "learner": learner, **L.scores(get(f"e41|{space}|{learner}"), y)})
        rows.append({"space": space, "variant": "styleproj", "learner": "ridge", **L.scores(get(f"e41|{space}|styleproj|ridge"), y)})
    for ref in ("tfidf_pc1_frozen", "behav_splitA_source_mean"):
        rows.append({"space": ref, "variant": "unsupervised", "learner": "-", **L.scores(get(f"ref|{ref}"), y)})
    e41 = pd.DataFrame(rows)
    beh = get("ref|behav_splitA_source_mean")
    pc1 = get("ref|tfidf_pc1_frozen")
    for i, r in e41.iterrows():
        if r.variant == "raw":
            p = get(f"e41|{r.space}|{r.learner}")
            for nm, ref in (("vs_behav_ref", beh), ("vs_tfidf_pc1", pc1)):
                d = dr(p, ref, y)
                e41.loc[i, f"dr_{nm}"], e41.loc[i, f"dr_{nm}_lo"], e41.loc[i, f"dr_{nm}_hi"] = d["dr"], d["lo"], d["hi"]
        if r.variant == "styleproj":
            d = dr(get(f"e41|{r.space}|styleproj|ridge"), get(f"e41|{r.space}|ridge"), y)
            e41.loc[i, "dr_vs_raw"], e41.loc[i, "dr_vs_raw_lo"], e41.loc[i, "dr_vs_raw_hi"] = d["dr"], d["lo"], d["hi"]
    d = dr(get("e41|m2v|ridge"), get("e41|m2v_strip|ridge"), y)
    e41.attrs["m2v_minus_strip"] = d
    e41.to_csv(out_dir / "e41_test.csv", index=False)
    r_t = e41[(e41.space == "tfidf") & (e41.variant == "raw") & (e41.learner == "ridge")].r.item()
    r_d = e41[(e41.space == "doc2vec") & (e41.variant == "raw") & (e41.learner == "ridge")].r.item()
    decision["bars"]["E4.1"] = {"bar": "ridge tfidf and doc2vec r >= .965 on TEST", "tfidf_r": r_t, "doc2vec_r": r_d,
                                "met": bool(r_t >= .965 and r_d >= .965), "m2v_minus_m2v_strip": d}
    print(e41[["space", "variant", "learner", "r", "rho", "rmse"]].round(4).to_string(index=False))

    # ================= E4.2 =================
    sched = json.loads((L.OUT / "e42_draw_schedule.json").read_text())
    full_r = {(r.space, r.learner): r.r for r in e41.itertuples() if r.variant == "raw"}
    rows = []
    for space in L.SPACES + ["m2v_strip"]:
        for learner in ("ridge", "gbm"):
            for nn in sched["n_grid"]:
                for d_ in range(sched["n_draws"]):
                    rows.append({"space": space, "learner": learner, "n": nn, "draw": d_,
                                 **L.scores(get(f"e42|{space}|{learner}|{nn}|{d_}"), y)})
    e42 = pd.DataFrame(rows)
    e42.to_csv(out_dir / "e42_test_draws.csv", index=False)
    summ, nstar = [], {}
    for (space, learner), g_ in e42.groupby(["space", "learner"]):
        ref = full_r[(space, learner)]
        ns = None
        for nn in sched["n_grid"]:
            q1, med, q3 = np.nanpercentile(g_[g_.n == nn].r, [25, 50, 75])
            summ.append({"space": space, "learner": learner, "n": nn, "median_r": med, "q1": q1, "q3": q3,
                         "gap_to_full": ref - med, "n_degenerate": int(g_[g_.n == nn].r.isna().sum())})
            if ns is None and ref - med <= 0.02:
                ns = nn
        summ.append({"space": space, "learner": learner, "n": len(P["dev_ids"]), "median_r": ref, "q1": ref, "q3": ref, "gap_to_full": 0})
        nstar[f"{space}|{learner}"] = ns if ns is not None else "> 320"
    pd.DataFrame(summ).to_csv(out_dir / "e42_test_summary.csv", index=False)
    met = any(isinstance(v, int) and v <= 40 for k, v in nstar.items() if k.endswith("|ridge"))
    decision["bars"]["E4.2"] = {"bar": ">= 1 space with n* <= 40 (ridge)", "n_star": nstar, "met": bool(met),
                                "zero_label_reference_tfidf_pc1_r": float(np.corrcoef(pc1, y)[0, 1])}
    print("\nE4.2 n* on TEST:", nstar)

    # ================= E4.3 =================
    e43_dev = pd.read_csv(L.OUT / "e43_summary.csv", dtype={"k": str})
    rows = []
    for regime in ("mixed", "originals"):
        for k in ["5", "10", "25", "50", "100", "all"]:
            for d_ in range(1 if k == "all" else 20):
                for space in L.SPACES + ["m2v_strip"]:
                    for learner in ("ridge", "gbm"):
                        rows.append({"regime": regime, "k": k, "draw": d_, "space": space, "learner": learner,
                                     **L.scores(get(f"e43|{regime}|{k}|{d_}|{space}|{learner}"), y)})
    e43 = pd.DataFrame(rows)
    e43.to_csv(out_dir / "e43_test_draws.csv", index=False)
    s43 = (e43.groupby(["regime", "k", "space", "learner"]).agg(median_r=("r", "median"), q1=("r", lambda s: s.quantile(.25)),
                                                                 q3=("r", lambda s: s.quantile(.75)), median_rmse=("rmse", "median"),
                                                                 n_draws=("r", "size")).reset_index())
    s43 = s43.merge(e43_dev[["regime", "k", "space", "learner", "n_short"]], how="left")
    s43.to_csv(out_dir / "e43_test_summary.csv", index=False)
    best25 = s43[(s43.regime == "mixed") & (s43.k == "25") & (s43.learner == "ridge") & (s43.space.isin(L.CONTENT))].median_r.max()
    decision["bars"]["E4.3"] = {"bar": "best content space (tfidf/doc2vec ridge) median r >= .90 at k = 25 mixed", "value": float(best25), "met": bool(best25 >= .90)}
    print(f"\nE4.3 best content r at k=25 mixed: {best25:.4f}")

    # ================= E4.4 =================
    sched44 = json.loads((L.OUT / "e44_control_schedule.json").read_text())
    is_ind = (meta.party == "I").to_numpy()
    rows = []
    for axis, eval_mask, lab in (("incumbency", (meta.incumbent == False).to_numpy(), "challengers"),
                                 ("chamber", (meta.chamber == "Senate").to_numpy(), "senators")):
        for space in L.SPACES + ["m2v_strip"]:
            for learner in ("ridge", "gbm"):
                ps = get(f"e44|{axis}|{space}|{learner}")
                pc = np.vstack([get(f"e44|{axis}|{space}|{learner}|ctrl{d_}") for d_ in range(sched44["n_controls"])])
                from importlib import import_module
                cdr = import_module("05_extrapolation").control_delta_r
                res = cdr(ps[eval_mask], pc[:, eval_mask], y[eval_mask], seed=BOOT_SEED)
                row = {"axis": axis, "eval": lab, "space": space, "learner": learner, "n_eval": int(eval_mask.sum()),
                       "shift_r": float(np.corrcoef(ps[eval_mask], y[eval_mask])[0, 1]),
                       "control_median_r": float(np.median([np.corrcoef(p[eval_mask], y[eval_mask])[0, 1] for p in pc])),
                       **{f"dr_{k}": v for k, v in res.items()}}
                if axis == "incumbency":
                    ni = eval_mask & ~is_ind
                    ii = eval_mask & is_ind
                    row["shift_r_nonind"] = float(np.corrcoef(ps[ni], y[ni])[0, 1])
                    row["ind_n"] = int(ii.sum())
                    row["ind_r"] = float(np.corrcoef(ps[ii], y[ii])[0, 1]) if ii.sum() > 2 else np.nan
                    row["ind_rmse"] = float(np.sqrt(np.mean((ps[ii] - y[ii]) ** 2)))
                rows.append(row)
    e44 = pd.DataFrame(rows)
    e44.to_csv(out_dir / "e44_test.csv", index=False)
    inc = e44[(e44.axis == "incumbency") & (e44.learner == "ridge") & (e44.space.isin(L.CONTENT))]
    decision["bars"]["E4.4"] = {"bar": "|Δr| < .01 with CI covering 0 (incumbent→challenger, ridge content spaces)",
                                "per_space": {r.space: {"dr": r.dr_dr, "lo": r.dr_lo, "hi": r.dr_hi} for r in inc.itertuples()},
                                "met": bool(((inc.dr_dr.abs() < .01) & (inc.dr_lo <= 0) & (inc.dr_hi >= 0)).all())}
    print("\nE4.4:\n" + e44[["axis", "space", "learner", "shift_r", "control_median_r", "dr_dr", "dr_lo", "dr_hi"]].round(4).to_string(index=False))

    # ================= E4.5 / E4.6 =================
    rows = []
    stacks = {k[4:]: get(k) for k in P if k.startswith("e45|")}
    single = {s: get(f"e41|{s}|ridge") for s in L.SPACES}
    best_single = max(single, key=lambda s: np.corrcoef(single[s], y)[0, 1])
    for name, p in stacks.items():
        ok = ~np.isnan(p)
        row = {"stack": name, "n": int(ok.sum()), **L.scores(p[ok], y[ok])}
        if "@150" not in name:
            for nm, ref in ((f"vs_best_single({best_single})", single[best_single]), ("vs_behav_ref", beh)):
                d = dr(p, ref, y)
                row.update({f"dr_{nm}": d["dr"], f"dr_{nm}_lo": d["lo"], f"dr_{nm}_hi": d["hi"]})
        rows.append(row)
    llm = get("e46|llm_alone")
    ok = ~np.isnan(llm)
    rows.append({"stack": "llm_alone@150", "n": int(ok.sum()), **L.scores(llm[ok], y[ok])})
    for s in L.SPACES:
        rows.append({"stack": f"{s}@150", "n": int(ok.sum()), **L.scores(single[s][ok], y[ok])})
    e45 = pd.DataFrame(rows)
    e45.to_csv(out_dir / "e45_test.csv", index=False)
    resid = pd.DataFrame({s: single[s] - y for s in L.SPACES}).corr().round(3)
    resid.to_csv(out_dir / "e45_test_residual_corr.csv")
    sa = e45[e45["stack"] == "stack_all"].iloc[0]
    k1 = f"dr_vs_best_single({best_single})"
    decision["bars"]["E4.5"] = {"bar": "stack_all beats best single (CI excl. 0) AND beats behavioral split-A reference",
                                "best_single": best_single, "stack_all_r": float(sa.r),
                                "dr_vs_best_single": [float(sa[k1]), float(sa[k1 + "_lo"]), float(sa[k1 + "_hi"])],
                                "dr_vs_behav_ref": [float(sa.dr_vs_behav_ref), float(sa.dr_vs_behav_ref_lo), float(sa.dr_vs_behav_ref_hi)],
                                "met": bool(sa[k1 + "_lo"] > 0 and sa.dr_vs_behav_ref_lo > 0)}
    print("\nE4.5:\n" + e45.round(4).to_string(index=False))
    print("residual correlations (TEST):\n", resid)

    # ================= E4.7 =================
    q = json.loads((L.OUT / "e47_qhat.json").read_text())
    sd_y = float(np.std(y))
    preds47 = {s: get(f"e41|{s}|ridge") for s in L.CONFORMAL_SPACES}
    preds47["stack_all"] = stacks["stack_all"]
    rows, cond = [], []
    nq = pd.qcut(meta.n_tweets, 4, labels=["Q1", "Q2", "Q3", "Q4"]).astype(str).to_numpy()
    strata = {"party=D": (meta.party == "D").to_numpy(), "party=R": (meta.party == "R").to_numpy(), "party=I": is_ind,
              "House": (meta.chamber == "House").to_numpy(), "Senate": (meta.chamber == "Senate").to_numpy(),
              "incumbent": (meta.incumbent == True).to_numpy(), "challenger": (meta.incumbent == False).to_numpy(),
              **{f"ntweets_{qq}": nq == qq for qq in ["Q1", "Q2", "Q3", "Q4"]}}
    for name, p in preds47.items():
        for a in q["alphas"]:
            qa = q["qhat"][name][str(a)]
            hit = np.abs(y - p) <= qa
            lo, hi = L.coverage_ci(int(hit.sum()), n)
            rows.append({"model": name, "alpha": a, "condition": "marginal", "coverage": float(hit.mean()), "ci_lo": lo, "ci_hi": hi,
                         "n": n, "qhat": qa, "width_rel_sd": 2 * qa / sd_y})
            for st, m_ in strata.items():
                lo, hi = L.coverage_ci(int(hit[m_].sum()), int(m_.sum()))
                cond.append({"model": name, "alpha": a, "condition": st, "coverage": float(hit[m_].mean()), "ci_lo": lo, "ci_hi": hi, "n": int(m_.sum())})
        # (a) scarcity shift: full-volume model on k = 25 TEST features, q̂ fixed
        if name != "stack_all":
            for a in q["alphas"]:
                qa = q["qhat"][name][str(a)]
                cov = [np.mean(np.abs(y - get(f"e47a|{d_}|{name}")) <= qa) for d_ in range(20)]
                cond.append({"model": name, "alpha": a, "condition": "shift:k=25 (mean of 20 draws)", "coverage": float(np.mean(cov)),
                             "ci_lo": float(np.min(cov)), "ci_hi": float(np.max(cov)), "n": n})
            # (b) incumbent-trained ridge + incumbent-calibrated q̂ → TEST challengers (§B.4(1))
            chal = (meta.incumbent == False).to_numpy()
            pi = get(f"e44|incumbency|{name}|ridge")
            for a in q["alphas"]:
                qa = q["qhat_incumbent_calibrated"][name][str(a)]
                hit = np.abs(y[chal] - pi[chal]) <= qa
                lo, hi = L.coverage_ci(int(hit.sum()), int(chal.sum()))
                cond.append({"model": name, "alpha": a, "condition": "shift:incumbent-calibrated→challengers", "coverage": float(hit.mean()),
                             "ci_lo": lo, "ci_hi": hi, "n": int(chal.sum()), "qhat": qa})
                # context: all-DEV model + all-DEV q̂ on the same challengers
                hit2 = np.abs(y[chal] - p[chal]) <= q["qhat"][name][str(a)]
                lo, hi = L.coverage_ci(int(hit2.sum()), int(chal.sum()))
                cond.append({"model": name, "alpha": a, "condition": "context:all-DEV model→challengers", "coverage": float(hit2.mean()),
                             "ci_lo": lo, "ci_hi": hi, "n": int(chal.sum())})
    e47 = pd.DataFrame(rows)
    e47.to_csv(out_dir / "e47_test_coverage.csv", index=False)
    pd.DataFrame(cond).to_csv(out_dir / "e47_test_conditional.csv", index=False)
    c10 = {s: float(e47[(e47.model == s) & (e47.alpha == 0.10)].coverage.item()) for s in L.CONTENT}
    decision["bars"]["E4.7"] = {"bar": "tfidf and doc2vec marginal coverage at α=.10 in [.86, .94]", "coverage": c10,
                                "met": bool(all(.86 <= v <= .94 for v in c10.values()))}
    print("\nE4.7 marginal coverage:\n" + e47.round(3).to_string(index=False))

    # ================= E4.8 =================
    picks = json.loads((L.OUT / "e48_n40_picks.json").read_text())
    rows = [{"space": k.split("|")[0], "strategy": k.split("|")[1], "dev_r_at_40": v["dev_r_at_40"],
             **L.scores(get(f"e48|{k}"), y)} for k, v in picks.items()]
    e48 = pd.DataFrame(rows)
    e48.to_csv(out_dir / "e48_test_n40.csv", index=False)
    al = pd.read_csv(L.OUT / "e48_aulc.csv").merge(pd.read_csv(L.OUT / "e48_summary.csv").drop_duplicates(["space", "strategy"])[["space", "strategy", "n_star_al"]])
    ok48 = []
    for space in L.CONTENT:
        rn = al[(al.space == space) & (al.strategy == "random")].n_star_al.item()
        for st in ("diversity", "hybrid"):
            r_ = al[(al.space == space) & (al.strategy == st)].iloc[0]
            ok48.append(bool(rn > 0 and r_.n_star_al > 0 and r_.n_star_al <= 0.75 * rn and r_.lo > 0))
    decision["bars"]["E4.8"] = {"bar": "on >= 1 content space, diversity or hybrid n*_AL <= .75 x random n*_AL AND AULC Δ CI > 0 (DEV inner folds, §B.2)",
                                "dev_table": al.round(4).to_dict("records"), "met": bool(any(ok48)),
                                "note": "n*_AL = -1 means not reached by n = 60 for any strategy → 'no room / no reach' rather than a strategy effect"}
    print("\nE4.8 n=40 picks on TEST:\n" + e48.round(4).to_string(index=False))

    (out_dir / "decision.json").write_text(json.dumps(decision, indent=1, default=float))
    print("\nBARS:", {k: v["met"] for k, v in decision["bars"].items()})

    if not dry:
        man_path = L.WS0 / "seal_manifest.json"
        man = json.loads(man_path.read_text())
        man["ws4_unseal"] = {"unsealed_on": date.today().isoformat(), "script": "analyses/ws4-supervised/scripts/07_unseal_evaluate.py",
                             "sealed_truth_sha256_verified": man["sealed_truth"]["sha256"], "n_test": n,
                             "predictions_frozen_on": P.get("frozen_on"), "note": "single TEST read; all later TEST contact is exploratory (D9)"}
        man_path.write_text(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
