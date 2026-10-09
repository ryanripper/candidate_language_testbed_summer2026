"""
10_figures.py — WS4 figures, implementing figure_specs.md (frozen 2026-09-15)
----------------------------------------------------------------------------
Reads only the outputs of the single 07 run (outputs/e4x_test_*.csv,
decision.json) plus DEV-side context explicitly allowed by the specs
(e45_meta_weights.json, e41_dev_residual_corr.csv, e48_summary.csv).
Touches no labels. Writes figures/fig*.png.
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import ws4lib as L

FIG = L.HERE.parent / "figures" if hasattr(L, "HERE") else L.OUT.parent / "figures"
FIG.mkdir(exist_ok=True)
OUT = L.OUT

plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9, "axes.titlesize": 10, "legend.fontsize": 8})

# Global conventions (figure_specs.md)
COL = {"tfidf": "#1f5fbf", "doc2vec": "#118a8a", "w2v": "#6e8b6e", "m2v": "#e07b00",
       "m2v_strip": "#e07b00", "behav": "#6a3d9a", "tfidf_doc": "#7aa6e0",
       "stack_content": "black", "stack_style": "black", "stack_all": "black"}
LS = {k: "-" for k in COL}
LS["m2v_strip"] = "--"
LABEL = {"tfidf": "TF-IDF", "doc2vec": "doc2vec", "w2v": "word2vec", "m2v": "Model2Vec (raw)",
         "m2v_strip": "Model2Vec (RT-stripped)", "behav": "behavioral (split A)", "tfidf_doc": "TF-IDF (doc)",
         "stack_content": "stack: content", "stack_style": "stack: style", "stack_all": "stack: cross-family"}
ORDER = ["tfidf", "doc2vec", "w2v", "m2v", "m2v_strip", "behav"]

dec = json.loads((OUT / "decision.json").read_text())
assert dec["mode"] == "unseal", "figures require the real 07 run"
N_TEST = dec["n_test"]
e41 = pd.read_csv(OUT / "e41_test.csv")
REF_PC1 = e41[e41.space == "tfidf_pc1_frozen"].r.item()
REF_BEH = e41[e41.space == "behav_splitA_source_mean"].r.item()


def refs(ax, pc1=True, beh=True, bar=None):
    if pc1:
        ax.axhline(REF_PC1, ls=":", color=COL["tfidf"], lw=1, label=f"frozen TF-IDF PC1, zero labels ({REF_PC1:.3f})")
    if beh:
        ax.axhline(REF_BEH, ls=":", color=COL["behav"], lw=1, label=f"behavioral split-A reference ({REF_BEH:.3f})")
    if bar is not None:
        ax.axhline(bar, color="0.3", lw=0.6, label=f"bar {bar}")


# ---------------------------------------------------------------- Fig 1  E4.1
spaces1 = ["tfidf", "doc2vec", "w2v", "m2v", "m2v_strip", "behav", "tfidf_doc"]
fig, ax = plt.subplots(figsize=(8.0, 3.8))
x = np.arange(len(spaces1))
for i, s in enumerate(spaces1):
    g = e41[e41.space == s]
    r_ridge = g[(g.variant == "raw") & (g.learner == "ridge")].r.item()
    r_gbm = g[(g.variant == "raw") & (g.learner == "gbm")].r.item()
    r_sp = g[g.variant == "styleproj"].r.item()
    # paired-bootstrap CI vs behavioral reference, drawn on the ridge point (spec: every claim shows a CI)
    rr = g[(g.variant == "raw") & (g.learner == "ridge")].iloc[0]
    lo, hi = REF_BEH + rr.dr_vs_behav_ref_lo, REF_BEH + rr.dr_vs_behav_ref_hi
    mk = "D" if s == "tfidf_doc" else "o"
    ax.plot([i, i], [lo, hi], color=COL[s], lw=1.2, alpha=.7)
    ax.plot(i, r_ridge, mk, color=COL[s], ms=7, mec=COL[s], label="ridge" if i == 0 else None)
    ax.plot(i + .18, r_gbm, mk, mfc="white", mec=COL[s], ms=7, label="GBM" if i == 0 else None)
    ax.plot(i - .18, r_sp, "s", mfc="white", mec=COL[s], ms=5, label="ridge, style PC projected out" if i == 0 else None)
refs(ax, bar=.965)
ax.set_xticks(x)
ax.set_xticklabels([LABEL[s] for s in spaces1], rotation=20, ha="right")
ax.set_ylabel("Pearson r with planted ideology (TEST, n = %d)" % N_TEST)
ax.set_ylim(.89, .99)
ax.set_title("Fig 1 — E4.1 certified probes on TEST (bars: paired-bootstrap 95% CI of Δr vs behavioral reference)", loc="left")
h, l = ax.get_legend_handles_labels()
ax.legend(h, l, loc="lower left", ncol=2, frameon=False)
fig.tight_layout(); fig.savefig(FIG / "fig1_certified_probes.png"); plt.close(fig)

# ---------------------------------------------------------------- Fig 2  E4.2
e42 = pd.read_csv(OUT / "e42_test_summary.csv")
nstar = dec["bars"]["E4.2"]["n_star"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.0), gridspec_kw={"width_ratios": [1.3, 1]})
for ax, learner in ((a1, "ridge"), (a2, "gbm")):
    for s in ORDER:
        g = e42[(e42.space == s) & (e42.learner == learner)].sort_values("n")
        g = g[g.n <= 320]
        ax.plot(g.n, g.median_r, LS[s], color=COL[s], lw=1.6, label=LABEL[s])
        ax.fill_between(g.n, g.q1, g.q3, color=COL[s], alpha=.12, lw=0)
        ns = nstar[f"{s}|{learner}"]
        if isinstance(ns, int):
            y_at = g[g.n == ns].median_r.item()
            ax.plot([ns, ns], [y_at - .035, y_at + .035], color=COL[s], lw=2.2)
    ax.set_xscale("log")
    ax.set_xticks([10, 20, 40, 80, 160, 320])
    ax.set_xticklabels([10, 20, 40, 80, 160, 320])
    ax.set_xlabel("labeled DEV candidates n")
    ax.set_ylim(.3, 1.0)
    ax.axhline(REF_PC1, ls=":", color=COL["tfidf"], lw=1)
a1.set_ylabel("median TEST r over 50 draws (band = IQR)")
a1.set_title("Fig 2 — E4.2 label efficiency, ridge (ticks = n*: first n within .02 of full-DEV TEST r)")
a2.set_title("supplement: GBM (n ≤ 40 degenerate, not shown)")
a1.text(10.5, REF_PC1 + .008, f"zero-label TF-IDF PC1 on TEST = {REF_PC1:.3f}", color=COL["tfidf"], fontsize=8)
a1.legend(loc="lower right", frameon=False)
fig.tight_layout(); fig.savefig(FIG / "fig2_label_efficiency.png"); plt.close(fig)

# ---------------------------------------------------------------- Fig 3  E4.3
e43 = pd.read_csv(OUT / "e43_test_summary.csv", dtype={"k": str})
e45 = pd.read_csv(OUT / "e45_test.csv")
llm150 = e45[e45["stack"] == "llm_alone@150"].iloc[0]
kpos = {"5": 5, "10": 10, "25": 25, "50": 50, "100": 100, "all": 250}
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
for ax, regime in zip(axes, ("mixed", "originals")):
    for s in ORDER:
        g = e43[(e43.regime == regime) & (e43.learner == "ridge") & (e43.space == s)]
        g = g.assign(kx=g.k.map(kpos)).sort_values("kx")
        if regime == "originals" and s in ("w2v", "m2v_strip"):
            # originals-only regime was run for tfidf/doc2vec/m2v/behav in 04 (README); skip empties
            if g.median_r.isna().all():
                continue
        ax.plot(g.kx, g.median_r, LS[s], color=COL[s], lw=1.6, label=LABEL[s])
        ax.fill_between(g.kx, g.q1, g.q3, color=COL[s], alpha=.12, lw=0)
    ax.plot(25, .90, "_", color="0.2", ms=14, mew=2, label="bar: r ≥ .90 at k = 25")
    ax.set_xscale("log")
    ax.set_xticks(list(kpos.values()))
    ax.set_xticklabels(["5", "10", "25", "50", "100", "all"])
    ax.set_xlabel("tweets per candidate k")
    ax.set_title(f"{regime} regime" + (" (retweets excluded)" if regime == "originals" else ""))
    # n_short annotation
    ns = (e43[(e43.regime == regime) & (e43.learner == "ridge") & (e43.space == "tfidf")]
          .assign(kx=lambda d: d.k.map(kpos)).sort_values("kx"))
    for _, r in ns.iterrows():
        if r.k != "all" and not pd.isna(r.n_short):
            ax.text(r.kx, -.08, f"<k: {int(r.n_short)}", ha="center", fontsize=7, color="0.4")
    ax.set_ylim(-.1, 1.0)
axes[0].plot(25, llm150.r, "*", color="crimson", ms=11, label=f"WS3 LLM ask-and-average, TEST∩150 (n = {int(llm150.n)}): {llm150.r:.3f}")
axes[0].set_ylabel("median TEST r over 20 draws (band = IQR), ridge")
axes[1].text(8, .02, "behavioral: no retweets → no signal", color=COL["behav"], fontsize=8)
axes[0].legend(loc="lower right", frameon=False, fontsize=7)
fig.suptitle("Fig 3 — E4.3 text scarcity on TEST (ridge trained on k-tweet DEV features, scored on k-tweet TEST features; grey numbers = candidates with < k tweets)", fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "fig3_text_scarcity.png"); plt.close(fig)

# ---------------------------------------------------------------- Fig 4  E4.4
e44 = pd.read_csv(OUT / "e44_test.csv")
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
rows = [(s, l) for s in ORDER for l in ("ridge", "gbm")]
for ax, axis, title in zip(axes, ("incumbency", "chamber"), ("incumbent-trained → TEST challengers", "House-trained → TEST senators")):
    g = e44[e44.axis == axis].set_index(["space", "learner"])
    ax.axvspan(-.01, .01, color="0.92", lw=0)
    ax.axvline(0, color="0.3", lw=.6)
    for i, (s, l) in enumerate(rows):
        r = g.loc[(s, l)]
        y = len(rows) - 1 - i
        ax.plot([r.dr_lo, r.dr_hi], [y, y], color=COL[s], lw=1.4)
        ax.plot(r.dr_dr, y, "o", color=COL[s], mfc=COL[s] if l == "ridge" else "white", ms=6)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{LABEL[s]} · {l}" for s, l in rows][::-1])
    ax.set_xlabel("Δr = median control r − shift-trained r (95% CI)")
    ax.set_title(title + f"  (n = {int(g.n_eval.iloc[0])})")
    xl = max(abs(g.dr_lo.min()), abs(g.dr_hi.max())) * 1.1
    ax.set_xlim(-xl, xl)
fig.suptitle("Fig 4 — E4.4 shift penalty on TEST (shaded = ±.01 band; filled = ridge, hollow = GBM; bar evaluated on ridge TF-IDF / doc2vec, incumbency)", fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "fig4_shift_penalty.png"); plt.close(fig)

# Independents table (spec: small table, not a panel)
ind = e44[(e44.axis == "incumbency")][["space", "learner", "ind_n", "ind_r", "ind_rmse", "shift_r_nonind"]].round(3)
ind.to_csv(FIG / "fig4_independents_table.csv", index=False)

# ---------------------------------------------------------------- Fig 5  E4.5
W = json.loads((OUT / "e45_meta_weights.json").read_text())
rdev = pd.read_csv(OUT / "e41_dev_residual_corr.csv", index_col=0)
rtest = pd.read_csv(OUT / "e45_test_residual_corr.csv", index_col=0)
fig, axes = plt.subplots(1, 4, figsize=(13.5, 4.3), gridspec_kw={"width_ratios": [1.1, 1.2, 1, 1]})
# (a) weights
a = axes[0]
stks = ["stack_content", "stack_style", "stack_all"]
bottom = np.zeros(3)
for s in L.SPACES:
    vals = np.array([W[k].get(s, 0.0) for k in stks])
    a.bar(range(3), vals, bottom=bottom, color=COL[s], label=LABEL[s], width=.6)
    bottom += vals
a.set_xticks(range(3)); a.set_xticklabels(["content", "style", "cross-family"])
a.set_ylabel("DEV-fit meta weight (non-negative ridge)")
a.set_title("(a) stack weights (DEV)")
a.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(0, -0.12), ncol=3)
# (b) TEST r with Δr CIs
a = axes[1]
best = dec["bars"]["E4.5"]["best_single"]
items = [("stack_content", None), ("stack_style", None), ("stack_all", None)]
ys = []
for i, (k, _) in enumerate(items):
    r = e45[e45["stack"] == k].iloc[0]
    a.plot(i, r.r, "o", color="black", ms=7)
    col = f"dr_vs_best_single({best})"
    a.plot([i, i], [r.r - (r[col] - r[col + "_lo"]), r.r + (r[col + "_hi"] - r[col])], color="black", lw=1.4)
    ys.append(r.r)
rb = e41[(e41.space == best) & (e41.variant == "raw") & (e41.learner == "ridge")].r.item()
a.axhline(rb, ls="--", color=COL[best], lw=1, label=f"best single ridge probe: {LABEL[best]} ({rb:.3f})")
a.axhline(REF_BEH, ls=":", color=COL["behav"], lw=1, label=f"behavioral split-A reference ({REF_BEH:.3f})")
a.set_xticks(range(3)); a.set_xticklabels(["content", "style", "cross-family"])
a.set_ylim(.970, .992)
a.set_title("(b) TEST r; bars = 95% CI of Δr vs best single")
a.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(0, -0.12))
# (c) residual-correlation heatmaps
for a, M, ttl in ((axes[2], rtest, "(c) residual corr, TEST (n = 303)"), (axes[3], rdev, "residual corr, DEV OOF (development)")):
    im = a.imshow(M.values, vmin=0, vmax=1, cmap="Blues")
    a.set_xticks(range(5)); a.set_yticks(range(5))
    a.set_xticklabels(M.columns, rotation=45, ha="right"); a.set_yticklabels(M.index)
    for i in range(5):
        for j in range(5):
            a.text(j, i, f"{M.values[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if M.values[i, j] > .6 else "black")
    a.set_title(ttl)
    a.spines[:].set_visible(False)
fig.suptitle("Fig 5 — E4.5 ensembles", fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "fig5_ensembles.png"); plt.close(fig)

# E4.6 footnote table
e45[e45["stack"].str.contains("@150")][["stack", "n", "r", "rho", "rmse"]].round(4).to_csv(FIG / "fig5_e46_footnote_table.csv", index=False)

# ---------------------------------------------------------------- Fig E4.7a
cov = pd.read_csv(OUT / "e47_test_coverage.csv")
models = L.CONFORMAL_SPACES + ["stack_all"]
fig, ax = plt.subplots(figsize=(9.5, 4.0))
xs = np.arange(len(models))
ax.axhspan(.86, .94, color="0.92", lw=0, label="bar band [.86, .94] at α = .10")
for a_, off, mk in ((0.10, -.17, "o"), (0.05, .17, "s")):
    ax.axhline(1 - a_, ls="--", color="0.5", lw=.7)
    for i, m in enumerate(models):
        r = cov[(cov.model == m) & (cov.alpha == a_)].iloc[0]
        ax.plot([i + off, i + off], [r.ci_lo, r.ci_hi], color=COL[m], lw=1.3)
        ax.plot(i + off, r.coverage, mk, color=COL[m], mfc=COL[m] if a_ == .10 else "white", ms=6)
        ax.text(i + off, .80, f"{r.width_rel_sd:.2f}", ha="center", fontsize=7, color="0.3")
ax.text(-.45, .815, "width / SD(truth):", fontsize=7, color="0.3", ha="left")
ax.set_xticks(xs); ax.set_xticklabels([LABEL[m] for m in models], rotation=20, ha="right")
ax.set_ylim(.78, 1.0)
ax.set_ylabel("empirical TEST coverage (Clopper–Pearson 95% CI)")
ax.set_title("Fig E4.7a — split-conformal coverage on TEST: filled circles α = .10 (nominal .90), hollow squares α = .05 (nominal .95)")
ax.legend(frameon=False, loc="upper left", fontsize=7)
fig.tight_layout(); fig.savefig(FIG / "fig6_conformal_coverage.png"); plt.close(fig)

# ---------------------------------------------------------------- Fig E4.7b
cond = pd.read_csv(OUT / "e47_test_conditional.csv")
cond = cond[cond.alpha == 0.10]
cols = ["party=D", "party=R", "party=I", "House", "Senate", "incumbent", "challenger",
        "ntweets_Q1", "ntweets_Q2", "ntweets_Q3", "ntweets_Q4",
        "shift:k=25 (mean of 20 draws)", "shift:incumbent-calibrated→challengers"]
short = ["D", "R", "I", "House", "Senate", "incumb.", "chall.", "tweets Q1", "Q2", "Q3", "Q4", "shift: k = 25", "shift: inc.→chall."]
Mv = cond.pivot(index="model", columns="condition", values="coverage").reindex(index=models, columns=cols)
Mn = cond.pivot(index="model", columns="condition", values="n").reindex(index=models, columns=cols)
fig, ax = plt.subplots(figsize=(11.5, 4.2))
im = ax.imshow(Mv.values, cmap="RdBu", vmin=.70, vmax=1.10)
ax.set_xticks(range(len(cols))); ax.set_xticklabels(short, rotation=35, ha="right")
ax.set_yticks(range(len(models))); ax.set_yticklabels([LABEL[m] for m in models])
for i in range(len(models)):
    for j in range(len(cols)):
        v = Mv.values[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.2f}\n({int(Mn.values[i, j])})", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(v - .90) > .12 else "black")
        else:
            ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="0.5")
ax.axvline(10.5, color="black", lw=1.2)
ax.spines[:].set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=.025, pad=.02); cb.set_label("coverage at α = .10 (nominal .90)")
ax.set_title("Fig E4.7b — conditional coverage on TEST, α = .10, DEV q̂ fixed; cell = coverage (n); right of line = shift", loc="left")
fig.tight_layout(); fig.savefig(FIG / "fig7_conditional_coverage.png"); plt.close(fig)

# ---------------------------------------------------------------- Fig E4.8
al = pd.read_csv(OUT / "e48_summary.csv")
n40 = pd.read_csv(OUT / "e48_test_n40.csv")
STRAT_LS = {"random": ("0.45", "-"), "uncertainty": (None, "-"), "diversity": (None, "--"), "hybrid": (None, ":")}
fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.9), sharey=True)
for ax, s in zip(axes, L.AL_SPACES):
    for st, (c, ls) in STRAT_LS.items():
        g = al[(al.space == s) & (al.strategy == st)].sort_values("n")
        c_ = c or COL[s]
        ax.plot(g.n, g.median_r, ls, color=c_, lw=1.6 if st != "random" else 1.8, label=st)
        if st == "random":
            ax.fill_between(g.n, g.q1, g.q3, color="0.5", alpha=.15, lw=0)
            ns = g.n_star_al.iloc[0]
            if ns > 0:
                ax.axvline(ns, color="0.45", lw=1); ax.text(ns, .3, f"random n*_AL = {int(ns)}", fontsize=7, color="0.3", rotation=90, va="bottom")
            else:
                ax.text(60 if s != "tfidf" else 10, .30 if s != "tfidf" else .93, "n*_AL not reached\nby n = 60 (random)", fontsize=7, color="0.3", ha="right" if s != "tfidf" else "left", va="bottom" if s != "tfidf" else "top")
        t = n40[(n40.space == s) & (n40.strategy == st)]
        if len(t):
            ax.plot(40, t.r.item(), "o" if st != "random" else "o", color=c_, mfc="white", mec=c_, ms=7, mew=1.6)
    ax.set_xlabel("labeled pool candidates n")
    ax.set_title(LABEL[s])
    ax.set_xlim(8, 62); ax.set_ylim(.25, 1.0)
axes[0].set_ylabel("median held-out-fold r, DEV inner folds (development)")
h, l = axes[0].get_legend_handles_labels()
h.append(Line2D([], [], marker="o", mfc="white", mec="black", ls="", ms=7)); l.append("n = 40 pick scored on TEST")
axes[0].legend(h, l, frameon=False, fontsize=7, loc="lower right", bbox_to_anchor=(1.02, 0.0))
fig.suptitle("Fig E4.8 — active learning: curves are DEV inner-fold numbers by design (§B.2); only the hollow n = 40 markers are TEST", fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "fig8_active_learning.png"); plt.close(fig)

# ---------------------------------------------------------------- Fig 0  decision table
fig, ax = plt.subplots(figsize=(11.5, 2.8))
ax.axis("off")
rows = []
for k, v in dec["bars"].items():
    if k == "E4.1":
        num = f"tfidf {v['tfidf_r']:.3f}, doc2vec {v['doc2vec_r']:.3f}"
    elif k == "E4.2":
        ns = {kk.split("|")[0]: vv for kk, vv in v["n_star"].items() if kk.endswith("|ridge")}
        num = ", ".join(f"{a} {b}" for a, b in ns.items())
    elif k == "E4.3":
        num = f"best content r at k = 25: {v['value']:.3f}"
    elif k == "E4.4":
        num = "; ".join(f"{s} Δr {d['dr']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]" for s, d in v["per_space"].items())
    elif k == "E4.5":
        d1, d2 = v["dr_vs_best_single"], v["dr_vs_behav_ref"]
        num = f"stack_all {v['stack_all_r']:.3f}; Δ vs {v['best_single']} {d1[0]:+.3f} [{d1[1]:+.3f}, {d1[2]:+.3f}]; Δ vs behav {d2[0]:+.3f} [{d2[1]:+.3f}, {d2[2]:+.3f}]"
    elif k == "E4.7":
        num = ", ".join(f"{a} {b:.3f}" for a, b in v["coverage"].items())
    elif k == "E4.8":
        num = "no content space reaches n*_AL by n = 60 under any strategy (DEV); see text"
    rows.append([k, v["bar"][:62] + ("…" if len(v["bar"]) > 62 else ""), num, "MET" if v["met"] else "not met"])
tbl = ax.table(cellText=rows, colLabels=["exp", "preregistered bar", "TEST value", "verdict"], loc="center",
               colWidths=[.05, .38, .50, .07], cellLoc="left")
tbl.auto_set_font_size(False); tbl.set_fontsize(6.8); tbl.scale(1, 1.35)
for (i, j), c in tbl.get_celld().items():
    if i > 0 and j == 3:
        c.set_facecolor("#d8ecd8" if rows[i - 1][3] == "MET" else "#f3d4d4")
ax.set_title(f"Fig 0 — WS4 decision table, TEST unsealed {dec['date']} (n = {N_TEST}); bars fixed in preregistration §5 + Addenda B", fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "fig0_decision_table.png"); plt.close(fig)

print("wrote", sorted(p.name for p in FIG.iterdir()))
