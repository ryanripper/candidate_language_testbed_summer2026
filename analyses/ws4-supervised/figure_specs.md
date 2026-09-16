# WS4 figure specifications — frozen 2026-09-15, before unseal

*Written after scripts 03–06, 08, 09 ran on DEV and before
`07_unseal_evaluate.py` touched TEST labels. `10_figures.py` implements these
specs from 07's outputs; any figure not listed here is exploratory (D9).
Every panel that shows a TEST number takes it from `outputs/e4x_test_*.csv`
written by the single 07 run; DEV development numbers may appear only as
grey context and are labelled "DEV (development)".*

Global conventions: one colour per feature space, fixed across all figures —
tfidf = blue, doc2vec = teal, w2v = grey-green, m2v = orange, m2v_strip =
orange dashed, behav = purple, stacks = black; unsupervised references as
dotted horizontal lines. r on the y-axis, axis range chosen per panel but
never truncated so that a .005 difference looks large without the CI visible.
Every between-instrument claim shows the paired-bootstrap 95 % CI.

| Fig | Content | Source | Notes |
|---|---|---|---|
| 1 | **Certified probes (E4.1).** Dot-and-CI chart: TEST r per space (ridge filled, GBM hollow), with and without style projection; dotted lines for frozen TF-IDF PC1 and behavioral split-A reference on the same TEST candidates; the .965 bar as a thin line. | `e41_test.csv` | m2v and m2v_strip adjacent; `tfidf_doc` as a diagnostic marker |
| 2 | **Label efficiency (E4.2).** Learning curves: median TEST r vs n (log-x), IQR band, one line per space, ridge only (GBM in a supplementary panel with n ≤ 40 marked degenerate). n\* per space marked with a vertical tick at the first n within .02 of the full-DEV TEST r. Horizontal dotted line = unsupervised TF-IDF PC1 on TEST (zero-label reference). | `e42_test_summary.csv` | y-range from .3 to 1.0 — the TF-IDF collapse at n = 10 is the point |
| 3 | **Text scarcity (E4.3).** Two panels (mixed / originals-only): median TEST r vs k (log-x), IQR band, per space. The .90-at-k = 25 bar as a marker; the WS3 LLM n = 150 score (.970 on all 150; TEST∩150 value from 07) as a reference point at k = 25 in the mixed panel. Count of candidates with < k tweets annotated under the x-axis. | `e43_test_summary.csv` | behav in the originals panel is a flat line near 0 — keep it, label "no retweets → no signal" |
| 4 | **Shift penalty (E4.4).** Forest plot: Δr = median control − shift-trained, with paired-bootstrap CI, one row per space × learner, two panels (incumbent→challenger; House→Senate). ±.01 band shaded. Independents (n = 9 on TEST) as a separate small table, not a panel. | `e44_test.csv` | |
| 5 | **Ensembles (E4.5).** (a) Weights bar chart for the three stacks (DEV-fit meta weights, `e45_meta_weights.json`); (b) TEST r for stacks vs best single vs behavioral reference with paired-bootstrap Δr CIs; (c) residual-correlation heatmap of the single-space ridge probes on TEST next to the DEV one from 02. | `e45_test.csv`, `e45_test_residual_corr.csv` | E4.6 (n = 52) as a footnote table only |
| E4.7a | **Conformal coverage by α.** Per model (six spaces + stack_all): empirical TEST coverage at α = .10 and .05 with Clopper–Pearson bars; nominal line; the [.86, .94] bar band shaded at α = .10. Mean interval width (relative to SD of truth) as text under each bar. | `e47_test_coverage.csv` | |
| E4.7b | **Conditional coverage heatmap.** Rows = model, columns = strata (party D/R/I, chamber, incumbency, n-tweets quartile Q1–Q4, plus the two shift conditions: k = 25 scarcity and incumbent-calibrated→challengers). Cell = coverage at α = .10, colour diverging around .90, cell text = coverage (n). | `e47_test_conditional.csv` | shift columns visually separated |
| E4.8 | **Active-learning curves.** One panel per space (tfidf, doc2vec, m2v_strip, behav): median DEV held-out-fold r vs n for the four strategies, `random` in grey with its n\*_AL marked (or "not reached by 60"); IQR band for random only. The n = 40 pick's TEST r per strategy as a marker at n = 40 (from 07). | `e48_summary.csv`, `e48_test_n40.csv` | These curves are DEV inner-fold numbers by design (§B.2); only the n = 40 markers are TEST |

Decision table figure (fig 0, optional): the §5 + Addendum B bar table with
met / not-met filled from `decision.json`.
