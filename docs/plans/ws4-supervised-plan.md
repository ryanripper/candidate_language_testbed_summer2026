# WS4 Execution Plan — Supervised Prediction of Planted Ideology

*Drafted 2026-09-08. Fourth workstream on the synthetic testbed, following the pattern of [extensions-execution-plan.md](extensions-execution-plan.md). Direction chosen by Ryan 2026-08-03 (planted-ideology prediction, over DW-NOMINATE real-data and election-outcome options); informal groundwork in [../../analyses/ws4-preanalysis/](../../analyses/ws4-preanalysis/ws4-preanalysis-writeup.md) (2026-08-07). Design choices below fixed by Ryan 2026-09-08: fold-wise label release; four headline experiments (label efficiency, text scarcity, incumbent→challenger, cross-family ensemble); feature spaces TF-IDF + w2v + doc2vec + Model2Vec + WS3 LLM + behavioral; continuous target.*

---

## 0. The question, and why accuracy is not it

WS0–WS3 established that unsupervised instruments already sit at the generator's ceiling: TF-IDF+SVD PC1 r = .974, behavioral (mean retweet-source ideology) r = .980, LLM ask-and-average r = .970 (n = 150), against a lexical ceiling of ≈ .973 (framing ↔ ideology). The 08-07 preanalysis then showed that a 5-fold ridge probe on *any* of five static spaces lands at r = .966–.973 — supervision erases every unsupervised deficit at n = 910, including GloVe's 47-point single-axis gap.

So "does supervision beat .974" is settled before it is asked. The questions supervision *can* answer on this testbed are about **cost and transfer**, which is where the real-data problem lives (labels exist only for incumbents via DW-NOMINATE; challengers are the ones worth predicting; many candidates have thin timelines):

1. **Label efficiency** — how few labeled candidates does each feature space need to reach ceiling?
2. **Text scarcity** — how does prediction degrade as tweets per candidate shrink, and which instruments degrade first?
3. **Incumbent→challenger extrapolation** — does a model trained only on "labeled incumbents" transfer to challengers, relative to a size-matched random training set?
4. **Cross-family ensemble** — the synthesis error-family finding (content family: LLM/TF-IDF/behavioral share residuals r ≈ .4–.6; style family: w2v/MiniLM/Model2Vec share theirs r ≈ .5–.6) becomes a testable hypothesis: a stack across families should beat any single instrument.

Two framing facts to state in every write-up. First, **effective n is 910 candidates, not 104.6k tweets**; every split is by candidate. Second, testbed prediction **overstates real-data transfer** — template artifacts are learnable — so results are read as upper bounds on what supervision buys, not forecasts.

**Session seed:** 20260908. Subfolder: `analyses/ws4-supervised/`.

---

## 1. Protocol — fold-wise label release (harness extension)

Supervision needs `true_ideology` as training data, which the WS0 seal was built to prevent. The blind habit is preserved at the *evaluation* boundary rather than the corpus boundary:

| Step | New artifact | What |
|---|---|---|
| 1.1 | `ws0-harness/05_ws4_splits.py` → `ws4_splits.json` | One-time candidate-level DEV / TEST partition, seed 20260908, stratified by party × chamber × incumbent × blind TF-IDF-score tercile. Ratio 2:1 (≈ 607 DEV / 303 TEST; decision D5). Also records the frozen inner 5-fold assignment within DEV. Never re-split. |
| 1.2 | `ws0-harness/06_label_release.py` → `ws4_dev_labels.parquet` | Reads `sealed_truth.parquet` once; writes candidate-level `true_ideology` for **DEV candidates only**. Hash-stamped in `seal_manifest.json`. This is the only truth any WS4 script touches before unseal. |
| 1.3 | `analyses/ws4-supervised/preregistration.md` | Filled from `preregistration_TEMPLATE.md` and dated **before** step 1.2 runs. |
| 1.4 | Unseal | TEST labels read exactly once, in the pre-registered final script, after every model, hyperparameter, draw schedule and figure spec is frozen. |

Rules: all model selection (ridge α, learner choice, ensemble weights) uses the inner 5 folds within DEV. TEST is scored once. Anything run against TEST afterwards is labeled *exploratory* in the write-up.

---

## 2. Feature spaces (candidate-level, all 910)

Every space is an **averaging construction** over a candidate's tweets, so it can be recomputed from any tweet subset (needed by E4.2). Fitting of vectorizers/embedders is unsupervised on the full blind corpus (transductive; stated). Dimensionality 100 throughout unless noted.

| Space | Family | Source / recipe | Notes |
|---|---|---|---|
| TF-IDF+SVD | content (lexical) | frozen WS0 vectorizer + SVD; candidate = mean tweet vector | champion baseline; frozen PC1 is the unsupervised reference |
| word2vec | content/style | canonical recipe, seed 20260720, freq-weighted token average | anchor; known style PC1 |
| doc2vec | content | PV-DBOW tagged by candidate_id; `infer_vector` for subsets | near-pure ideology PC1 in preanalysis; regenerated under seed 20260908 (D7) |
| Model2Vec | contextual (static) | WS1 Tier A `potion-base-8M`, mean tweet embedding | real-data default; WS1 residuals fell in the style family |
| Behavioral | behavioral | split-A features: mean retweet-source ideology (20 org ideologies are observable metadata), 20-dim RT-source share, RT share, log volume | the .980 instrument, and the "style half" of the ensemble hypothesis |
| LLM (WS3) | declared-scale | `scores_main.csv` mean of 5 reps | **n = 150 only** → secondary analysis E4.6 |

Dropped per preanalysis: fastText (w2v twin), GloVe (no unique strength). Style-PC projection is *not* applied by default — the preanalysis probes saw past the confound unaided; E4.1 tests this explicitly.

Learners: **ridge** (primary; α by inner CV on a log grid) and **HistGradientBoosting** (secondary, default depth, early stopping on inner folds) to check linear sufficiency (D6). No further learners are scored.

---

## 3. Experiments

All metrics on TEST unless stated; primary metric **Pearson r vs `true_ideology`** (`ws0/metrics.py::axis_recovery`), secondaries RMSE and Spearman ρ. Uncertainty: paired bootstrap over TEST candidates, 2,000 resamples, for every between-instrument comparison. Every stochastic draw schedule is seeded from 20260908 and recorded.

### E4.1 — Certified probes (the preanalysis, done properly)

Fit each space × learner on full DEV; score TEST. Report alongside the *unsupervised* reference computed on the same TEST candidates (frozen TF-IDF PC1, frozen behavioral score) so the comparison is apples-to-apples. Also: with vs without style-PC projection.

- Pre-registered expectation: all content probes within .01 of the lexical ceiling; supervision does not beat the behavioral .980; style projection changes r by < .005.
- Bar: TF-IDF and doc2vec probes r ≥ .965 on TEST (else the preanalysis was flattered by truth-visible axis selection — itself a finding).

### E4.2 — Label efficiency (headline)

For n ∈ {10, 20, 40, 80, 160, 320, all DEV}: draw n DEV candidates (stratified by party), fit ridge, score TEST; 50 draws per n per space. Report median and IQR of r; define **n\*** = smallest n whose median r is within .02 of the full-DEV r.

- Zero-label reference: the unsupervised PC1 (needs one label for sign orientation only) as a horizontal line.
- Hypothesis: TF-IDF and doc2vec reach n\* ≤ 40; word2vec and Model2Vec need more (their signal is smeared across PCs, per preanalysis finding 3); behavioral is near-flat (it is already a scale, not a space).
- Real-data translation: the incumbent count on any real cycle (~400) is far above every plausible n\*, which is the point.

### E4.3 — Text scarcity (headline)

For k ∈ {5, 10, 25, 50, 100, all} tweets per candidate, recompute every feature space from a random k-subset for **both** DEV and TEST candidates (candidates with fewer than k tweets keep all; report count), fit on DEV, score TEST; 20 draws per k. Two regimes: mixed (retweets included, corpus-wide assumption) and originals-only.

- Hypothesis: behavioral degrades fastest (≈ 26 % retweets → k = 5 carries ~1 retweet); content spaces degrade gracefully; doc2vec `infer_vector` may be noisier than mean-of-tweets at small k.
- k = 25 mixed is the WS3 bundle size — the LLM's n = 150 score is plotted as a reference point at that k (E4.6 support).

### E4.4 — Incumbent→challenger extrapolation (headline)

Train on DEV incumbents only (≈ 247), predict **TEST challengers** (≈ 180). Control: 50 random DEV subsets of the same size, same TEST challengers. **Shift penalty Δr = r(control) − r(incumbent-trained)**, with bootstrap CI. Repeat with chamber (House→Senate) as a second shift axis.

- Honest pre-registration: the generator does not condition text on incumbency (planted ideology distributions are near-identical: means .016 vs −.009, SD .64 both), so Δr ≈ 0 is the expected testbed answer. The experiment's value is the **harness** — it is the exact evaluation the real-data DW-NOMINATE phase (§9 of the parent plan) will run, and a null here calibrates what "no shift" looks like. If Δr is *not* ≈ 0, the generator has an incumbency artifact worth documenting.
- 30 Independents are all challengers; report them separately, never train on them alone.

### E4.5 — Cross-family ensemble (headline)

Stacking: inner-fold out-of-fold predictions from each single-space ridge become inputs to a non-negative ridge meta-learner (weights reported). Three stacks: content-only {TF-IDF, doc2vec}, style/behavioral-only {w2v, Model2Vec, behavioral}, and **cross-family** (all five). Score TEST.

- Bar: cross-family stack beats the best single instrument on TEST with a paired-bootstrap 95 % CI excluding zero, **and** beats the behavioral .980 reference. If the ceiling binds (likely: room above .980 is thin), the finding is that the generator's residual noise is the limit — quantified, not assumed.
- Replication of the synthesis finding on TEST: residual-correlation matrix of the single-space probes; pre-registered prediction is the same two-block structure (content vs style) found on the n = 150 pilot support.

### E4.6 — LLM score as a feature (secondary, n = 150)

On `subsample_150`: inner-CV within DEV ∩ 150 (≈ 100) adding the WS3 score to the cross-family stack; report on TEST ∩ 150 (≈ 50) as descriptive only. Any claim here is flagged low-power. If D1 scale-up is ever reopened, this becomes a full-support experiment without structural change.

---

## 4. Success criteria & decision rules

| Experiment | Success bar | Decision if met | Decision if not |
|---|---|---|---|
| E4.1 | content probes ≥ .965 on TEST | preanalysis certified; carry TF-IDF + doc2vec | preanalysis was truth-flattered; re-examine axis selection |
| E4.2 | at least one space has n\* ≤ 40 | real-data plan may treat DW-NOMINATE incumbents as ample labels; report cheapest space | label acquisition becomes a real-data risk; note in §9 sketch |
| E4.3 | content r ≥ .90 at k = 25 | thin-timeline candidates are predictable at WS3 bundle size | text scarcity is the binding constraint; real-data plan needs minimum-volume filters |
| E4.4 | \|Δr\| < .01 with CI covering 0 | harness validated; null calibrated | document generator incumbency artifact; treat as testbed bug or feature |
| E4.5 | cross-family > best single, CI excl. 0 | error-family hypothesis confirmed; ensembles enter the real-data pipeline | ceiling-bound: single content instrument suffices; ensembles deferred to real data |

Negative results are publishable under the D3 convention (separate per-workstream article).

---

## 5. Deliverables

`analyses/ws4-supervised/`: `preregistration.md` (dated before label release), `README.md`, scripts `01_features.py` (all spaces, subset-capable) · `02_probes.py` (E4.1) · `03_label_efficiency.py` · `04_text_scarcity.py` · `05_extrapolation.py` · `06_ensemble.py` · `07_unseal_evaluate.py` (single TEST read) · `08_figures.py`; `outputs/` (feature matrices as npz, all draw-level results as parquet, `decision.json`); `figures/` — fig 1 certified probe table, fig 2 label-efficiency curves with n\* markers, fig 3 scarcity curves (mixed vs originals-only), fig 4 shift-penalty forest plot, fig 5 ensemble weights + residual-correlation heatmap; `ws4-writeup.md`; `article_draft.md`. Harness additions: `05_ws4_splits.py`, `06_label_release.py`, updated `seal_manifest.json` and `ws0-harness/README.md`. Add a `REGENERATE.md` for excluded npz per repo convention. Writing sample stays frozen (D4).

---

## 6. Decision points for Ryan

*All five resolved 2026-09-08 by Ryan: recommended option in every case (2:1 split; GBM included; regenerate matrices under 20260908; incumbency + chamber; labeled post-unseal exploration allowed).*

- **D5 — DEV/TEST ratio.** 2:1 (≈ 607/303; recommended — TEST r CI half-width ≈ .005 at r = .97, DEV large enough for n = 320 draws) vs 1:1 (stronger test, weaker curves).
- **D6 — Nonlinear learner.** Include HistGradientBoosting as the secondary learner (recommended; ~minutes) or ridge-only.
- **D7 — Regenerate vs reuse.** Regenerate doc2vec and Model2Vec candidate matrices under seed 20260908 inside WS4 (recommended: the preanalysis npz is not in git and was built truth-visible, even though construction used no truth) vs load `ws4-preanalysis/outputs/candidate_vectors_all.npz`.
- **D8 — Extrapolation axes.** Incumbency only, or incumbency + chamber (recommended; free).
- **D9 — Post-unseal exploration.** Allow a clearly-labeled exploratory section after the single TEST read (recommended, mirrors WS1 "exploratory best" practice) or hard stop.

---

## 7. Sequencing

| Session | Work |
|---|---|
| 1 | Resolve D5–D9 → write `preregistration.md` → harness `05`/`06` → `01_features.py` (heaviest step: Model2Vec re-embed ≈ minutes, doc2vec retrain ≈ 10 min) → E4.1 on inner folds |
| 2 | E4.2–E4.5 on DEV inner folds (all draws), figure specs frozen, `07_unseal_evaluate.py` written and dry-run on DEV |
| 3 | Unseal → TEST scoring → figures → `ws4-writeup.md` → `article_draft.md` → memory/NOTES-readout update |

## 8. Risks

- **Ceiling compression.** Differences of .005 are the whole game; paired bootstrap is mandatory and effect sizes are reported as Δr with CIs, never as bare ranks.
- **Draw budget.** E4.2 (7 × 50 × 5 spaces × 2 learners) and E4.3 (6 × 20 × 5 × 2 regimes) are ~7k ridge fits on ≤ 607 × 100 — seconds each; GBM adds minutes. Fine in-session.
- **doc2vec nondeterminism.** `infer_vector` is stochastic; fix `seed` and `epochs`, report inference SD on a probe candidate.
- **Leakage via transductive fitting.** Vectorizers/embedders see TEST *text* (not labels); identical to WS0 baselines; stated in write-up.
- **Independents (n = 30).** Extreme on neither pole; report but never stratify on them alone.

## 9. Real-data hook

E4.4 is the DW-NOMINATE evaluation verbatim: labels for incumbents, predictions for challengers, on the frozen historical corpora named in the parent plan §9. E4.2 says how many labels are needed; E4.3 says how many tweets. Nothing here needs rework to port — only the gold standard changes.
