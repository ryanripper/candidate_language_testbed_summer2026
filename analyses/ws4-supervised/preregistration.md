# Preregistration — Workstream 4: Supervised prediction of planted ideology

*Filled in completely before `ws0-harness/06_label_release.py` is run and before
any WS4 script reads `sealed_truth.parquet`. Plan: `docs/plans/ws4-supervised-plan.md`
(drafted 2026-09-08; decisions D5–D9 resolved 2026-09-08 with the recommended options).*

**Date locked:** 2026-09-08
**Session seed:** 20260908

## 1. Question

Given that unsupervised instruments already recover planted ideology at the
generator's ceiling (TF-IDF PC1 r = .974, behavioral r = .980), what does
supervision buy in **label efficiency**, **robustness to text scarcity**,
**incumbent→challenger transfer**, and **cross-family ensembling** — the
questions that matter for the real-data (DW-NOMINATE-labeled) use case?

## 2. Methods being compared

Feature spaces (candidate-level, 100-dim unless noted; every one is a mean
over the candidate's tweets so it can be recomputed from tweet subsets):

| Space | Recipe |
|---|---|
| `tfidf` | WS0 recipe: tokenizer from `02_freeze_baselines.py`, `TfidfVectorizer(min_df=5, sublinear_tf=True)` fit on the 910 candidate documents (full blind corpus), `TruncatedSVD(100, random_state=20260720)`; candidate vector for a tweet subset = mean of the subset's tweet-level TF-IDF rows projected through the same SVD |
| `w2v` | WS0 canonical word2vec (sg=1, 100d, window 5, min_count 5, 10 epochs, workers=1, seed 20260720, crc32 hashfxn); candidate vector = frequency-weighted token average over the subset |
| `doc2vec` | gensim PV-DBOW, 100d, window 5, min_count 5, 20 epochs, workers=1, seed 20260908, tweets tagged by `candidate_id`; full-corpus candidate vector = learned tag vector; subset vector = `infer_vector` over the concatenated subset tokens (epochs=50, seeded) |
| `m2v` | Model2Vec `minishlab/potion-base-8M` (WS1 Tier A), mean of tweet embeddings (256-d native) |
| `behav` | from the WS0 split-A tweets only: mean retweet-source ideology (org ideologies from `analyses/ws3-llm-scaling/outputs/org_ideologies.json`, observable metadata), 20-dim retweet-source share vector, retweet share, log(1 + n tweets); candidates with zero retweets in the subset get source-mean = 0 and a missing-indicator |
| `llm` | WS3 `scores_main.csv` `llm_score` (n = 150 only; §E4.6 secondary) |

Learners: `Ridge` (α ∈ 10^{−3…3}, 13-point log grid, chosen by inner 5-fold
CV within DEV; features standardized on the training fold) — **primary**; and
`HistGradientBoostingRegressor` (default depth, `early_stopping=True`,
`random_state=20260908`) — **secondary**.

Ensembles (E4.5): non-negative ridge meta-learner (`Ridge(positive=True,
fit_intercept=True)`, α by inner CV) over out-of-fold single-space ridge
predictions. Three stacks: content {tfidf, doc2vec}; style/behavioral
{w2v, m2v, behav}; cross-family {all five}.

Nothing else is scored. Style-PC projection is evaluated only as the
with/without contrast in E4.1.

## 3. Primary metric

Pearson r between predicted and true ideology on TEST candidates
(`ws0-harness/metrics.py::axis_recovery`, `pearson_r` field). All
between-instrument comparisons: paired bootstrap over TEST candidates,
2,000 resamples, seed 20260908, reporting Δr with 95 % percentile CI.

## 4. Comparison set / baselines

Computed on the same TEST candidates from frozen arrays
(`ws0-harness/baselines/axis_scores.csv` for TF-IDF PC1; behavioral score
recomputed with the WS0 split-A recipe used in WS3/synthesis):
unsupervised TF-IDF PC1 (full-corpus r = .974) and behavioral (full-corpus
r = .980). Lexical ceiling ≈ .973 (framing ↔ ideology).

## 5. Success criteria & decision rules

| Exp | Bar | Met → | Not met → |
|---|---|---|---|
| E4.1 certified probes | ridge on `tfidf` and `doc2vec` r ≥ .965 on TEST | preanalysis certified; carry both | preanalysis was truth-flattered; document |
| E4.2 label efficiency | ≥ 1 space with n\* ≤ 40, where n\* = smallest n whose median r (50 draws) is within .02 of the full-DEV r for that space | incumbent labels are ample on real data | label acquisition flagged as real-data risk |
| E4.3 text scarcity | best content space r ≥ .90 at k = 25 (mixed regime) | thin timelines predictable at WS3 bundle size | minimum-volume filter needed for real data |
| E4.4 extrapolation | Δr = r(random-control) − r(incumbent-trained) has \|Δr\| < .01 and CI covers 0 | harness validated; null calibrated | generator incumbency artifact; document |
| E4.5 ensemble | cross-family stack beats best single instrument on TEST with CI excluding 0 AND beats behavioral reference on TEST | error-family hypothesis confirmed | ceiling-bound; single content instrument suffices |

Pre-registered expectations (stated so they can be wrong): E4.1 all content
probes within .01 of ceiling and no probe beats behavioral; style projection
|Δr| < .005. E4.2 `tfidf`/`doc2vec` n\* ≤ 40, `w2v`/`m2v` larger, `behav` flat.
E4.3 `behav` degrades fastest; content gracefully. E4.4 Δr ≈ 0 (generator
does not condition on incumbency; planted-ideology means .016 vs −.009).
E4.5 likely ceiling-bound; residual-correlation matrix reproduces the two
synthesis families (content vs style).

## 6. Secondary / diagnostic metrics

RMSE and Spearman ρ alongside every r; IQR across draws; ensemble weights;
residual-correlation matrix of single-space probes on TEST; `doc2vec`
`infer_vector` SD on one probe candidate; chamber (House→Senate) as second
shift axis in E4.4; E4.6 LLM-as-feature on `subsample_150` (DEV∩150 inner CV,
TEST∩150 descriptive, flagged low-power); Independents (n = 30) reported
separately in E4.4.

## 7. Unseal plan

Splits: `ws0-harness/ws4_splits.json` (this session; seed 20260908):
candidate-level DEV/TEST 2:1, stratified by party × chamber × incumbent ×
blind-TF-IDF-score tercile; frozen inner 5-fold assignment within DEV
(same strata). WS0 `tweet_split_ab.parquet` supplies split-A tweets for
`behav`. `subsample_150` for E4.6.

Truth access, exactly two reads:
1. `ws0-harness/06_label_release.py` → `ws4_dev_labels.parquet`
   (candidate-level `true_ideology`, DEV only; hash recorded in
   `seal_manifest.json`). Every fitting / tuning / draw-schedule script
   reads only this file.
2. `analyses/ws4-supervised/scripts/07_unseal_evaluate.py` → TEST labels,
   once, after scripts 01–06 and the figure specs are frozen. All TEST
   numbers in the write-up come from this single run. Any later TEST
   contact is labeled *exploratory* (D9).

## 8. Confound gate

Not a distance workstream, but the style axis is still diagnosed: E4.1
reports every probe with and without projecting out the PC most correlated
with retweet share (`metrics.project_out`), on each space. Expectation: the
supervised reader sees past the confound unaided (preanalysis finding 5).

---

## Addendum A — 2026-09-08 (pre-unseal; TEST untouched)

Added after the E4.1 DEV inner-fold run, before any TEST contact:

- **`m2v_strip` diagnostic variant.** Raw-text Model2Vec embeds the `RT @OrgHandle:`
  prefix, so it sees retweet-source identity — the WS3 corpus quirk — which the
  tokenized spaces (mentions stripped) do not. A variant with RT prefixes and
  @mentions removed before embedding is added as a diagnostic column in E4.1 and
  E4.3. `m2v` (raw) remains the preregistered Model2Vec instrument; any headline
  claim about "contextual embeddings" is reported with both.
- **`tfidf_doc` diagnostic column.** The WS0 candidate-document construction is
  reported next to the subset-capable mean-of-tweet-rows `tfidf` construction so
  the two can be compared on DEV/TEST. `tfidf` (mean of tweet rows) remains the
  preregistered instrument because it is the one E4.3 can recompute from subsets.
- **Behavioral reference.** The unsupervised behavioral reference in §4 is the
  **split-A** source mean (matching the `behav` feature's data), not the
  full-corpus .980; the full-corpus value is additionally reported for context.

DEV-only observation recorded for honesty (not a TEST result): ridge OOF r on
DEV — m2v .978, tfidf .973, doc2vec .972, m2v_strip .969, w2v .965, behav .963;
frozen TF-IDF PC1 .973. The m2v edge disappears when RT prefixes are stripped.

---

## Addendum B — 2026-09-10 (pre-unseal; TEST untouched)

*Drafted after Addendum A, before scripts 03–06 are run and before any TEST
contact. Adds two experiments that turn the point estimates above into
reliability claims: does the supervised probe know when it is wrong
(E4.7), and can it choose which labels it needs (E4.8)? Both reuse the
frozen DEV inner folds and the E4.1 feature caches; neither adds a new
feature space or learner. Draw seed for this addendum: 20260910.*

### B.1 E4.7 — Conformal coverage of the ridge probe

**Method.** Split conformal regression on top of the primary ridge probe,
per space (`tfidf`, `doc2vec`, `w2v`, `m2v`, `m2v_strip`, `behav`, and the
cross-family stack from E4.5).

1. Calibration scores = out-of-fold absolute residuals |y − ŷ| from the
   frozen 5-fold inner CV on DEV (n = 607), with α chosen as in §2.
2. q̂_α = the ⌈(n + 1)(1 − α)⌉ / n empirical quantile of the scores
   (finite-sample correction), for α ∈ {.10, .05}.
3. Final model refit on all of DEV; TEST interval = ŷ ± q̂_α.

Using OOF residuals as calibration scores is mildly conservative (the
refit model trains on 5/4 as much data as each fold model); this is
stated, not corrected. No conformalized quantile regression, no
locally-weighted scores, no adaptive α — the plain version is the
preregistered one.

**Metrics** (all on TEST, one read via `07_unseal_evaluate.py`):

- Marginal empirical coverage at α = .10 and .05, with Clopper–Pearson 95 % CI.
- Mean interval width, and width relative to the SD of true ideology.
- Conditional coverage by party, chamber, incumbency, Independent status,
  and text-volume quartile (n tweets).
- **Coverage under shift**, reusing the E4.3 and E4.4 designs with the
  DEV-calibrated q̂ held fixed: (a) TEST candidates featurized from
  k = 25 tweets (scarcity regime); (b) q̂ calibrated on DEV incumbents
  only, evaluated on TEST challengers (incumbent→challenger).

**Bar and decision rule** (added row for §5):

| Exp | Bar | Met → | Not met → |
|---|---|---|---|
| E4.7 conformal | for the primary content spaces (`tfidf`, `doc2vec`) marginal coverage at α = .10 lies in [.86, .94] on TEST (binomial 95 % band for n = 303 around .90 is ≈ .866–.934, rounded) | probe is calibrated; intervals usable for real-data triage | report over/under-coverage and its conditional pattern; do not tune α post hoc |

**Expectations** (stated so they can be wrong): marginal coverage nominal
for all text spaces; `behav` nominal but widest (its residuals are the
largest in E4.1); conditional coverage lowest for Independents and the
bottom text-volume quartile; k = 25 scarcity produces **under**-coverage
(≥ .05 below nominal) because q̂ was calibrated on full-volume residuals;
incumbent→challenger coverage stays nominal (consistent with the E4.4
Δr ≈ 0 expectation — if E4.4 finds an incumbency artifact, E4.7(b) should
show it as under-coverage on the same candidates).

### B.2 E4.8 — Label-efficient active learning

**Method.** Pool-based active learning simulated inside the frozen inner
folds, so no TEST contact until unseal. For each inner fold, pool = the
four training folds (~485 candidates), evaluation = the held-out fold
(~121). Spaces: `tfidf`, `doc2vec`, `m2v_strip`, `behav` (the two
preregistered content probes, the de-leaked contextual probe, and the
non-text control). Learner: the primary ridge, α re-selected by 3-fold CV
inside the labeled set at each step (5-fold is not defined below n = 15).

Protocol per (space, fold, draw): seed set n₀ = 10 (stratified random on
party × chamber), batch size 5, stop at n = 60. 20 draws per fold (100
curves per strategy per space). Seed 20260910 + draw index.

Strategies (fixed list; nothing added post hoc):

| Strategy | Query rule |
|---|---|
| `random` | control: uniform without replacement |
| `uncertainty` | largest predictive variance under the Bayesian-ridge reading of the current model, xᵀ(XᵀX + αI)⁻¹x, on standardized features |
| `diversity` | k-center greedy in the standardized feature space (farthest-first from the labeled set) |
| `hybrid` | `diversity` for the first 4 batches, `uncertainty` thereafter |

**Metrics.**

- Learning curve: median (IQR) r on the held-out fold at each n.
- **n\*_AL** per strategy: smallest n whose median r is within .02 of the
  full-DEV r for that space — the same definition as E4.2, so E4.2's
  `random` curve is the control and is recomputed here under the same
  seeds rather than copied.
- Area under the learning curve (AULC) over n ∈ [10, 60], paired
  bootstrap over draws, Δ vs `random` with 95 % CI.
- At unseal: the n = 40 model from each strategy (one median-draw pick,
  chosen on DEV before unseal) scored on TEST for a single r; descriptive.

**Bar and decision rule** (added row for §5):

| Exp | Bar | Met → | Not met → |
|---|---|---|---|
| E4.8 active learning | on ≥ 1 content space, `diversity` or `hybrid` reaches n\* at ≤ 0.75 × the `random` n\* AND AULC Δ > 0 with CI excluding 0 | query strategy is worth the engineering on real data | random labeling suffices at this label budget; report the crossover n if any |

**Expectations.** Because E4.1 puts the probes at the generator ceiling
and E4.2 is expected to find n\* ≤ 40 for content spaces, the headroom is
small and concentrated at n ≤ 25. `diversity` > `hybrid` > `uncertainty` >
`random` at n = 15–20; differences vanish by n = 40. `uncertainty` alone may
under-perform `random` early (it chases leverage points, which in a
near-linear ceiling-bound problem are outliers, not information). `behav`
shows no strategy effect. If `random` already hits n\* ≤ 20, the bar's
0.75 factor becomes a difference of ≤ 5 labels and the result is reported
as "no room for AL" rather than as a failed bar.

### B.3 Implementation and truth access

- New scripts: `08_conformal.py` (E4.7 calibration on DEV; writes q̂ table
  and DEV OOF coverage as a sanity check) and `09_active_learning.py`
  (E4.8 curves on DEV). Both read labels only via
  `ws4lib.load_dev_labels()`. `07_unseal_evaluate.py` is extended to apply
  the frozen q̂ table and score the frozen n = 40 AL models; it remains the
  single TEST read. Script numbering after 07 is deliberate: 08/09 run
  before 07 in the pipeline order, and `README.md` records the order.
- Figure specs to freeze before unseal: fig E4.7a coverage-by-α with CI
  bars per space; fig E4.7b conditional coverage heatmap (space × stratum);
  fig E4.8 learning curves, four strategies, one panel per space, `random`
  drawn in grey with its n\* marked.
- No change to §1–§8 above. E4.5's cross-family stack is the only ensemble
  conformalized; no ensemble-of-conformal or conformal-of-ensemble
  variants.
- Deviations from this addendum after unseal are labeled *exploratory*
  per D9, as for the main preregistration.

**Why these two.** The workstream's claim is that a supervised probe is a
*reliability method* checked against planted truth, not just a high-r
predictor. E4.7 tests whether its stated uncertainty is honest, including
under the two shifts (scarcity, incumbency) the plan already studies;
E4.8 tests whether the label-efficiency result of E4.2 can be improved by
choosing labels rather than drawing them. Both are cheap on this corpus
and directly transfer to the DW-NOMINATE real-data recipe, where labels
exist only for incumbents and timelines vary in volume.

### B.4 Clarifications — 2026-09-10, later the same day (pre-unseal; TEST untouched)

Recorded after re-reading session-1 code against B.1–B.3 and before any of
scripts 03–09 is written. No session-1 output is regenerated: B.1's
calibration scores are the frozen inner-fold ridge OOF residuals already in
`outputs/e41_dev_oof.csv` (columns `{space}|raw|ridge`), and B.1 step 3's
"refit on all of DEV" models are the `{space}|ridge` entries already in
`intermediate/e41_models.joblib` from 02_probes.py. Only the cross-family
stack's OOF predictions are still to come (06).

1. **E4.7(b) model.** "q̂ calibrated on DEV incumbents only, evaluated on
   TEST challengers" uses the **E4.4 incumbent-trained ridge** (fit on DEV
   incumbents, α by inner CV among incumbents) for both the point prediction
   and the calibration residuals — i.e. inner-fold OOF residuals computed
   within the DEV-incumbent subset. It does *not* mean filtering the
   all-DEV model's residuals to incumbents. This is the reading under which
   E4.7(b) is a shift test paired with E4.4; the all-DEV model's coverage on
   TEST challengers is reported alongside as context, not as the bar.
2. **Seeds.** `ws4lib.SEED = 20260908` stays the seed for everything in
   §2–§7 (splits, 03–06 draws, §3 paired bootstrap). The Addendum B seed
   20260910 is exposed as `ws4lib.SEED_B` and is used only for the E4.7/E4.8
   draws in 08/09. The E4.8 `random` control is drawn under `SEED_B` (per
   B.2), so it is a recomputation, not a copy of E4.2.
3. **Script numbering.** `08_conformal.py` and `09_active_learning.py` take
   08/09 as stated in B.3; the figure script previously listed in
   `README.md` as `08_figures.py` becomes `10_figures.py`. Pipeline order
   recorded in `README.md`: 01→02→03→04→05→06→08→09→(freeze figure
   specs)→07→10.
4. **Library changes** (`scripts/ws4lib.py`), all additive:
   `fit_ridge(..., n_splits=5)` gains the `n_splits` argument for B.2's
   3-fold α selection (default unchanged, so 02_probes.py is byte-for-byte
   the same computation); new helpers `conformal_quantile`, `coverage_ci`,
   `ridge_predictive_variance`, `fit_ridge_al`, `kcenter_greedy`,
   `stratified_seed_set`, and the B.1/B.2 constants. None reads labels.
5. **Conditional coverage strata** (B.1): party, chamber, incumbency and
   Independent status come from `ws0-harness/baselines/candidate_metadata.csv`;
   text-volume quartiles are cut on `n_tweets` from the same file, computed
   over TEST candidates at unseal.
