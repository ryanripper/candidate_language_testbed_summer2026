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
