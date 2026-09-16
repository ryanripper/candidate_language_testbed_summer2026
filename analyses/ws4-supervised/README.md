# WS4 — Supervised prediction of planted ideology

Plan: [`docs/plans/ws4-supervised-plan.md`](../../docs/plans/ws4-supervised-plan.md).
Preregistration: [`preregistration.md`](preregistration.md) (locked 2026-09-08,
before label release; Addendum A dated the same day, pre-unseal; Addendum B
dated 2026-09-10, pre-unseal — adds E4.7 conformal coverage and E4.8 active
learning, seed 20260910).

**Status (2026-09-15): session 2 of 3 complete.** Scripts 03–06, 08, 09 run
on DEV; figure specs frozen (`figure_specs.md`); every frozen model's TEST
predictions computed label-free by `07a_freeze_predictions.py`
(`intermediate/test_predictions.joblib`, 7,416 arrays); `07_unseal_evaluate.py`
written and dry-run against DEV (`intermediate/dryrun/`, never opened the
sealed file). Addendum C (implementation clarifications) added to the
preregistration. **TEST labels are still sealed** — `seal_manifest.json` has no
`ws4_unseal` entry and `outputs/decision.json` does not exist.
Session 3 = `python 07_unseal_evaluate.py --unseal` → `10_figures.py` →
`ws4-writeup.md` → `article_draft.md`.

## Protocol

| Layer | File | Truth contact |
|---|---|---|
| Split | `ws0-harness/ws4_splits.json` — DEV 607 / TEST 303 (2:1), stratified party × chamber × incumbent × blind-TF-IDF tercile; inner 5-fold within DEV (121–122 each); seed 20260908 | none (blind) |
| Labels | `ws0-harness/ws4_dev_labels.parquet` — `true_ideology` for DEV only; hash in `seal_manifest.json` | read #1 (`06_label_release.py`) |
| Evaluation | `scripts/07_unseal_evaluate.py` (not yet written) | read #2 — once |

## Scripts

| # | Script | Reads labels? | Status |
|---|---|---|---|
| — | `ws4lib.py` | DEV only | shared: paths, `Featurizer` (any tweet subset → candidate matrix), ridge/GBM, paired bootstrap |
| 01 | `01_features.py` | no | done — tweet-level caches → `intermediate/`; `outputs/features_full.npz`; `outputs/unsupervised_refs.csv` |
| 02 | `02_probes.py` | DEV | done — E4.1 OOF on DEV inner folds; final DEV models → `intermediate/e41_models.joblib` |
| 03 | `03_label_efficiency.py` | DEV | done — E4.2 out-of-draw r on DEV; frozen draw schedule `outputs/e42_draw_schedule.json` |
| 04 | `04_text_scarcity.py` | DEV | done — E4.3 DEV OOF per k × regime × draw; frozen subset schedule `intermediate/e43_schedule.npz` + cached doc2vec inference |
| 05 | `05_extrapolation.py` | DEV | done — E4.4 shift + 50 control models (`intermediate/e44_models.joblib`); incumbent OOF residuals for E4.7(b) |
| 06 | `06_ensemble.py` | DEV | done — E4.5 three stacks (`intermediate/e45_meta.joblib`, weights in `outputs/e45_meta_weights.json`); E4.6 on DEV∩150 |
| 08 | `08_conformal.py` | DEV | done — q̂ table `outputs/e47_qhat.json` (all-DEV and incumbent-calibrated); LOFO coverage sanity |
| 09 | `09_active_learning.py` | DEV | done — 4 strategies × 4 spaces × 100 curves; n = 40 picks `outputs/e48_n40_picks.json` |
| 07a | `07a_freeze_predictions.py` | DEV | done — label-free TEST predictions for every frozen model → `intermediate/test_predictions.joblib` (Addendum C) |
| 07 | `07_unseal_evaluate.py` | **TEST, once** | written + dry-run; **not yet run with `--unseal`** — fits nothing, scores the frozen arrays, writes `decision.json` + manifest entry |
| 10 | `10_figures.py` | — | todo (specs frozen in `figure_specs.md`) |

**Pipeline order** (Addendum B §B.3: 08 and 09 run *before* 07; numbering
after 07 marks them as additions, not as post-unseal steps):

```
01 → 02 → 03 → 04 → 05 → 06 → 08 → 09 → [freeze figure specs] → 07a → 07 → 10
```

Sandbox note: a tool call is capped at ≈3 min, so 03/04/05/07a/09 checkpoint
to `intermediate/` and resume; re-run the script until it prints its summary.
`intermediate/` is now ≈1 GB (gitignored).

Seeds: `ws4lib.SEED = 20260908` for everything preregistered on 2026-09-08
(splits, draws in 03–06, paired bootstrap §3); `ws4lib.SEED_B = 20260910` for
the Addendum B draws in 08/09 only.

## E4.1 on DEV (out-of-fold, ridge unless noted) — development numbers, not the certified result

| space | r | note |
|---|---|---|
| m2v (raw text) | .978 | sees `RT @OrgHandle:` prefixes → retweet-source identity (Addendum A) |
| tfidf (mean of tweet rows) | .973 | GBM .974; WS0 candidate-doc construction .967 ridge / .974 GBM |
| doc2vec | .972 | |
| m2v_strip (RT prefixes + mentions removed) | .969 | the m2v edge is the prefix |
| w2v | .965 | |
| behav (split A) | .963 | GBM .968 |
| *frozen TF-IDF PC1 (unsupervised)* | *.973* | |
| *behavioral split-A source mean (unsupervised)* | *.965* | full-corpus reference is .980 |

Style-PC projection lowers every ridge probe (−.001 to −.017): the supervised
reader does not need the confound gate (preanalysis finding 5 holds).
GBM trails ridge badly on the dense 100–256-d spaces (n = 607) and helps only on
`behav` and TF-IDF.

Residual correlations (DEV OOF): the four text spaces form one block
(.61–.84); `behav` is separate (.33–.40). That is a **text-vs-behavior**
partition, not the content-vs-style partition the synthesis found on n = 150 —
to be re-examined on TEST in E4.5.

Excluded binaries: see [`REGENERATE.md`](REGENERATE.md).

## Session-2 DEV development numbers (not results; TEST comes from 07)

**E4.2 label efficiency** (ridge, median out-of-draw r over 50 draws; n\* = first n within .02 of the full-DEV OOF r):

|   n |   behav |   doc2vec |   m2v |   m2v_strip |   tfidf |   w2v |
|----:|--------:|----------:|------:|------------:|--------:|------:|
|  10 |   .930  |     .795 | .892 |       .855 |   .355 | .830  |
|  20 |   .938 |     .878 | .916 |       .894 |   .508 | .874 |
|  40 |   .949 |     .926 | .937 |       .921 |   .712 | .898 |
|  80 |   .955 |     .943 | .954 |       .942 |   .865 | .918 |
| 160 |   .959 |     .962 | .966 |       .958 |   .946 | .947 |
| 320 |   .961 |     .969 | .975 |       .967 |   .970 | .963 |
| 607 |   .963 |     .972 | .978 |       .969 |   .973 | .965 |

n\*: behav 40, doc2vec / m2v / m2v_strip / w2v 160, **tfidf 320**. The §5
expectation (tfidf / doc2vec n\* ≤ 40) is wrong on DEV: with standardized
100-d inputs and n = 10 the ridge cannot find the TF-IDF axis that the
zero-label PC1 (.973) hands over for free — supervision only overtakes the
unsupervised instrument at n ≈ 320. `behav` is nearly flat, as predicted.
GBM is degenerate at n ≤ 40 (Addendum C §3). Candidate post-unseal
exploration (D9): unstandardized / PC-truncated ridge at small n.

**E4.3 text scarcity** (ridge, median DEV OOF r over 20 draws):

| k | mixed: tfidf / doc2vec / m2v / m2v_strip / w2v / behav | originals: tfidf / doc2vec / m2v / behav |
|---|---|---|
| 5 | .777 / .782 / .832 / .768 / .749 / .600 | .779 / .782 / .773 / −.05 |
| 10 | .863 / .866 / .896 / .855 / .841 / .755 | .866 / .868 / .864 / −.02 |
| 25 | .934 / .930 / .947 / .930 / .922 / .892 | .933 / .930 / .931 / −.04 |
| 50 | .962 / .954 / .969 / .957 / .951 / .940 | .955 / .953 / .952 / −.03 |
| 100 | .971 / .964 / .976 / .967 / .962 / .960 | .962 / .959 / .959 / −.01 |
| all | .973 / .972 / .978 / .969 / .965 / .963 | .964 / .960 / .961 / −.02 |

k = 25 mixed content ≈ .93 (bar .90). In the originals-only regime `m2v` ==
`m2v_strip` to 3 decimals — the raw-m2v edge is entirely the `RT @` prefix —
and `behav` is dead by construction. Candidates with < k tweets: 5 (k = 25),
97 (k = 50), 457 (k = 100) mixed; 36 / 246 / 643 originals.

**E4.4 shift** (ridge; DEV challengers / senators; Δ = control median − shift-trained):
incumbency Δ ∈ [−.004, +.003] across spaces; chamber Δ ∈ [−.004, 0]. Null as
preregistered; the TEST CI is the result.

**E4.5 stacks** (DEV meta-OOF r): content .976, style .981, **cross-family
.982** vs best single m2v .978 and behav split-A .963 (full-corpus .980).
Cross-family weights: m2v .47, doc2vec .29, behav .27, tfidf .01, w2v 0.
E4.6 (DEV∩150, n = 98): stack_all .9816 → +llm .9818 (llm weight .14);
llm alone .968. Low power by design.

**E4.7 conformal** (q̂ at α = .10 / .05, all-DEV calibration; width relative
to SD of truth): tfidf .238/.295 (0.73 SD) · doc2vec .253/.324 · w2v
.273/.327 · m2v .223/.272 · m2v_strip .265/.308 · behav .272/.345 · stack_all
.194/.234 (0.60 SD). Leave-one-fold-out DEV coverage .895–.905 at α = .10.
Incumbent-calibrated q̂ is wider (.28–.32 at α = .10; n = 246 models).

**E4.8 active learning** (DEV inner folds): no content space reaches n\*_AL
by n = 60 under any strategy (consistent with E4.2's n\* = 160–320), so the
n\* clause of the bar cannot be met — "no reach", per §B.2. AULC Δ vs random:
tfidf uncertainty +.050 [.037, .063], hybrid +.032, diversity +.015 (all
CI > 0); doc2vec all three strategies *below* random (−.011 to −.014, CI
< 0); m2v_strip null; behav diversity/hybrid below random. The §B.2
expectation (diversity > hybrid > uncertainty > random) is wrong on DEV:
uncertainty sampling is the only strategy that helps, and only on TF-IDF.
