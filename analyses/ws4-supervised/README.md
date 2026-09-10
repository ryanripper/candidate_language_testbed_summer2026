# WS4 — Supervised prediction of planted ideology

Plan: [`docs/plans/ws4-supervised-plan.md`](../../docs/plans/ws4-supervised-plan.md).
Preregistration: [`preregistration.md`](preregistration.md) (locked 2026-09-08,
before label release; Addendum A dated the same day, pre-unseal; Addendum B
dated 2026-09-10, pre-unseal — adds E4.7 conformal coverage and E4.8 active
learning, seed 20260910).

**Status (2026-09-10): session 1 of 3 complete.** Harness extended, features
built, E4.1 run on DEV inner folds. Addendum B added; no session-1 output
needed regeneration (its inputs are the frozen inner-fold OOF residuals in
`outputs/e41_dev_oof.csv` and the full-DEV ridge models in
`intermediate/e41_models.joblib`). **TEST labels are still sealed.**

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
| 03 | `03_label_efficiency.py` | DEV | todo (E4.2) |
| 04 | `04_text_scarcity.py` | DEV | todo (E4.3) |
| 05 | `05_extrapolation.py` | DEV | todo (E4.4) |
| 06 | `06_ensemble.py` | DEV | todo (E4.5, E4.6) |
| 08 | `08_conformal.py` | DEV | todo (E4.7, Addendum B §B.1) — q̂ table from E4.1/E4.5 OOF residuals; DEV OOF coverage sanity check |
| 09 | `09_active_learning.py` | DEV | todo (E4.8, Addendum B §B.2) — four strategies on inner folds; frozen n = 40 model picks |
| 07 | `07_unseal_evaluate.py` | **TEST, once** | todo — also applies the frozen q̂ table and scores the n = 40 AL models (Addendum B §B.3) |
| 10 | `10_figures.py` | — | todo (was `08_figures.py`; renumbered when Addendum B claimed 08/09) |

**Pipeline order** (Addendum B §B.3: 08 and 09 run *before* 07; numbering
after 07 marks them as additions, not as post-unseal steps):

```
01 → 02 → 03 → 04 → 05 → 06 → 08 → 09 → [freeze figure specs] → 07 → 10
```

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
