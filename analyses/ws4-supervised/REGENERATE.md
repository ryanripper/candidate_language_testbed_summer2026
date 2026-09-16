# Regenerating the excluded WS4 artifacts

`intermediate/` (~310 MB: tweet-level caches for tfidf / w2v / m2v / m2v_strip,
`doc2vec.model`, `e41_models.joblib`) and `ws0-harness/ws4_dev_labels.parquet`
are not committed.

Requires a verified WS0 harness (`ws0-harness/REGENERATE.md`) plus
`pip install scikit-learn scipy gensim model2vec pyarrow joblib`.

```bash
cd ws0-harness
python 05_ws4_splits.py        # refuses to overwrite ws4_splits.json (frozen); already committed
python 06_label_release.py     # DEV-only labels; requires dated preregistration.md
cd ../analyses/ws4-supervised/scripts
python 01_features.py          # ~2 min; downloads potion-base-8M on first use
python 02_probes.py            # E4.1 on DEV inner folds
```

Seeds: 20260720 for the w2v / TF-IDF anchors, 20260908 otherwise. gensim runs
`workers=1` with a crc32 hashfxn — do not raise `workers`.

## Session 2 additions (2026-09-15)

`intermediate/` grew to ≈1 GB: `e43_schedule.npz` + `e43_doc2vec/` (per-draw
doc2vec inference), `e44_models.joblib` (≈250 MB, GBM controls), `e45_meta.joblib`,
per-stage checkpoints `e4*_partial_*.joblib` / `e4x_pred_*.joblib`, and
`test_predictions.joblib` (the frozen TEST predictions 07 scores). To rebuild:

```bash
cd analyses/ws4-supervised/scripts
python 03_label_efficiency.py; python 04_text_scarcity.py; python 05_extrapolation.py
python 06_ensemble.py; python 08_conformal.py; python 09_active_learning.py
python 07a_freeze_predictions.py          # label-free; ~25 min total
python 07_unseal_evaluate.py --dry-run    # exercises 07 without opening sealed_truth
```

Each script resumes from its checkpoints; re-run until it prints its summary.
`outputs/*.csv|json` (schedules, DEV summaries, q̂ table, AL picks) are committed.
