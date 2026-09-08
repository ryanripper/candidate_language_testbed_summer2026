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
