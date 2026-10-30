# Regenerating the excluded WS0 artifacts

Four groups of files in this folder are excluded from version control (see the
repository `.gitignore`) because they are large and fully derived. Everything
here is deterministic — same inputs, same seeds, same outputs.

Run from inside `ws0-harness/`:

| Excluded file | Rebuild with | Notes |
|---|---|---|
| `blind_corpus.parquet`, `sealed_truth.parquet` | `python 01_seal_corpus.py` | Reads `../data/synthetic-candidate-tweets/synthetic_candidate_tweets_2022.csv.gz`. Also rewrites the committed `seal_manifest.json`; the load-bearing stamp is `source_sha256` (guarded in-script and pinned as a constant in `04_verify_harness.py`) — the parquet hashes can legitimately drift with pandas/pyarrow versions, so a parquet-hash diff alone does not mean a different split. |
| `tweet_split_ab.parquet` | `python 03_build_splits.py` | Seed 20260725. The A/B assignment reproduces exactly (content-pinned in `pins.py`). **`splits.json` is NOT rewritten** by default — see the frozen-subsample note below. |
| `baselines/candidate_vectors.npz`, `baselines/D_w2v_raw.npy`, `baselines/D_w2v_corrected.npy`, `baselines/D_tfidf.npy` | `python 02_freeze_baselines.py` | ~10 minutes. Word2vec is retrained single-threaded with a deterministic hashfxn (seed 20260720) — **do not** raise `workers`, it destroys reproducibility. |

**Frozen subsample (2026-10-30 audit):** the committed `splits.json` is the
canonical 2026-07-25 draw and every n=150 result (WS2 judge, WS3 pilot,
synthesis) used exactly those ids. Re-running `03_build_splits.py` does *not*
reproduce it (122/150 ids match: one candidate sits on a within-party tercile
boundary of `axis_scores.csv` and flips, reshuffling two strata's RNG draws —
not a library-version effect). So 03 now only reports whether the recomputed
subsample matches the pin and leaves `splits.json` alone; `--force` overwrites
it, after which `04_verify_harness.py` fails by design. Do not regenerate
`splits.json`; it is a committed input, not a derived artifact.

Then always:

```bash
python 04_verify_harness.py    # must print HARNESS VERIFIED
```

`04_verify_harness.py` regenerates `baselines/baseline_validation.csv` and checks
it against the frozen 07-20 reference values. If it prints anything other than
HARNESS VERIFIED, the rebuild does not match the artifacts the committed results
were computed from, and downstream numbers should not be compared to the ones in
the write-ups.

**Committed and not regenerable from here:** `baselines/pca_scores.npz`,
`baselines/axis_scores.csv`, `baselines/candidate_metadata.csv`,
`baselines/frozen_validation_20260720.csv`, `baselines/manifest.json`.
