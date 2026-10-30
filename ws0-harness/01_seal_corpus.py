"""
01_seal_corpus.py — WS0.1
-------------------------
Shared-harness seal script (extensions-execution-plan.md, step 0.1).

Splits the synthetic corpus into:
  blind_corpus.parquet  — every observable column (what workstreams may read)
  sealed_truth.parquet  — tweet_id/candidate_id + true_topic, true_framing,
                          true_ideology (read ONLY at pre-registered
                          validation steps)

Both files are hash-stamped into seal_manifest.json. Any workstream can
verify it is reading the canonical blind corpus by checking the sha256.

Blind protocol (established 2026-07-20, embeddings-pca-analysis):
design decisions are made against blind_corpus only; sealed_truth is opened
once per workstream, at the validation step named in its preregistration.md.

Session seed convention: 20260725 (run date). The seal itself is
deterministic — no randomness here.

Rebuild semantics (2026-10-30): a plain run rebuilds the two parquet views
and verifies their CONTENT against the pins in pins.py; if they match it
leaves the committed seal_manifest.json untouched (parquet byte hashes drift
with pandas/pyarrow, and the manifest also carries downstream entries such
as ws4_label_release / ws4_unseal that must not be lost). Only --reseal
rewrites the manifest, and even then it preserves keys it does not own.
"""

import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

import pins

HERE = Path(__file__).resolve().parent
DATA = (
    HERE.parent
    / "data"
    / "synthetic-candidate-tweets"
    / "synthetic_candidate_tweets_2022.csv.gz"
)
TRUE_COLS = ["true_topic", "true_framing", "true_ideology"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    # Guard the committed seal record: rebuilding the parquet views is fine,
    # but silently rewriting seal_manifest.json would let a modified corpus
    # pass verification against its own fresh manifest. Rebuilds must either
    # keep the manifest byte-identical or be explicitly flagged as a reseal.
    manifest_path = HERE / "seal_manifest.json"
    reseal = "--reseal" in sys.argv
    prior = json.load(open(manifest_path)) if manifest_path.exists() else None
    if prior is not None and not reseal:
        if sha256(DATA) != prior.get("source_sha256"):
            sys.exit(
                "ERROR: source corpus hash does not match the committed "
                "seal_manifest.json. Refusing to overwrite the seal record. "
                "If the corpus was accidentally regenerated, restore it with "
                "`git checkout -- :/data/synthetic-candidate-tweets/` "
                "(the :/ prefix works from any directory). If a "
                "reseal is genuinely intended, rerun with --reseal (and say "
                "so loudly in the commit message)."
            )

    df = pd.read_csv(DATA, compression="gzip")
    n_cand = df["candidate_id"].nunique()
    print(f"Loaded {len(df):,} rows, {n_cand} candidates")
    assert all(c in df.columns for c in TRUE_COLS), "true_* columns missing"

    # Stable row order: chronological as shipped (firehose order preserved).
    sealed = df[["tweet_id", "candidate_id"] + TRUE_COLS].copy()
    blind = df.drop(columns=TRUE_COLS)

    blind_path = HERE / "blind_corpus.parquet"
    sealed_path = HERE / "sealed_truth.parquet"
    blind.to_parquet(blind_path, index=False, compression="zstd")
    sealed.to_parquet(sealed_path, index=False, compression="zstd")

    blind_ok = pins.frame_sha256(blind) == pins.BLIND_CONTENT_SHA256
    sealed_ok = pins.frame_sha256(sealed) == pins.SEALED_CONTENT_SHA256
    print(f"blind_corpus content matches 2026-07-25 pin : {'yes' if blind_ok else 'NO'}")
    print(f"sealed_truth content matches 2026-07-25 pin : {'yes' if sealed_ok else 'NO'}")

    if prior is not None and not reseal:
        if not (blind_ok and sealed_ok):
            sys.exit("ERROR: rebuilt parquet content differs from the pinned seal "
                     "although the source corpus hash matched — investigate before "
                     "trusting any downstream number. Manifest left untouched.")
        if (sha256(blind_path) != prior["blind_corpus"]["sha256"]
                or sha256(sealed_path) != prior["sealed_truth"]["sha256"]):
            print("note: parquet byte hashes differ from seal_manifest.json "
                  "(serialization drift only — content pins verified).")
        print("seal_manifest.json left unchanged (plain rebuild).")
        return

    # --reseal (or first-ever seal): write the manifest, re-stamping the
    # date, but carry over any keys this script does not own (ws4_* etc.).
    manifest = dict(prior) if prior is not None else {}
    manifest.update({
        "sealed_on": date.today().isoformat(),
        "source_file": DATA.name,
        "source_sha256": sha256(DATA),
        "blind_corpus": {
            "file": blind_path.name,
            "sha256": sha256(blind_path),
            "rows": int(len(blind)),
            "candidates": int(n_cand),
            "columns": list(blind.columns),
        },
        "sealed_truth": {
            "file": sealed_path.name,
            "sha256": sha256(sealed_path),
            "rows": int(len(sealed)),
            "columns": list(sealed.columns),
        },
        "protocol": (
            "Workstreams read blind_corpus.parquet only. sealed_truth.parquet "
            "is opened once per workstream at the validation step declared in "
            "its preregistration.md, written BEFORE unsealing."
        ),
    })
    with open(HERE / "seal_manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"blind_corpus.parquet : {len(blind):,} rows, "
          f"{len(blind.columns)} cols, sha256 {manifest['blind_corpus']['sha256'][:12]}…")
    print(f"sealed_truth.parquet : {len(sealed):,} rows, "
          f"sha256 {manifest['sealed_truth']['sha256'][:12]}…")
    print("Seal manifest written.")


if __name__ == "__main__":
    main()
