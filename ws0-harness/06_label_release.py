"""
06_label_release.py — WS4 step 1.2
-----------------------------------
The ONE pre-unseal read of sealed_truth.parquet permitted to WS4. Writes
candidate-level true_ideology for DEV candidates only (ws4_dev_labels.parquet)
and records its hash in seal_manifest.json. TEST labels stay sealed until
analyses/ws4-supervised/scripts/07_unseal_evaluate.py.

Requires: ws4_splits.json (05_ws4_splits.py) and a dated
analyses/ws4-supervised/preregistration.md — checked before anything is read.
"""

import hashlib
import json
from datetime import date
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PREREG = HERE.parent / "analyses" / "ws4-supervised" / "preregistration.md"
OUT = HERE / "ws4_dev_labels.parquet"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT.name} already exists — label release is one-time.")
    if not PREREG.exists() or "Date locked:** 2026-" not in PREREG.read_text():
        raise SystemExit("preregistration.md missing or undated — write it first.")

    splits = json.loads((HERE / "ws4_splits.json").read_text())
    dev = set(splits["dev"])

    truth = pd.read_parquet(HERE / "sealed_truth.parquet", columns=["candidate_id", "true_ideology"])
    cand = truth.groupby("candidate_id")["true_ideology"].agg(["first", "nunique"]).reset_index()
    assert (cand["nunique"] == 1).all(), "true_ideology is not constant within candidate"
    labels = (
        cand[cand.candidate_id.isin(dev)][["candidate_id", "first"]]
        .rename(columns={"first": "true_ideology"})
        .sort_values("candidate_id")
        .reset_index(drop=True)
    )
    assert len(labels) == len(dev), (len(labels), len(dev))
    labels.to_parquet(OUT, index=False)

    manifest_path = HERE / "seal_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["ws4_label_release"] = {
        "released_on": date.today().isoformat(),
        "file": OUT.name,
        "sha256": sha256(OUT),
        "n_candidates": int(len(labels)),
        "scope": "DEV candidates only (ws4_splits.json); TEST labels remain sealed until "
                 "analyses/ws4-supervised/scripts/07_unseal_evaluate.py",
        "preregistration": str(PREREG.relative_to(HERE.parent)),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Released {len(labels)} DEV labels -> {OUT.name}; manifest updated.")


if __name__ == "__main__":
    main()
