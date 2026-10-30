"""The committed splits.json must be the frozen 2026-07-25 subsample draw.

Every n=150 result in the write-ups used exactly these candidate ids, and the
draw is known not to reproduce from 03_build_splits.py (see ws0-harness/pins.py
and REGENERATE.md). CI fails if the committed ids ever change.
"""

import json
from pathlib import Path

import pins

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "ws0-harness" / "splits.json"


def test_committed_subsample_matches_pin():
    ids = json.loads(SPLITS.read_text())["subsample_150"]["candidate_ids"]
    assert len(ids) == 150
    assert len(set(ids)) == 150
    assert pins.ids_sha256(ids) == pins.SUBSAMPLE_IDS_SHA256


def test_ids_sha256_is_order_invariant():
    assert pins.ids_sha256(["C0002", "C0001"]) == pins.ids_sha256(["C0001", "C0002"])
