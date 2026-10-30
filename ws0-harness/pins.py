"""
pins.py — content commitments for the frozen WS0 splits (added 2026-10-30)
--------------------------------------------------------------------------
Why this exists: a 2026-10-30 fresh-clone audit found that re-running
03_build_splits.py reproduces the tweet-level A/B split exactly but NOT the
150-candidate subsample (122/150 ids matched). One candidate sits on a
within-party tercile boundary of axis_scores.csv and now falls the other
way, which reshuffles the RNG draws in two strata. The committed
splits.json is canonical — every WS2/WS3/synthesis result used exactly those
150 ids — so 03 must never silently overwrite it, and 04 must fail if it
changes. These hashes are deliberately constants, not manifest lookups,
for the same reason SOURCE_SHA256 is in 04_verify_harness.py.

Hashes are over CONTENT (sorted ids / sorted tweet_id-split pairs), not over
parquet or json bytes, so they survive pandas/pyarrow serialization drift.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import TYPE_CHECKING

# pandas is imported lazily inside the frame helpers so that ids_sha256 (and
# the CI test that uses it) work with the slim requirements-ci.txt, which
# deliberately omits pandas.
if TYPE_CHECKING:
    import pandas as pd

# sha256 of ",".join(sorted(subsample_150.candidate_ids)) — the 2026-07-25 draw.
SUBSAMPLE_IDS_SHA256 = "ae149fe94476e6502c5c955d3c6c3c302c642909afca368c2bc4e09bb8d5b105"

# sha256 of "\n".join(f"{tweet_id}\t{split}") sorted by tweet_id — the
# 2026-07-25 A/B assignment (reproduces exactly from seed 20260725).
AB_CONTENT_SHA256 = "fcd67ab82d87c7939c97a35f83395029635492b9ab1c82c3cab8b4822602421a"


# Content hashes of the 2026-07-25 sealed parquet views (see frame_sha256).
# Both reproduce exactly from 01_seal_corpus.py; only the parquet BYTES drift
# with pandas/pyarrow versions.
BLIND_CONTENT_SHA256 = "129a7062cdf6198d1d7bd4ee9c0e17b7ebdc84a5a9d25f586f61a3424ca2e6e8"
SEALED_CONTENT_SHA256 = "7b2de0e9d1181101efb1a731bceecfb9fee55e8aa263e2ee7efc5c1eca54c8db"


def frame_sha256(df: pd.DataFrame) -> str:
    """Order-invariant content hash of a tweet-level frame (rows sorted by
    tweet_id, index ignored; per-row hashes from pandas, so dtype-exact)."""
    import pandas as pd

    df = df.sort_values("tweet_id").reset_index(drop=True)
    row_hashes = pd.util.hash_pandas_object(df, index=False).to_numpy()
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()


def ids_sha256(ids: Iterable[str]) -> str:
    return hashlib.sha256(",".join(sorted(map(str, ids))).encode()).hexdigest()


def ab_content_sha256(ab: pd.DataFrame) -> str:
    ab = ab.sort_values("tweet_id")
    body = "\n".join(f"{t}\t{s}" for t, s in zip(ab["tweet_id"].astype(str),
                                                 ab["split"].astype(str)))
    return hashlib.sha256(body.encode()).hexdigest()
