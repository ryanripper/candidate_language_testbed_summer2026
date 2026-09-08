"""
05_ws4_splits.py — WS4 step 1.1
--------------------------------
One-time candidate-level DEV / TEST partition for the supervised workstream,
plus a frozen inner 5-fold assignment within DEV. Blind: reads only
blind_corpus.parquet and the frozen blind TF-IDF axis score.

Stratification: party x chamber x incumbent x within-party tercile of the
blind TF-IDF partisan score (baselines/axis_scores.csv). DEV:TEST = 2:1
(plan decision D5). Independents (n=30) are their own stratum cells.

Output: ws4_splits.json  (never re-run once WS4 has read labels — the file
records its own creation date and seed).
"""

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SEED = 20260908
TEST_FRAC = 1 / 3
N_INNER = 5
OUT = HERE / "ws4_splits.json"


def main() -> None:
    import sys
    if OUT.exists() and "--force" not in sys.argv:
        raise SystemExit(f"{OUT.name} already exists — splits are frozen; pass --force only to rebuild deliberately.")

    meta = pd.read_csv(HERE / "baselines" / "candidate_metadata.csv")
    axis = pd.read_csv(HERE / "baselines" / "axis_scores.csv")
    df = meta.merge(axis[["candidate_id", "tfidf_partisan_score"]], on="candidate_id")
    df = df.sort_values("candidate_id").reset_index(drop=True)

    # within-party tercile of blind TF-IDF score
    df["tercile"] = (
        df.groupby("party")["tfidf_partisan_score"]
        .transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=False))
    )
    df["stratum"] = (
        df["party"] + "|" + df["chamber"] + "|" + df["incumbent"].astype(str) + "|" + df["tercile"].astype(str)
    )

    rng = np.random.default_rng(SEED)
    df["split"] = "DEV"
    df["inner_fold"] = -1
    fold_counter = 0  # global round-robin so fold sizes stay balanced across strata
    for _, idx in df.groupby("stratum").groups.items():
        idx = np.array(sorted(idx))
        rng.shuffle(idx)
        n_test = int(round(len(idx) * TEST_FRAC))
        test_idx = idx[:n_test]
        dev_idx = idx[n_test:]
        df.loc[test_idx, "split"] = "TEST"
        # inner folds: global round-robin continuing across strata (balanced sizes)
        for i in dev_idx:
            df.loc[i, "inner_fold"] = fold_counter % N_INNER
            fold_counter += 1

    dev = df[df.split == "DEV"]
    test = df[df.split == "TEST"]
    summary = {
        "n_dev": int(len(dev)),
        "n_test": int(len(test)),
        "dev_party": dev.party.value_counts().to_dict(),
        "test_party": test.party.value_counts().to_dict(),
        "dev_incumbent": int(dev.incumbent.sum()),
        "test_incumbent": int(test.incumbent.sum()),
        "test_challenger": int((~test.incumbent).sum()),
        "dev_senate": int((dev.chamber == "Senate").sum()),
        "test_senate": int((test.chamber == "Senate").sum()),
        "inner_fold_sizes": dev.inner_fold.value_counts().sort_index().tolist(),
    }
    print(json.dumps(summary, indent=1))

    out = {
        "created": date.today().isoformat(),
        "seed": SEED,
        "method": (
            "candidate-level DEV/TEST 2:1, stratified by party x chamber x incumbent x "
            "within-party tercile of blind TF-IDF partisan score; inner 5-fold assigned "
            "round-robin within stratum after seeded shuffle"
        ),
        "summary": summary,
        "dev": sorted(dev.candidate_id.tolist()),
        "test": sorted(test.candidate_id.tolist()),
        "inner_fold": {r.candidate_id: int(r.inner_fold) for r in dev.itertuples()},
        "stratum": {r.candidate_id: r.stratum for r in df.itertuples()},
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(f"{OUT.name} written.")


if __name__ == "__main__":
    main()
