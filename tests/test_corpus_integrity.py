"""Integrity tests for the shipped, hash-pinned corpus.

The blind protocol depends on the sealed artifact being exactly what the
manifest says it is. These tests re-verify the SHA-256 pin and the
structural invariants every workstream assumes.
"""

import csv
import gzip
import hashlib
import json

import pytest


@pytest.fixture(scope="module")
def manifest(manifest_path):
    return json.loads(manifest_path.read_text())


@pytest.fixture(scope="module")
def rows(corpus_path):
    with gzip.open(corpus_path, "rt", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_source_hash_matches_seal_manifest(corpus_path, manifest):
    h = hashlib.sha256()
    with open(corpus_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    assert h.hexdigest() == manifest["source_sha256"]


def test_row_and_candidate_counts_match_manifest(rows, manifest):
    assert len(rows) == manifest["blind_corpus"]["rows"]
    candidates = {r["candidate_id"] for r in rows}
    assert len(candidates) == manifest["blind_corpus"]["candidates"]


def test_schema_is_sixteen_documented_columns(rows):
    expected = [
        "tweet_id", "candidate_id", "candidate_name", "handle", "party",
        "chamber", "state", "district", "incumbent", "timestamp_utc",
        "is_retweet", "retweeted_handle", "text", "true_topic",
        "true_framing", "true_ideology",
    ]
    assert list(rows[0].keys()) == expected


def test_tweet_ids_unique_and_chronological(rows):
    ids = [r["tweet_id"] for r in rows]
    assert len(ids) == len(set(ids))
    timestamps = [r["timestamp_utc"] for r in rows]
    assert timestamps == sorted(timestamps)


def test_ideology_bounded_and_constant_per_candidate(rows):
    seen = {}
    for r in rows:
        v = float(r["true_ideology"])
        assert -1.0 <= v <= 1.0
        prev = seen.setdefault(r["candidate_id"], v)
        assert prev == v, f"{r['candidate_id']} has multiple ideologies"


def test_retweet_rows_follow_documented_conventions(rows):
    """Retweets carry true_topic='retweet_source', an empty true_framing
    (a documented generator limitation), and a nonempty @source; originals
    carry no source."""
    n_rt = 0
    for r in rows:
        if r["is_retweet"] == "True":
            n_rt += 1
            assert r["true_topic"] == "retweet_source"
            assert r["true_framing"] == ""
            assert r["retweeted_handle"].startswith("@")
            assert r["text"].startswith("RT @")
        else:
            assert r["retweeted_handle"] == ""
            assert r["true_topic"] != "retweet_source"
    # README documents a 26.5% retweet share.
    assert n_rt / len(rows) == pytest.approx(0.265, abs=0.01)


def test_party_composition_matches_readme(rows):
    parties = {}
    for r in rows:
        parties[r["candidate_id"]] = r["party"]
    counts = {p: list(parties.values()).count(p) for p in ("D", "R", "I")}
    assert counts == {"D": 433, "R": 447, "I": 30}


def test_manifest_keeps_downstream_entries(manifest):
    # 01_seal_corpus.py used to rebuild the manifest from scratch and drop
    # these; a plain rebuild must now leave the manifest untouched.
    assert "ws4_label_release" in manifest
    assert "ws4_unseal" in manifest
