"""
01_features.py — WS4 feature spaces (blind; no labels read)
------------------------------------------------------------
Fits every unsupervised representation once on the full blind corpus and
writes TWEET-LEVEL caches to intermediate/ so scripts 03–04 can featurize
arbitrary tweet subsets, plus the full-corpus candidate matrices to
outputs/features_full.npz.

Spaces (preregistration §2): tfidf, w2v, doc2vec, m2v, behav.
Also written: outputs/unsupervised_refs.csv — frozen TF-IDF PC1 (from
ws0 baselines/axis_scores.csv) and the behavioral split-A score, per
candidate, for the apples-to-apples TEST comparison in 07.

Seeds: 20260720 for the w2v / TF-IDF anchors (WS0 recipe), 20260908 otherwise.
"""

import json
import time
from datetime import date

import numpy as np
import pandas as pd
from gensim.models import Doc2Vec, Word2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

import ws4lib as L

ANCHOR_SEED = 20260720


def main() -> None:
    t0 = time.time()
    blind = L.load_blind()
    cands = L.cand_order()
    cid = blind.candidate_id.to_numpy()
    print(f"{len(blind):,} tweets, {len(cands)} candidates")

    tokens = blind.text.astype(str).map(L.tokenize).tolist()
    rows_all = {c: np.flatnonzero(cid == c) for c in cands}

    # ---- word2vec (WS0 canonical recipe) ----
    print("word2vec…")
    w2v = Word2Vec(sentences=tokens, vector_size=L.DIM, window=5, min_count=5, sg=1,
                   workers=1, epochs=10, seed=ANCHOR_SEED, hashfxn=L.det_hash)
    tw = np.zeros((len(blind), L.DIM), dtype=np.float32)
    tw_n = np.zeros(len(blind), dtype=np.int32)
    for i, toks in enumerate(tokens):
        v = [w2v.wv[t] for t in toks if t in w2v.wv]
        if v:
            tw[i] = np.mean(v, axis=0)
            tw_n[i] = len(v)
    np.save(L.INTER / "tweet_w2v.npy", tw)
    np.save(L.INTER / "tweet_w2v_ntok.npy", tw_n)
    # frequency-weighted candidate average (WS0 construction) for the full matrix
    X_w2v = np.vstack([
        (tw[rows_all[c]] * tw_n[rows_all[c]][:, None]).sum(0) / max(tw_n[rows_all[c]].sum(), 1)
        for c in cands])

    # ---- TF-IDF + SVD: fit on candidate documents (WS0), transform tweets ----
    print("tfidf…")
    doc_strings = [" ".join(t for r in rows_all[c] for t in tokens[r]) for c in cands]
    tfidf = TfidfVectorizer(min_df=5, sublinear_tf=True)
    Xdoc = tfidf.fit_transform(doc_strings)
    svd = TruncatedSVD(n_components=L.DIM, random_state=ANCHOR_SEED)
    X_tfidf_doc = svd.fit_transform(Xdoc)                      # WS0 candidate-document construction
    tweet_strings = [" ".join(t) for t in tokens]
    T = svd.transform(tfidf.transform(tweet_strings)).astype(np.float32)
    np.save(L.INTER / "tweet_tfidf.npy", T)
    X_tfidf = np.vstack([T[rows_all[c]].mean(0) for c in cands])  # subset-capable construction

    # ---- doc2vec PV-DBOW tagged by candidate ----
    print("doc2vec…")
    tagged = [TaggedDocument(toks, [c]) for toks, c in zip(tokens, cid)]
    d2v = Doc2Vec(tagged, dm=0, vector_size=L.DIM, window=5, min_count=5, epochs=20,
                  workers=1, seed=L.SEED, hashfxn=L.det_hash)
    d2v.save(str(L.INTER / "doc2vec.model"))
    X_d2v = np.vstack([d2v.dv[c] for c in cands]).astype(np.float32)
    # inference-noise diagnostic on one probe candidate
    probe = cands[0]
    ptoks = [t for r in rows_all[probe] for t in tokens[r]]
    infs = []
    for s in range(10):
        d2v.random.seed(s)
        infs.append(d2v.infer_vector(ptoks, epochs=50))
    infs = np.vstack(infs)
    inf_cos_sd = float(np.std([np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)
                               for i, a in enumerate(infs) for b in infs[i + 1:]]))
    inf_vs_learned = float(np.corrcoef(infs.mean(0), X_d2v[0])[0, 1])

    # ---- Model2Vec (WS1 Tier A) ----
    print("model2vec…")
    from model2vec import StaticModel
    m2v = StaticModel.from_pretrained("minishlab/potion-base-8M")
    E = m2v.encode(blind.text.astype(str).tolist(), show_progress_bar=False).astype(np.float32)
    np.save(L.INTER / "tweet_m2v.npy", E)
    X_m2v = np.vstack([E[rows_all[c]].mean(0) for c in cands])
    # diagnostic variant: strip "RT @Handle:" prefixes and @mentions before embedding.
    # The WS3 corpus quirk is that ALL identifier cues live in RT prefixes; raw-text
    # m2v therefore sees retweet-source identity, which the tokenized spaces do not.
    stripped = blind.text.astype(str).str.replace(r"^RT @\w+:\s*", "", regex=True) \
                                     .str.replace(r"@\w+", " ", regex=True).tolist()
    Es = m2v.encode(stripped, show_progress_bar=False).astype(np.float32)
    np.save(L.INTER / "tweet_m2v_strip.npy", Es)
    X_m2v_strip = np.vstack([Es[rows_all[c]].mean(0) for c in cands])

    # ---- behavioral (split A only) ----
    print("behavioral…")
    F = L.Featurizer(blind)
    X_behav = F.features("behav", rows_all, behav_split_a_only=True)

    np.savez_compressed(L.OUT / "features_full.npz", candidate_ids=np.array(cands),
                        tfidf=X_tfidf, tfidf_doc=X_tfidf_doc, w2v=X_w2v, doc2vec=X_d2v,
                        m2v=X_m2v, m2v_strip=X_m2v_strip, behav=X_behav)

    # ---- unsupervised references (blind) ----
    axis = pd.read_csv(L.WS0 / "baselines" / "axis_scores.csv").set_index("candidate_id").loc[cands]
    refs = pd.DataFrame({"candidate_id": cands,
                         "tfidf_pc1_frozen": axis.tfidf_partisan_score.to_numpy(),
                         "behav_splitA_source_mean": X_behav[:, 0],
                         "behav_has_retweet_A": X_behav[:, 1]})
    refs.to_csv(L.OUT / "unsupervised_refs.csv", index=False)

    manifest = {
        "built_on": date.today().isoformat(), "seed": L.SEED, "anchor_seed": ANCHOR_SEED,
        "n_tweets": int(len(blind)), "n_candidates": len(cands),
        "w2v_vocab": len(w2v.wv), "tfidf_vocab": len(tfidf.vocabulary_),
        "tfidf_svd_explained": float(svd.explained_variance_ratio_.sum()),
        "doc2vec": {"dm": 0, "dim": L.DIM, "epochs": 20, "infer_epochs": 50,
                    "probe_candidate": probe, "infer_pairwise_cos_sd": inf_cos_sd,
                    "infer_mean_vs_learned_r": inf_vs_learned},
        "m2v_model": "minishlab/potion-base-8M", "m2v_dim": int(E.shape[1]),
        "behav_dims": ["src_mean", "has_rt", "rt_share", "log1p_n"] + [f"share:{o}" for o in F.orgs],
        "labels_read": False,
        "runtime_s": round(time.time() - t0, 1),
    }
    (L.OUT / "features_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
