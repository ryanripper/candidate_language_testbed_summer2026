# candidate_language_testbed_summer2026

[![CI](https://github.com/ryanripper/candidate-language-testbed/actions/workflows/ci.yml/badge.svg)](https://github.com/ryanripper/candidate-language-testbed/actions/workflows/ci.yml)

**How well do text-based instruments recover a politician's ideological position — and how would you know?**

Ryan Ripper — 2026

On real congressional tweets you cannot answer that question cleanly: there is no ground truth, only proxies (DW-NOMINATE, donor scores, expert surveys) that are themselves contested. This project builds a **synthetic corpus with planted ground truth** — 910 fictional 2022-cycle candidates, 104,601 tweets, each candidate generated from a known latent ideology and a known topic mix — and then runs six families of instruments against it under a blind protocol, so that "this method recovers ideology at r = .97" is a measurement rather than a claim.

The design question is not *which method wins*. It is **which methods fail, how, and whether you could have detected the failure without the answer key** — because on real data you never have the answer key.

## Data

The corpus is generated, not collected. Everything — names, handles, organizations — is fictional, and the generator is deterministic under `SEED = 20260719`.

| | |
| --- | --- |
| Candidates | 910 (447 R, 433 D, 30 I) |
| Chambers | 841 House, 69 Senate |
| Tweets | 104,601 (26.5% retweets) |
| Tokens | 1,439,390 across a raw vocabulary of 828 (harness tokenizer: lowercased, URL/mention-stripped, `[a-z][a-z']+`) |
| Median tweets per candidate | 99 |
| Topics | 10 policy topics, plus `campaign_logistics` (uncorrelated with ideology) and `retweet_source` |

`data/synthetic-candidate-tweets/generate_synthetic_candidates.py` plants the structure the instruments are then asked to recover:

- Each candidate gets a `true_ideology` score in [−1, 1] (liberal → conservative), drawn from party-conditional distributions.
- Each candidate gets a Dirichlet topic mix over 11 topics — the 10 policy topics plus `campaign_logistics`, which is uncorrelated with ideology and carries the largest alpha.
- Tweet text is assembled from topic-specific phrase banks, where the probability of drawing liberal- vs. conservative-coded framing is a logistic function of `true_ideology` — so lexical choice encodes the latent score.
- Retweets are drawn from a pool of fictional org/pundit accounts, each with its own ideology; candidates preferentially retweet nearby accounts, modeling retweets-as-endorsed-speech.

The flat corpus ships as `synthetic_candidate_tweets_2022.csv.gz` with 16 columns. `ws0-harness/01_seal_corpus.py` splits it into a **13-column blind working view** and a **sealed truth file** holding `true_ideology`, `true_topic`, and `true_framing`. Both views, and the source file, are SHA-256 stamped in `ws0-harness/seal_manifest.json` (sealed 2026-07-25).

## Method

Every workstream follows the same discipline, enforced by a shared harness:

1. **Seal the truth.** `ws0-harness/01_seal_corpus.py` produces the blind view and the hash-stamped sealed truth file.
2. **Pre-register.** Each workstream writes `preregistration.md` — hypotheses, metrics, decision rules, and the bar to clear — *dated, before unsealing*.
3. **Analyze blind.** Including a mandatory **confound gate**: diagnose the dominant principal components against observable covariates (retweet share, volume, topic entropy) before making any distance claim.
4. **Unseal once.** A single validation script per workstream. Results are reported against the frozen baseline table, not against re-derived numbers.

Anything discovered after the unseal is labelled **EXPLORATORY** and stays labelled that way downstream. Several headline numbers in this repo carry that label.

## Results

Axis recovery — Pearson r against planted ideology, n = 910 unless noted:

| Instrument | r vs truth | Notes |
|---|---|---|
| Behavioral (retweet-source ideology) | **.977** | n = 146 pilot support (.980 at n = 905); generator ceiling ≈ .98 |
| TF-IDF + SVD | **.974** | the frozen baseline, and the bar to beat |
| LLM ask-and-average | **.970** | n = 150 pilot; **zero corpus training** |
| Model2Vec (static distilled) | .900 | best of the embedding family |
| word2vec | .878 | n = 150 pilot support (.885 at n = 910) |
| MiniLM sentence transformer | .721 | style-contaminated axis |

Distance validity (do pairwise text distances reproduce ideological distances?): Model2Vec corrected **.640** *(exploratory)* ≈ TF-IDF **.624** > word2vec .592 ≫ MiniLM .397. Within the retweet-content topic slice alone: **.849**.

Topic recovery: in the blind bake-off, **nobody cleared the .60 ARI bar** — the LLM entrant won at .289. Three post-unseal refinements then took it to ARI **.890**: the retweet-routing convention (an observable-column rule that was *available* blind — the bake-off simply didn't think of it) lifted it to .765, dissolving two genre themes the unseal exposed as boundary errors to .824, and coarsening the taxonomy to K = 13 to .890. Only the first step could have been applied without the answer key.

Held-out behavior prediction (retweet-source choice, estimated on split A, evaluated on split B): oracle 2.317 < TF-IDF 2.348 < LLM 2.368 < Model2Vec 2.430, against nulls at ≈ 2.98–3.00.

### The four findings that transfer

**1. Direction is much easier than distance.** Four instrument families place candidates on a left–right axis at r ≥ .90, yet no high-dimensional text geometry gets above .66 agreement with the true pairwise-distance structure. Recovering *who is to the left of whom* and recovering *how far apart they are* are different problems with a large difficulty gap between them.

**2. The retweet-style confound is universal.** Every embedding space tested — word2vec, GloVe, fastText, doc2vec, MiniLM, Model2Vec, TF-IDF — devotes a dominant principal component to **retweet share**, an observable covariate, at |r| = .84–.97 (the five WS4 spaces at .897–.96; Model2Vec −.97, MiniLM −.84). It is PC1 in the word-vector models, MiniLM, and Model2Vec, and PC2 in doc2vec and TF-IDF. This is detectable blind, which is the point: the confound gate catches it before any substantive claim is made. Correcting for it is what lifts word2vec distance validity from .278 to .592.

**3. Instrument errors cluster into two families.** Partialling out the oracle and correlating the residuals reveals a **content family** (LLM–behavioral .58, LLM–TF-IDF .55) and a **style family** (word2vec–MiniLM .61, MiniLM–Model2Vec .47), with cross-family residuals at .15–.38 for eight of the nine pairs (the exception is TF-IDF–word2vec at .45, the leakiest boundary between the families). Practical consequence: an ensemble gains almost nothing from a second instrument *within* a family. Buy diversity *across* families.

**4. The "far from whom, on what" decomposition can be estimated truth-free.** Replacing the oracle with LLM pilot scores reproduces the WS2 signal-tier ladder almost exactly (tier-order Spearman ρ = **.97**) — LLM taxonomy topics + LLM positional scores + per-topic centroid distances, with no ground truth anywhere in the pipeline. That is precisely the configuration available on real data.

## Requirements

Python 3.10+, plus `requirements.txt`:

```
numpy>=1.26                   pandas>=2.0             scipy>=1.11
scikit-learn>=1.3             matplotlib>=3.8         pyarrow>=14.0.1
gensim>=4.3.3                 sentence-transformers>=2.2
model2vec>=0.3                umap-learn>=0.5         hdbscan>=0.8
```

The original runs did not record a lockfile, so these are floors at the versions the pipeline was developed against rather than exact pins. **Results are seed-pinned, not version-pinned** — minor numerical drift across library versions is possible.

## Setup

```bash
git clone https://github.com/ryanripper/candidate-language-testbed.git
cd candidate-language-testbed
pip install -r requirements.txt
```

## Usage

Build the harness first — the analyses read its outputs, and several of its artifacts are gitignored because they are regenerable:

```bash
cd ws0-harness
python 01_seal_corpus.py        # rebuilds the blind/sealed corpus views
python metrics.py               # self-tests
python 02_freeze_baselines.py   # ~10 min, word2vec retrain, single-threaded
python 03_build_splits.py
python 04_verify_harness.py     # must print HARNESS VERIFIED
```

Then run any analysis folder's `scripts/` in numeric order. Every script is seed-pinned, but three steps are not bit-reproducible despite the seed: the pilot's word2vec (`00-embeddings-pca/scripts/02_embeddings.py`, `workers=4`, no pinned `hashfxn` — the harness's `02_freeze_baselines.py` retrains with the canonical recipe of seed 20260720, `workers=1`, deterministic `hashfxn`, and its arrays are the frozen baselines going forward), WS2's `LdaMulticore(workers=3)`, and WS4's multithreaded GloVe/fastText/doc2vec. The frozen numbers are verified within declared tolerances by `04_verify_harness.py`, not re-derived exactly.

The LLM-dependent steps (WS2 topic labelling, WS3 scoring) were performed by in-session LLM agents at pilot scale — there is no committed API-calling code, so they are the only steps that cannot be re-run from this repo. Their outputs are committed so the downstream analyses reproduce without re-scoring; `analyses/ws3-llm-scaling/outputs/raw_scores.tar.gz` is the audit archive of the raw scores, and `analyses/ws3-llm-scaling/prompts/scoring_prompt_v1.md` is the frozen scoring prompt.

## Testing and continuous integration

The maintained surface — `ws0-harness/metrics.py`, the corpus generator, and the sealed artifact itself — is covered by a pytest suite in `tests/`:

- **Metrics** (`test_metrics.py`): every scoring function the workstreams share, including the degenerate-PC guard, Mantel determinism under seed, and Procrustes invariance to rotation/scale.
- **Generator** (`test_generator.py`): determinism under `SEED = 20260719` (the roster must reproduce the sealed corpus's 910 candidates and 447 R / 433 D / 30 I split), ideology bounds and party ordering, topic-mix validity, the framing-encodes-ideology logistic link, and retweet-source proximity. The suite never regenerates or overwrites the hash-pinned corpus.
- **Corpus integrity** (`test_corpus_integrity.py`): re-verifies the SHA-256 pin in `seal_manifest.json` and the structural invariants downstream analyses assume (schema, chronological IDs, one ideology per candidate, retweet conventions).

```bash
pip install -r requirements-ci.txt
pytest        # 42 tests, a few seconds
ruff check .  # lint (analyses/ is excluded: those scripts are archived as run)
```

CI (GitHub Actions) runs both on Python 3.10 and 3.12 for every push and pull request, using the minimal `requirements-ci.txt` — the heavy embedding dependencies are not needed to test the harness.

## Project structure

```
├── data/synthetic-candidate-tweets/   generator + the 104,601-tweet corpus
├── ws0-harness/                       sealed corpus, frozen baselines, shared
│                                      metrics, frozen evaluation splits
├── analyses/
│   ├── 00-embeddings-pca/             pilot: word2vec + PCA + distances (07-20)
│   ├── ws1-sentence-transformers/     MiniLM / Model2Vec — pre-registered
│   │                                  negative result
│   ├── ws2-topic-bakeoff/             LDA / NMF / LSA / BERTopic-style / LLM
│   ├── ws3-llm-scaling/               LLM ask-and-average ideological scaling
│   ├── ws4-preanalysis/               static-embedding bake-off (w2v, GloVe,
│   │                                  fastText, doc2vec) — informal, truth-visible
│   ├── ws4-supervised/                supervised ideology prediction — preregistered,
│   │                                  DEV/TEST label release; in progress
│   └── synthesis/                     agreement matrix, Mantel/Procrustes,
│                                      divergence cases, consolidated table
├── tests/                             pytest suite: metrics, generator,
│                                      sealed-corpus integrity
├── .github/workflows/ci.yml           CI: ruff + pytest (py 3.10 / 3.12)
└── docs/
    ├── writeups.md                    index of every stage write-up + article draft
    ├── plans/                         research plan and execution plan
    ├── skills/                        2026 methods survey the project draws on
    ├── reference/                     supervised/unsupervised workflow guide
    └── NOTES.md, NOTES-readout.md     working notes and their read-out
```

Each analysis folder carries its own `scripts/` (numbered pipeline), `outputs/`, and `figures/`, plus a stage write-up. The supporting documents vary by stage:

| Folder | `README` | `preregistration` | `REGENERATE` | Write-up |
| --- | :-: | :-: | :-: | --- |
| `00-embeddings-pca` | | | | `00_workflow_outline.md` |
| `ws1-sentence-transformers` | ✓ | ✓ | ✓ | `ws1-writeup.md` |
| `ws2-topic-bakeoff` | ✓ | ✓ | ✓ | `ws2-writeup.md` |
| `ws3-llm-scaling` | ✓ | ✓ | ✓ | `ws3-writeup.md` |
| `ws4-preanalysis` | ✓ | | ✓ | `ws4-preanalysis-writeup.md` |
| `ws4-supervised` | ✓ | ✓ | ✓ | *(pending unseal)* |
| `synthesis` | ✓ | | | `synthesis-writeup.md` |

The gaps are deliberate rather than oversights: `00-embeddings-pca` is the pre-harness pilot and predates the protocol, while `ws4-preanalysis` and `synthesis` are respectively truth-visible and post-hoc, so neither had a blind result to pre-register.

### Not included in this repository

Roughly 106 MB of derived binary artifacts are excluded to keep the repo cloneable: WS1's 16 distance matrices and candidate representations, WS2's two per-topic distance tensors, the WS0 sealed-corpus parquets and baseline distance matrices, and WS4's feature matrices. `ws0-harness/`, `ws1-sentence-transformers/`, `ws2-topic-bakeoff/`, and `ws4-preanalysis/` each carry a `REGENERATE.md` naming the exact script that rebuilds each file; `.gitignore` also lists every excluded path against the script that produces it. All CSV/JSON results, all figures, all scripts, all manifests, and the source corpus are committed.

Also omitted: a technical writing sample PDF and an annotated companion, both job-application artifacts rather than research outputs.

### A note on paths and historical documents

This research was produced in a sandboxed working directory, not a git repo. Converting it required a folder reorganization and a set of path fixes — scripts had hardcoded sandbox absolute paths and located each other by their old directory names. Every one of those edits is itemized in [docs/repo-restructure-notes.md](docs/repo-restructure-notes.md), along with a folder mapping table. No seed, hyperparameter, metric, or decision rule changed.

The **pre-registrations and write-ups were deliberately not edited.** (A 2026-08 post-hoc audit appended clearly-dated errata sections to two write-ups; the original dated text above each separator remains untouched.) Several are dated documents written *before* the corresponding unseal, and their evidential value depends on not being rewritten afterward. They therefore refer to folders by their original names — use the mapping table when following a path mentioned in one of them.

## Standing caveats

Read these before quoting any number above.

- **Template-generated text flatters every recovery number.** Planted sentences recur verbatim across candidates. These are *transfer* results — a method that fails here would fail on real tweets; a method that succeeds here has only earned a real-data trial.
- **The frozen baselines saw split-B text**, so leakage mildly favors the baselines.
- **WS3 annotator agents and the orchestrator share a model family** (blinded, and reported as such).
- **Score-derived distance matrices are rank-1 by construction**, so the .93–.95 agreement among score-based geometries is partly bookkeeping.
- **LLM coverage is the n = 150 stratified pilot only.** The full 910 × 5 scale-up cleared its pre-registered bar (r = .970 ≥ .90) and was recommended, but was deferred on 2026-07-27 and is not funded.
- **WS4 preanalysis is truth-visible and informal** — directional only, and must not be quoted alongside the frozen blind numbers without this caveat.
- **The "raw → corrected" distance-validity lift conflates two effects.** The raw branch uses uncentered vectors while the corrected branch centers *and* projects out the style PC, so part of the .28 → .60 improvement is mean-centering, not style removal. The between-instrument comparisons are unaffected (every instrument uses the same convention), but the lift should not be read as a pure measure of the confound's cost.

## Status

WS0–WS3 and the synthesis stage are complete. WS4 (supervised prediction of planted ideology) is in progress: the informal preanalysis found that **every feature space saturates the generator ceiling under light ridge supervision** (out-of-fold r = .966–.973 against a ceiling of ≈ .973), which moved the WS4 questions away from raw accuracy toward label efficiency, text scarcity, incumbent→challenger transfer, and cross-family ensembles. The certified workstream (`analyses/ws4-supervised/`, plan in `docs/plans/ws4-supervised-plan.md`) is preregistered with a DEV/TEST candidate split and DEV-only label release; feature spaces and the E4.1 development probes are done, and **TEST labels remain sealed** pending the remaining experiments.

## Tech stack

NumPy, pandas, SciPy, scikit-learn (PCA, TF-IDF/SVD, ridge, clustering metrics), gensim (word2vec, doc2vec), sentence-transformers (MiniLM), Model2Vec (static distilled embeddings), UMAP + HDBSCAN (BERTopic-style topic entrant), matplotlib (figures), PyArrow (parquet I/O for the sealed corpus), and in-session LLM agents for the WS2 labelling and WS3 scoring steps (pilot scale; no API-calling code is committed — see Usage).

## A note on AI tooling

This project was built with Claude (Anthropic) as a research and coding assistant, working in a sandboxed session alongside the author. The division of labor, so the reader can weigh it:

- **Author:** research questions, the extension roadmap, every decision point in the plan documents (D1–D9 — each recorded with its date and the option chosen), scope calls (testbed-first, separate per-workstream articles, deferring the LLM scale-up), review of all results and write-ups, and git history.
- **Claude:** drafting the plan and preregistration documents from the author's decisions, writing and running the numbered pipeline scripts, the shared harness, tests and CI scaffolding, figures, and first drafts of the stage write-ups; the repo restructure (path fixes itemized in `docs/repo-restructure-notes.md`).
- **In-session LLM agents** (also Claude) served as the *instruments under test* in WS2 (topic labelling, LLM-as-judge) and WS3 (ask-and-average scoring). They were blinded — fresh agents, identifiers stripped, no truth access — but they share a model family with the orchestrating assistant, a caveat repeated wherever those results are cited.

Two consequences worth stating. First, the blind protocol was designed partly *because* an AI assistant was in the loop: the seal, the dated preregistrations, the one-read unseal scripts, and the DEV-only label release in WS4 exist so that no participant — human or model — could let the answer key steer design choices, and the hashes in `ws0-harness/seal_manifest.json` let a reader verify that. Second, the assistant's contributions are reproducible artifacts, not judgments: every number in this README comes from a committed, seed-pinned script that anyone can re-run without any AI tooling at all.

## Author

Ryan Ripper — 2026
