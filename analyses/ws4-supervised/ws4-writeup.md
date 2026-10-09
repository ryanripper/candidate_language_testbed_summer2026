# WS4 Session Write-Up — Supervised Prediction of Planted Ideology

*Sessions 2026-09-08 (prereg + harness + E4.1 on DEV), 2026-09-10 (Addendum B),
2026-09-15/16 (scripts 03–09, 07a, figure specs, Addendum C) and **2026-10-09
(unseal, figures, this write-up)**. Seeds 20260908 (everything preregistered
on 09-08) and 20260910 (Addendum B draws). This is the working narrative; the
polished piece is `article_draft.md`, the operational summary is `README.md`.*

## What was run

Workstream 4 per `../../docs/plans/ws4-supervised-plan.md`: supervised
prediction of `true_ideology` from six candidate-level feature spaces, with
the WS0 blind habit moved from the corpus boundary to the **evaluation**
boundary (fold-wise label release, decision D5–D9).

Order of operations, protocol intact end to end:

1. **Preregistration locked 2026-09-08** before any label was released:
   seven experiments after Addendum B (E4.1 certified probes, E4.2 label
   efficiency, E4.3 text scarcity, E4.4 incumbent→challenger and House→Senate
   shift, E4.5 cross-family stacking, E4.6 LLM-as-feature on the 150,
   E4.7 split-conformal coverage, E4.8 pool-based active learning), one bar
   each, expectations stated so they could be wrong.
2. **Harness**: candidate-level DEV 607 / TEST 303 split (2:1, stratified
   party × chamber × incumbent × blind-TF-IDF tercile), frozen inner 5-fold
   within DEV; `06_label_release.py` wrote DEV-only truth (read #1 of
   `sealed_truth.parquet`, hash in `seal_manifest.json`).
3. **Scripts 01–06, 08, 09 on DEV** (sessions 2): every model, draw
   schedule, meta-weight and q̂ fixed from DEV labels only; DEV development
   numbers recorded in `README.md` for honesty.
4. **Figure specs frozen** (`figure_specs.md`, 2026-09-15) and **all TEST
   predictions frozen label-free** (`07a_freeze_predictions.py`, 7,416 arrays,
   Addendum C) before the read.
5. **Single unseal 2026-10-09** (`07_unseal_evaluate.py --unseal`, read #2):
   hash of `sealed_truth.parquet` verified against the manifest, 303 TEST
   labels read once, frozen arrays scored, bars applied, `decision.json` and
   the `ws4_unseal` manifest entry written. The script fits nothing. A
   dry-run against DEV (never opening the sealed file) had been exercised
   on 09-16 and again on 10-09 in the new environment before the real run.
6. **Figures** (`10_figures.py`) implemented from the frozen specs, reading
   only 07's outputs plus the DEV context the specs allow.

## Results in one breath

**Six of seven preregistered bars met; E4.8 (active learning) not met.**
Supervision lands every text space at the generator ceiling
(ridge r = .970–.982 on TEST), the cross-family stack is the best single
number on the testbed (r = .986, +.005 [.003, .007] over the best single
probe), the incumbent→challenger and House→Senate shifts cost nothing
(|Δr| ≤ .004, CIs cover 0), the conformal intervals are calibrated
marginally (.91–.94 at α = .10) but **break under text scarcity** (≈ .70
coverage at k = 25), label efficiency is poor for TF-IDF (n* = 320) and
excellent for the behavioral features (n* = 40), and no active-learning
strategy reaches the ceiling within 60 labels on any content space.

The headline that will travel: **the zero-label TF-IDF PC1 (r = .976 on
TEST) is not beaten by any supervised single-space probe except raw
Model2Vec (.982, Δ = +.006 [.002, .010]) — and raw Model2Vec's edge is the
`RT @OrgHandle:` prefix** (m2v − m2v_strip = +.007 [.004, .010]; stripped
Model2Vec = .975, indistinguishable from PC1). On this generator the
supervised reader buys calibration, ensembling and transfer guarantees,
not accuracy.

## Per-experiment results (TEST, n = 303, single read)

### E4.1 — Certified probes (bar: ridge TF-IDF and doc2vec ≥ .965) — MET

| space | ridge r | GBM r | ridge, style PC projected | Δr vs behav ref [CI] | Δr vs TF-IDF PC1 [CI] |
|---|---|---|---|---|---|
| TF-IDF | **.974** | .975 | .967 | +.005 [−.003, +.013] | −.001 [−.006, +.003] |
| doc2vec | **.974** | .905 | .974 | +.005 [−.003, +.014] | −.001 [−.005, +.003] |
| word2vec | .970 | .938 | .950 | +.000 [−.008, +.009] | −.006 [−.012, −.001] |
| Model2Vec (raw) | **.982** | .949 | .978 | +.012 [+.006, +.020] | +.006 [+.002, +.010] |
| Model2Vec (RT-stripped) | .975 | .933 | .958 | +.006 [−.002, +.014] | −.001 [−.005, +.004] |
| behavioral (split A) | .967 | .972 | .967 | −.002 [−.005, +.002] | −.008 [−.017, −.000] |
| *TF-IDF PC1, frozen, zero labels* | *.976* | | | | |
| *behavioral split-A mean, zero labels* | *.969* | | | | |

Spearman ρ .935–.957, RMSE .12–.16 for ridge probes. Expectations: "all
content probes within .01 of ceiling" — yes; "no probe beats behavioral" —
wrong, every text probe matches or beats the split-A behavioral reference,
raw m2v significantly so; "style projection |Δr| < .005" — wrong for
TF-IDF (−.008), w2v (−.020) and m2v_strip (−.017): the projection *hurts*
(it removes signal, not confound), confirming preanalysis finding 5 that
the supervised reader needs no confound gate. GBM is competitive only on
TF-IDF and behavioral (D6: linear sufficiency on the dense spaces holds).
`tfidf_doc` (WS0 candidate-document construction) .974 ridge / .976 GBM —
the mean-of-tweet-rows construction loses nothing.

### E4.2 — Label efficiency (bar: ≥ 1 space with n* ≤ 40, ridge) — MET, via `behav` only

Median TEST r over 50 draws (ridge):

| n | behav | doc2vec | m2v | m2v_strip | TF-IDF | w2v |
|---|---|---|---|---|---|---|
| 10 | .938 | .801 | .892 | .857 | **.347** | .844 |
| 20 | .942 | .885 | .920 | .899 | .536 | .883 |
| 40 | .953 | .931 | .941 | .926 | .728 | .908 |
| 80 | .960 | .946 | .954 | .944 | .871 | .926 |
| 160 | .964 | .964 | .967 | .960 | .948 | .950 |
| 320 | .966 | .971 | .977 | .969 | .969 | .965 |
| 607 | .967 | .974 | .982 | .975 | .974 | .970 |

n\* (first n within .02 of the full-DEV TEST r): **behav 40**; doc2vec,
m2v, m2v_strip, w2v 160; **TF-IDF 320**. The preregistered expectation
(TF-IDF/doc2vec n\* ≤ 40) is wrong in exactly the way DEV foretold: a ridge
on 100 standardized SVD components cannot find at n = 10 the axis the
zero-label PC1 hands over for free (.976), and supervision on TF-IDF only
overtakes its own unsupervised instrument at n ≈ 320. The bar is met on
the letter (behav n\* = 40) but its "met →" reading — "incumbent labels
are ample on real data" — holds only for the behavioral features, whose
20-dimensional retweet-source share is nearly a direct readout of the
planted mechanism. For text, label acquisition stays a real-data risk.
GBM: degenerate at n ≤ 40 (all 50 draws, Addendum C §3); n\* 80 (behav,
TF-IDF) to > 320 (doc2vec). Post-unseal exploration candidate (D9, not
run): unstandardized / PC-truncated ridge at small n.

### E4.3 — Text scarcity (bar: best content space ≥ .90 at k = 25, mixed) — MET (.938)

Median TEST r over 20 draws, ridge trained and scored on k-tweet features:

| k | mixed: TF-IDF / doc2vec / m2v / m2v_strip / w2v / behav | originals: TF-IDF / doc2vec / m2v / m2v_strip / w2v / behav |
|---|---|---|
| 5 | .770 / .773 / .836 / .767 / .742 / .622 | .785 / .783 / .781 / .781 / .779 / −.00 |
| 10 | .871 / .872 / .905 / .868 / .850 / .763 | .874 / .870 / .870 / .870 / .861 / −.00 |
| 25 | **.938** / .934 / .953 / .937 / .925 / .893 | .938 / .939 / .939 / .939 / .934 / −.02 |
| 50 | .965 / .958 / .972 / .962 / .956 / .942 | .963 / .959 / .962 / .962 / .959 / −.01 |
| 100 | .973 / .968 / .980 / .972 / .966 / .963 | .969 / .965 / .969 / .969 / .965 / −.04 |
| all | .974 / .974 / .982 / .975 / .970 / .967 | .970 / .966 / .970 / .970 / .966 / −.04 |

Content degrades gracefully (k = 25 ≈ .94, k = 10 ≈ .87); behavioral
degrades fastest in the mixed regime (.893 at k = 25, .622 at k = 5) and is
dead by construction without retweets. In the originals regime raw and
stripped Model2Vec are identical to three decimals — the raw-m2v edge is
entirely the retweet prefix, as Addendum A predicted. The WS3 LLM
ask-and-average score (n = 25 tweets, cue-stripped) sits at **.973 on
TEST∩150 (n = 52)**, above every k = 25 ridge probe — the declared-scale
reader is the most text-efficient instrument we have at that bundle size.
Candidates with < k tweets: 5 (k = 25), 97 (k = 50), 457 (k = 100) mixed;
36 / 246 / 643 originals.

### E4.4 — Shift penalty (bar: |Δr| < .01, CI covers 0, ridge content spaces, incumbency) — MET

Δr = median of 50 size-matched random-control models − shift-trained model,
paired bootstrap over TEST evaluation candidates:

| axis (eval set) | TF-IDF | doc2vec | w2v | m2v | m2v_strip | behav |
|---|---|---|---|---|---|---|
| incumbent → challengers (n = 179) | −.002 [−.007, +.002] | +.002 [−.003, +.007] | −.004 [−.007, +.001] | +.004 [+.001, +.008] | +.001 [−.003, +.005] | .000 [−.006, +.005] |
| House → senators (n = 23) | −.001 [−.003, +.001] | +.000 [−.001, +.002] | −.001 [−.003, +.001] | −.001 [−.003, +.002] | −.001 [−.004, +.002] | −.000 [−.001, +.001] |

Null as preregistered: training only on incumbents (or only on House
candidates) transfers to challengers (senators) at the same accuracy as a
size-matched random training set. The one CI excluding zero (raw m2v,
+.004) is below the .01 band and is the prefix leak again (incumbents
retweet a different mix of org handles). GBM is noisier (w2v −.021
[−.035, −.008]; doc2vec wide) and not part of the bar. Independents on
TEST (n = 9): ridge r .88–.98, RMSE .09–.22 — no evidence of an
Independent-specific failure at this n.

### E4.5 / E4.6 — Ensembles (bar: stack_all beats best single AND behavioral reference, CIs excl. 0) — MET

| stack (DEV-fit non-negative ridge meta) | TEST r | Δr vs best single (m2v .982) | Δr vs behav ref (.969) |
|---|---|---|---|
| content (TF-IDF .55, doc2vec .48) | .979 | −.003 [−.005, +.000] | +.010 [+.002, +.018] |
| style (m2v .64, behav .31, w2v .07) | .985 | +.003 [+.001, +.006] | +.016 [+.010, +.023] |
| **cross-family (m2v .47, doc2vec .29, behav .27, TF-IDF .01, w2v 0)** | **.986** | **+.005 [+.003, +.007]** | **+.017 [+.011, +.024]** |

RMSE .106 for the cross-family stack vs .123 for raw m2v. The error-family
hypothesis is confirmed in direction but the geometry differs from the
n = 150 synthesis: TEST residual correlations put the four text spaces in
one block (.55–.83) and the behavioral probe apart (.26–.37) — a
**text-vs-behavior** partition, reproducing the DEV OOF matrix
(.61–.84 / .33–.40) almost exactly, not the content-vs-style partition the
synthesis found. The synthesis families were instrument families at n = 150
with the LLM in the mix; at n = 303 with ridge probes the only independent
error source is the behavioral readout, and the stack's gain comes from
mixing it in (weight .27), not from mixing content with style.

E4.6 (TEST∩150, n = 52, descriptive, low power): LLM alone .973;
stack_all@150 .985 → stack_all+llm@150 .984 (LLM meta-weight .14). Adding
the declared-scale score to a stack already at .985 does nothing
measurable at n = 52, consistent with the WS3 finding that the LLM reads
the same lexical signal as TF-IDF.

### E4.7 — Conformal coverage (bar: TF-IDF and doc2vec marginal coverage at α = .10 in [.86, .94]) — MET

| model | α = .10 coverage [CP 95%] | width / SD(truth) | α = .05 coverage | width / SD |
|---|---|---|---|---|
| TF-IDF | **.914** [.877, .943] | 0.75 | .950 | 0.93 |
| doc2vec | **.931** [.896, .957] | 0.80 | .974 | 1.03 |
| w2v | .908 | 0.86 | .957 | 1.03 |
| m2v | .927 | 0.71 | .964 | 0.86 |
| m2v_strip | .944 | 0.84 | .957 | 0.98 |
| behav | .921 | 0.86 | .964 | 1.09 |
| cross-family stack | .941 | **0.61** | .967 | 0.74 |

Marginal coverage is nominal-to-conservative everywhere (OOF-residual
calibration is mildly conservative, as §B.1 stated); the stack gives the
tightest usable interval (± 0.19 on a target with SD ≈ 0.63). Conditional
coverage at α = .10: **under-coverage in the bottom text-volume quartile
(.79–.87, n = 78) and for Independents (.78, n = 9; CI uninformative)**,
over-coverage in Q3–Q4 and for senators (1.00, n = 23) — exactly the
pattern §B.1 expected. Under shift: (a) **k = 25 scarcity breaks the
intervals: coverage .68–.76 at α = .10 and .78–.83 at α = .05** with q̂
held at its full-volume value — the expected under-coverage, but larger
than the ≥ .05 drop expected (it is .15–.22); (b) incumbent-calibrated
q̂ on TEST challengers stays nominal (.92–.96 at α = .10; .97–.98 at
α = .05), matching E4.4's null. Practical reading: the intervals are
trustworthy for candidates featurized at the volume they were calibrated
on and nowhere else; real-data use needs per-volume-stratum calibration
(mondrian conformal) or volume as a conformity-score covariate — both
exploratory, neither run.

### E4.8 — Active learning (bar: diversity or hybrid n\*_AL ≤ .75 × random AND AULC Δ CI > 0, on a content space) — NOT MET

DEV inner folds by design (§B.2); TEST contact only through the n = 40
picks. No content space reaches n\*_AL by n = 60 under *any* strategy
(TF-IDF, doc2vec, m2v_strip all −1; random on `behav` reaches it at 30), so
the n\* clause cannot be satisfied and the verdict is **"no reach"** per
§B.2's last paragraph, not a strategy failure — but it is also not the
"no room" case the paragraph anticipated: there is plenty of room on
TF-IDF, and no query strategy closes it. AULC Δ vs random: TF-IDF
uncertainty **+.050 [.037, .063]**, hybrid +.032, diversity +.015 (all
CI > 0); doc2vec all three strategies *below* random (−.011 to −.014,
CI < 0); m2v_strip null; behav diversity/hybrid below random. The §B.2
expected ordering (diversity > hybrid > uncertainty > random) is reversed:
uncertainty sampling is the only strategy that helps, and only where the
learner is far from ceiling (TF-IDF). TEST r of the n = 40 picks follows
the DEV ordering — TF-IDF: uncertainty .764, random .725, diversity .713,
hybrid .699; doc2vec: hybrid .929, uncertainty .920, random .915,
diversity .882; m2v_strip and behav within noise of random (behav random
.963 is the best n = 40 number on TEST). Reading: at this label budget,
random labeling suffices on every space that is near ceiling, and the
one space with headroom (TF-IDF) wants uncertainty, not diversity. The
resume phrase "label-efficient active learning" is therefore supported
only as "evaluated and found not worth the engineering at n ≤ 60".

## Decisions taken during the session (execution notes; everything decidable was preregistered)

- **Environment**: this session ran on the local Linux VM (Python 3.10,
  fresh `scikit-learn 1.7.2`); 07 fits nothing, so the sklearn-version
  caveat in `REGENERATE.md` does not bite. The dry run was repeated in
  this environment before unsealing; bit-identical outputs were not
  checked against the 09-16 dry run (the dry-run numbers are meaningless
  by construction).
- **Unseal log**: `outputs/unseal_log.txt` captures 07's console output
  through E4.7; the E4.8 block and the BARS line were cut by a pipe close
  and are recovered from `decision.json` / `e48_test_n40.csv` (note
  appended to the log). `decision.json` is the record.
- **Figures** follow `figure_specs.md` as frozen; two small additions
  flagged as such: the Independents table (`fig4_independents_table.csv`)
  and the E4.6 footnote table (`fig5_e46_footnote_table.csv`) are the
  "table, not panel" items the specs asked for; Fig 3's reference marker
  uses the TEST∩150 LLM score from 07 as the spec says.
- **No post-unseal exploration was run** (D9 candidates listed, not
  executed): small-n ridge without standardization; mondrian conformal by
  volume stratum; uncertainty sampling beyond n = 60 on TF-IDF.

## What this closes and what it opens

Closes: the supervised-vs-unsupervised question on this generator
(ceiling-bound; supervision buys calibration and a .005 ensemble gain, not
accuracy); the incumbent→challenger transfer worry for the testbed (null,
twice — E4.4 and E4.7b agree); the confound-gate question for supervised
readers (projection hurts); the Model2Vec edge (it is the RT prefix,
confirmed three ways — strip ablation, originals regime, E4.4 leak).

Opens: (1) **volume-conditional calibration** is the real-data blocker,
not point accuracy — conformal intervals calibrated on full timelines are
badly over-confident on thin ones; (2) **TF-IDF's label hunger** says a
real-data pipeline with DW-NOMINATE labels for ~100 incumbents should not
start from standardized SVD components; (3) active learning is a
non-result at n ≤ 60 and should be dropped from the method list unless a
far-from-ceiling problem motivates it; (4) the real-data follow-on sketch
(§9 of the extensions plan) is still untouched and now has concrete
guidance: behavioral features first where retweets exist, Model2Vec
stripped of handles, cross-family stacking, per-volume calibration.

## Honest-broker notes (repeat wherever WS4 is cited)

- Every TEST number comes from one script run on one date; the models,
  weights, q̂ and figure specs were fixed from DEV labels only. DEV
  development numbers appear in `README.md` and are not results.
- Effective n is 910 candidates (303 TEST), not 104.6k tweets. CIs are
  paired bootstraps over TEST candidates; a .005 difference is real only
  when its CI says so.
- Six bars met, one not met; two preregistered *expectations* were wrong
  in the useful direction (TF-IDF n\* = 320 not ≤ 40; style projection
  hurts; the §B.2 strategy ordering is reversed) and are reported as such.
- Testbed prediction overstates real-data transfer: the generator plants
  ideology lexically and behaviorally with no annotation noise; results
  are upper bounds on what supervision buys.
- Raw Model2Vec's .982 contains retweet-source identity read from
  `RT @OrgHandle:` prefixes; report it with `m2v_strip` (.975) every time.
- E4.6 is n = 52 and descriptive. E4.8 is a DEV-side experiment by design
  with TEST contact only at n = 40.
- D1 (WS3 API scale-up) remains pinned; D4 (writing sample frozen) holds.
