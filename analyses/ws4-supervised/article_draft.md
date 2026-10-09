# What does supervision buy when the unsupervised instrument is already at the ceiling? Seven preregistered answers from 303 sealed candidates

*Draft, 2026-10-09. Fourth paper on the synthetic candidate-language testbed (one per workstream, per decision D3). Companion to `ws4-writeup.md` (narrative) and `README.md` (operational). All TEST numbers come from a single unseal on 2026-10-09.*

## The question

By the end of workstream 3 the testbed had a problem that most applied NLP projects would envy: three unsupervised instruments — a frozen TF-IDF axis (r = .974 with planted ideology), the mean ideology of a candidate's retweet sources (.980), and an LLM asked to rate 25 tweets (.970) — all sat at the generator's lexical ceiling (≈ .973). A supervised model cannot meaningfully beat that, and an August preanalysis confirmed it: a ridge probe on any of five embedding spaces lands at .966–.973 with 910 labels.

So "is supervision more accurate" is not the question. The questions that matter on real data are about **cost and transfer**: labels exist only for incumbents (DW-NOMINATE), challengers are the ones worth predicting, many timelines are thin, and a point estimate without an interval is hard to act on. We preregistered seven experiments on those questions, fixed every model, weight and figure on 607 development candidates, froze the predictions for the 303 held-out candidates, and then read their labels once.

## The protocol

The WS0 seal was designed to keep truth away from every analysis. Supervision needs truth as training data, so the blind habit moved from the corpus boundary to the evaluation boundary. A one-time candidate-level split (DEV 607 / TEST 303, stratified on party × chamber × incumbency × a blind text score) and a label-release script that wrote DEV truth only, hash-stamped in the seal manifest. Seven experiments with one bar each, written down before the release. Every model selection on frozen inner folds within DEV. Then a label-free script that stored every model's prediction for the 303 TEST candidates (7,416 arrays), and finally a script that fits nothing: it verifies the sealed file's hash, reads the 303 labels, scores the stored arrays, and writes a decision file. Six bars met, one not.

## Result 1 — Supervision does not beat the zero-label axis, except by cheating

Ridge probes on TEST: TF-IDF .974, doc2vec .974, word2vec .970, behavioral features .967, Model2Vec .982. The frozen unsupervised TF-IDF axis, which has never seen a label, scores .976 on the same 303 candidates. No supervised single-space probe beats it with a confidence interval that excludes zero — except raw Model2Vec, by +.006 [+.002, +.010].

That exception is instructive rather than impressive. Model2Vec embeds the raw tweet text, and in this corpus a retweet begins with `RT @OrgHandle:`. Twenty organizational handles carry known ideologies; the embedding of the prefix is a behavioral identity leaking into a "content" space. Strip the prefixes and mentions and Model2Vec drops to .975, indistinguishable from the unsupervised axis (m2v − m2v_strip = +.007 [+.004, +.010]). In the originals-only regime of the scarcity experiment, where there are no retweets, raw and stripped Model2Vec are identical to three decimals. We report both numbers everywhere.

Two preregistered expectations were wrong here, usefully. We expected no probe to beat the behavioral reference; every text probe matches or beats it. And we expected projecting out the style confound (the principal component most correlated with retweet share) to be harmless; it hurts every space it touches (TF-IDF −.008, word2vec −.020). The supervised reader needs no confound gate — the gate removes signal.

## Result 2 — Labels are cheap for behavior, expensive for bag-of-words

How many labeled candidates does each space need to come within .02 of its full-data accuracy? Behavioral features: **40**. doc2vec, Model2Vec, word2vec: 160. TF-IDF: **320**. At n = 10 the TF-IDF ridge scores .35 on TEST while the zero-label TF-IDF axis scores .976 — a ridge on a hundred standardized SVD components cannot find at ten labels the direction that unsupervised variance hands over for free, and supervision on this space only overtakes its own unsupervised instrument at roughly 320 labels. The preregistered expectation (TF-IDF and doc2vec at ≤ 40) was wrong; the bar was met only through the behavioral features, whose 20-dimensional retweet-source share is nearly a direct readout of how the generator plants ideology. For a real-data pipeline with DW-NOMINATE labels for a hundred-odd incumbents, this says: do not start from standardized SVD components, and if retweet behavior is observable, start there.

## Result 3 — Thin timelines degrade gracefully, and the LLM is the most text-efficient reader

Training and scoring on k tweets per candidate: at k = 25 the content spaces score .93–.95 (bar .90, met); at k = 10, .87–.90; at k = 5, .77–.84. Behavioral features degrade fastest in the mixed regime (.89 at k = 25, .62 at k = 5) and are dead without retweets. The WS3 LLM score, built from 25 cue-stripped tweets, sits at .973 on the TEST candidates it covers (n = 52) — above every k = 25 supervised probe. At that bundle size the declared-scale reader extracts more from the text than any learned one.

## Result 4 — Training on incumbents transfers to challengers, twice over

A model trained only on DEV incumbents, evaluated on TEST challengers, against fifty size-matched random training sets: Δr = −.002 [−.007, +.002] for TF-IDF, +.002 [−.003, +.007] for doc2vec, within ±.004 for every space; House-trained models on senators, within ±.001. The one interval excluding zero (raw Model2Vec, +.004) is the prefix leak again — incumbents retweet a different mix of handles. The generator does not condition on incumbency, so this null validates the harness rather than making a claim about Congress; but the same null appears independently in the conformal experiment below, which is the point of running it both ways.

## Result 5 — The ensemble gain is real, small, and comes from behavior, not style

A non-negative ridge stack across all five spaces, with weights fit on DEV (Model2Vec .47, doc2vec .29, behavioral .27, TF-IDF .01, word2vec 0), scores .986 on TEST: +.005 [+.003, +.007] over the best single probe and +.017 [+.011, +.024] over the behavioral reference, RMSE .106 against .123. The bar is met. But the residual geometry is not the one the n = 150 synthesis found. On TEST, the four text spaces form one residual block (correlations .55–.83) and the behavioral probe stands apart (.26–.37): a text-versus-behavior partition, reproducing the DEV out-of-fold matrix almost exactly, not the content-versus-style families the synthesis saw with the LLM in the mix. The stack's gain comes from mixing in the behavioral readout. Adding the LLM score to the stack on the 52 candidates where it exists changes nothing (.985 → .984, meta-weight .14).

## Result 6 — The intervals are calibrated, until the text gets thin

Split-conformal intervals on top of each ridge probe, calibrated on DEV out-of-fold residuals: marginal coverage at α = .10 is .91 (TF-IDF), .93 (doc2vec), .91–.94 elsewhere; the stack gives the tightest interval, ± 0.19 on a target with standard deviation 0.63. The bar is met. Conditional coverage shows the expected weak spots — .79–.87 in the bottom text-volume quartile, .78 for the nine Independents — and the expected strengths. Under incumbent→challenger shift, with the quantile calibrated on incumbents only, coverage stays nominal (.92–.96): E4.4's null, seen from the other side.

Under text scarcity it does not. Score a full-volume model on candidates featurized from 25 tweets, with the full-volume quantile held fixed, and coverage at α = .10 falls to **.68–.76**; at α = .05, to .78–.83. We preregistered under-coverage of at least .05; we got .15–.22. The intervals are trustworthy for candidates featurized at the volume they were calibrated on and nowhere else. For real data this is the operative finding of the workstream: point accuracy survives thin timelines, calibration does not, and any deployment needs per-volume-stratum calibration (Mondrian conformal) or volume as a covariate in the conformity score. Neither was run; both are exploratory.

## Result 7 — Active learning is not worth it at this budget

Pool-based active learning simulated inside the DEV inner folds — seed set of 10, batches of 5, to 60 labels; random, uncertainty, diversity and hybrid query rules; 100 curves per strategy per space. No content space reaches its ceiling-adjacent target by 60 labels under any strategy, so the preregistered bar cannot be met ("no reach" by the preregistration's own rule). The area-under-curve comparison is informative anyway: uncertainty sampling helps on TF-IDF (+.050 [+.037, +.063] over random) and nothing else helps anywhere; on doc2vec all three strategies are *worse* than random. The expected ordering (diversity first) is reversed. The n = 40 models scored on TEST agree: TF-IDF uncertainty .764 against random .725; everything else within noise. On a near-linear, near-ceiling problem, diversity sampling chases outliers and random labeling is already close to optimal. We will stop listing active learning as a method unless a far-from-ceiling problem motivates it.

## What this buys, and what it doesn't

Supervision on this testbed buys calibration, a small and certified ensemble gain, and a transfer guarantee across the incumbent/challenger and House/Senate boundaries. It does not buy accuracy over a zero-label axis, and on the bag-of-words space it costs hundreds of labels to catch up. The workstream's single most actionable finding is negative and about intervals, not points: a conformal band calibrated on rich timelines is badly over-confident on thin ones.

## Limitations

The generator plants ideology lexically and behaviorally, with no annotation noise; every number here is an upper bound on what supervision buys on real tweets. Effective n is 303 TEST candidates, and every comparison carries a paired-bootstrap interval that should be read before the point estimate. The LLM-as-feature result is n = 52 and descriptive. The active-learning curves are development-fold numbers by design, with TEST contact only at n = 40. Raw Model2Vec's .982 is partly retweet-source identity; the stripped .975 is the honest contextual number. The WS3 API scale-up remains deferred, so the LLM comparisons rest on the 150-candidate pilot. No analysis has been run against TEST after the unseal; the three listed candidates (unstandardized small-n ridge, volume-stratified conformal, uncertainty sampling past 60) would be exploratory and labeled as such.
