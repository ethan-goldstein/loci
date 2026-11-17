# LOCI results

Generated 2026-11-17 by `loci bench`, `loci eval`, `loci ab`, `loci repeats` and `loci report`. Every number below is measured on this machine; re-run the commands to reproduce.

Nothing here contains conversation text. Projects are labelled by size (P1 is the largest), never by name.

## 1. Is the index correct, and how fast is it?

Machine: Apple M5, 10 cores, 24 GB, Node v26.7.0, darwin-arm64. Single thread, no SIMD, no native code.

HNSW parameters: M = 16, M0 = 32, efConstruction = 200. Recall is the share of the true 10 nearest neighbours (found by scanning every vector) that the graph returns.

**real corpus**: 7,692 vectors, 768 dimensions, 500 queries.

- Build: 8.6 s. Graph: 1.0 MB on top of 22.5 MB of vectors.
- Mean layer-0 degree 18.3, top layer 3.
- Exact scan (the oracle): p50 3.95 ms, p95 4.51 ms.

| ef | recall@10 vs exact | p50 ms | p95 ms | queries/s | speed-up vs exact |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.9010 | 0.221 | 0.316 | 4,457 | 17.9x |
| 32 | 0.9526 | 0.386 | 0.522 | 2,563 | 10.2x |
| 64 | 0.9814 | 0.673 | 0.930 | 1,459 | 5.9x |
| 128 | 0.9956 | 1.257 | 1.619 | 801 | 3.1x |
| 256 | 0.9990 | 2.179 | 2.725 | 462 | 1.8x |

On the real corpus at ef = 64 the graph returns 98.1% of the true top 10 in 0.67 ms.

The real corpus is small, so the same code is also measured on a larger synthetic set. These vectors are random clusters, not text: they show how the index scales, not how it behaves on language.

**synthetic (64 Gaussian clusters)**: 50,000 vectors, 128 dimensions, 500 queries.

- Build: 11.6 s. Graph: 6.4 MB on top of 24.4 MB of vectors.
- Mean layer-0 degree 18.5, top layer 3.
- Exact scan (the oracle): p50 4.62 ms, p95 5.24 ms.

| ef | recall@10 vs exact | p50 ms | p95 ms | queries/s | speed-up vs exact |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.8158 | 0.057 | 0.078 | 16,945 | 81.1x |
| 32 | 0.9386 | 0.083 | 0.106 | 11,718 | 55.8x |
| 64 | 0.9916 | 0.121 | 0.154 | 8,121 | 38.3x |
| 128 | 0.9996 | 0.168 | 0.201 | 6,051 | 27.5x |
| 256 | 1.0000 | 0.222 | 0.309 | 4,442 | 20.8x |

### Lexical index against SQLite FTS5

The from-scratch BM25 uses the same formula and tokeniser as SQLite FTS5, so FTS5 serves as its oracle. Over 500 real prompts, 100.00% of its top-10 results match FTS5's top 10 (ties at the cut-off counted as matches).

- 7,692 documents, 20,729 distinct terms, built in 142 ms.
- Query latency: p50 0.224 ms, p95 0.567 ms.
- Process memory with everything loaded: 294 MB.

## 2. Does retrieval find the right earlier work?

No manual labels. Each query is a real prompt; the engine may only return chunks written **before** that prompt and never from the same session. A result counts as relevant when it comes from the same project as the prompt.

981 queries from 81 sessions across 11 projects, over a corpus of 7,692 chunks (median 4,310 eligible candidates per query).

Scores are macro-averaged by project, over the 9 projects with at least 10 queries (977 queries). Brackets are 95% intervals from a bootstrap that resamples whole sessions within each project.

| system | project-precision@10 | MRR | precision lift over recency |
|---|---:|---:|---:|
| random | 0.207 [0.196, 0.217] | 0.284 [0.268, 0.298] | -0.290 [-0.349, -0.243] |
| majority | 0.207 [0.183, 0.217] | 0.207 [0.183, 0.217] | -0.289 [-0.366, -0.236] |
| recency | 0.496 [0.446, 0.557] | 0.559 [0.505, 0.619] | baseline |
| bm25 | 0.328 [0.300, 0.351] | 0.492 [0.447, 0.518] | -0.168 [-0.236, -0.121] |
| dense | 0.367 [0.327, 0.390] | 0.526 [0.460, 0.553] | -0.129 [-0.204, -0.080] |
| hybrid | 0.368 [0.332, 0.390] | 0.541 [0.482, 0.565] | -0.128 [-0.200, -0.081] |
| hybrid+recency | 0.430 [0.376, 0.458] | 0.628 [0.550, 0.666] | -0.066 [-0.137, -0.026] |

- **random** is the closed-form expectation of a random ranking: what the corpus gives away for free.
- **majority** always answers with the largest project.
- **recency** returns the 10 most recent earlier chunks and ignores the prompt entirely. It is the baseline to beat, because the last thing you worked on is usually the same project.
- **hybrid** fuses the dense and BM25 rankings by reciprocal rank.
- **hybrid+recency** adds the recency list as a third ranking in the same fusion, with no tuned weights.

**Verdict.** Hybrid retrieval is well above chance (0.368 against 0.207) but loses to the recency baseline on project-precision@10 (-0.128 [-0.200, -0.081]). On MRR it is not distinguishable from recency (-0.018 [-0.102, +0.028]).

### How stable is that verdict?

The headline is a mean over projects. Projects with fewer than 10 queries are left out of it, because otherwise a project with one query weighs as much as a project with five hundred. That floor was added after exactly this happened: between two runs a few hours apart, one new single-query project moved the every-project average by about 0.03 and changed its verdict, while no other project's score moved. The same comparison under each way of averaging, with the lift over recency in brackets:

| averaging | queries | projects | recency | hybrid | hybrid+recency |
|---|---:|---:|---:|---:|---:|
| macro, projects with 10+ queries (headline) | 977 | 9 | 0.496 | 0.368 (-0.128 [-0.200, -0.081]) | 0.430 (-0.066 [-0.137, -0.026]) |
| macro, every project | 981 | 11 | 0.406 | 0.383 (-0.023 [-0.080, +0.017]) | 0.458 (+0.052 [-0.006, +0.084]) |
| micro, every query counts equally | 981 | 11 | 0.521 | 0.444 (-0.077 [-0.344, +0.107]) | 0.470 (-0.051 [-0.255, +0.089]) |

Hybrid's estimate is below recency in all 3 views, but the gap is outside the noise in only 1 of them. Read that as "probably behind", not as settled.

Adding recency as a third signal: hybrid+recency loses to recency on precision (-0.066 [-0.137, -0.026]) and is not distinguishable from it on MRR (+0.069 [-0.016, +0.112]).

The recency-fused ranker was added after the first run showed plain hybrid losing to recency, and it is scored on the same queries. It has no tuned parameters, but it is still a post-hoc change: treat its numbers as exploratory until the eval is re-run on sessions recorded after this date.

### Ablation: project names and paths hidden

A ranker could score well just by matching folder names. Here every project name and file path is replaced with a placeholder in both the prompts and the corpus, then the corpus is re-embedded and the eval repeated.

| system | project-precision@10 | MRR | precision lift over recency |
|---|---:|---:|---:|
| bm25 (masked) | 0.326 [0.298, 0.348] | 0.500 [0.452, 0.525] | -0.170 [-0.239, -0.122] |
| dense (masked) | 0.366 [0.321, 0.390] | 0.526 [0.458, 0.554] | -0.130 [-0.207, -0.080] |
| hybrid (masked) | 0.368 [0.329, 0.389] | 0.549 [0.484, 0.573] | -0.128 [-0.201, -0.080] |

Hiding names and paths moves hybrid precision by 0.000 (0.368 to 0.368): the result does not depend on those surface cues.

### By project

Precision@10 per project. Recency wins where a project was worked in one concentrated burst; retrieval wins where work on a project was spread out or interleaved with others.

| project | queries | random | recency | hybrid | hybrid+recency |
|---|---:|---:|---:|---:|---:|
| P1 | 538 | 0.449 | 0.419 | 0.518 | 0.503 |
| P2 | 119 | 0.192 | 0.533 | 0.139 | 0.215 |
| P3 | 91 | 0.982 | 0.978 | 0.981 | 0.980 |
| P4 | 66 | 0.146 | 0.867 | 0.106 | 0.256 |
| P5 | 60 | 0.015 | 0.340 | 0.032 | 0.092 |
| P6 | 54 | 0.015 | 0.909 | 0.402 | 0.572 |
| P7 | 20 | 0.033 | 0.030 | 0.330 | 0.430 |
| P8 | 17 | 0.005 | 0.224 | 0.465 | 0.488 |
| P9 | 12 | 0.022 | 0.167 | 0.342 | 0.333 |
| P10 | 3 | 0.038 | 0.000 | 0.200 | 0.167 |
| P11 | 1 | 0.002 | 0.000 | 0.700 | 1.000 |

### Known-item test: title to session

Each session has a short title that is never indexed. Using only the title as the query, does a chunk of that session come back? 74 sessions, searching the whole corpus.

| system | hit@1 | hit@10 | MRR |
|---|---:|---:|---:|
| bm25 | 0.500 | 0.851 | 0.601 |
| dense | 0.662 | 0.865 | 0.734 |
| hybrid | 0.649 | 0.851 | 0.717 |
| random (expected) | | 0.114 | |

## 3. Does recalled context make an agent's answers better?

The hypothesis, arms, metric, sample and decision rule were committed in [docs/memory-ab-prereg.md](docs/memory-ab-prereg.md) before the harness was written or run.

One session is held out at a time. A local model writes questions from that session's own text; a question is kept only if its answer appears word for word in the passage and not in the question. The same model then answers each question once per arm, with nothing but the context changed. Recalled context is formatted as the MCP `search` tool returns it. Answers from all arms are pooled and shuffled, and the judge sees only the question, the reference answer and the answer to grade.

268 questions from 67 sessions across 14 projects, over a corpus of 7,692 chunks. 67 sessions were eligible; 335 passages were tried and 80% gave a kept question (most common rejection: answer length, 39). Writing, answering and judging all used `llama3.2:3b` (a80c4f17acd5) on this machine at temperature 0. Each session counts once. Brackets are 95% intervals from a bootstrap that resamples whole sessions.

| arm | context | judged correct | contains reference | wrong, not an abstention |
|---|---|---:|---:|---:|
| none | nothing | 0.030 [0.011, 0.049] | 0.007 [0.000, 0.019] | 0.116 [0.082, 0.153] |
| memory | search over earlier sessions only | 0.060 [0.026, 0.093] | 0.041 [0.011, 0.075] | 0.552 [0.485, 0.619] |
| recency | the most recent earlier chunks, question ignored | 0.015 [0.000, 0.034] | 0.007 [0.000, 0.019] | 0.228 [0.160, 0.295] |
| oracle | the passage the question was written from | 0.735 [0.679, 0.784] | 0.765 [0.716, 0.810] | 0.254 [0.201, 0.313] |
| instore | search over the whole store | 0.496 [0.437, 0.552] | 0.519 [0.451, 0.586] | 0.474 [0.414, 0.534] |

- **judged correct** is the model judge. **contains reference** needs no model: the reference answer appears in the answer.
- **wrong, not an abstention** is the share of answers that were wrong (by judged correctness) without saying "I do not know": the cost of context that misleads.

| difference | judged correct | contains reference |
|---|---:|---:|
| memory - none (primary) | +0.030 [-0.007, +0.071] | +0.034 [+0.004, +0.071] |
| memory - recency | +0.045 [+0.015, +0.075] | +0.034 [+0.007, +0.067] |
| recency - none | -0.015 [-0.041, +0.011] | +0.000 [-0.015, +0.015] |
| oracle - none | +0.705 [+0.646, +0.761] | +0.757 [+0.709, +0.802] |
| instore - none | +0.466 [+0.399, +0.526] | +0.511 [+0.444, +0.578] |

### Can the grading be trusted?

The judge was checked on 100 invented answers with known labels, none of them used while writing its prompt.

| grader | sensitivity | specificity | balanced accuracy |
|---|---:|---:|---:|
| judge (`llama3.2:3b`) | 0.825 | 1.000 | 0.912 |
| contains reference (no model) | 0.750 | 1.000 | 0.875 |

The bar set in advance was 0.90 balanced accuracy. The judge cleared it, so the verdict rests on judged correctness.
On the 1,340 real answers the two graders agreed on 92%: 50 were correct for the judge only and 51 for the string check only.

**Verdict: null.** No measurable difference between answering with recalled context and answering without it: +0.030 [-0.007, +0.071] on judged correctness. The interval rules out a gain of 0.10 or more, the smallest gain set in advance as worth having.

The two outcomes do not agree on that. On the contains-reference outcome the difference is +0.034 [+0.004, +0.071], an interval that lies above zero, where the headline's includes zero. The verdict follows the outcome chosen in advance; the other is a reason not to read it as more settled than it is.

For scale: with the source passage in hand the model was right 74% of the time, against 3% with nothing (oracle - none = +0.705 [+0.646, +0.761]). Memory recovered 4% of that gap.

### How stable is that verdict?

The headline counts each session once. Section 2 showed that a verdict can turn on how groups are weighted, so the same difference is given here under five ways of averaging, two of them with a floor on group size. The list was fixed while the run was in progress, before any score was read. Sessions are what is resampled in every row; in the two project rows they are resampled within their project.

| averaging | sessions | questions | projects | memory - none |
|---|---:|---:|---:|---:|
| mean over sessions (the preregistered headline) | 67 | 268 | 14 | +0.030 [-0.007, +0.071] |
| mean over questions | 67 | 268 | 14 | +0.030 [-0.007, +0.071] |
| mean over sessions that produced all 4 questions | 67 | 268 | 14 | +0.030 [-0.007, +0.071] |
| mean over projects, every project | 67 | 268 | 14 | +0.046 [+0.018, +0.073] |
| mean over projects with 10+ questions | 54 | 216 | 5 | +0.028 [-0.007, +0.071] |

The estimate is above zero in all 5 views, but the interval excludes zero in only 1 of them: mean over projects, every project. The verdict stays with the preregistered headline.
Every session produced all 4 questions, so the first three rows are the same calculation.
The every-project row gives a project with one session the same weight as a project with twenty, and a one-session project adds no spread to its interval. That is why the preregistration did not use it as the headline, and why the last row sets a floor.

Secondary results, reported as planned. No verdict rests on them.

- **Against recency.** Memory beats the most recent earlier chunks (+0.045 [+0.015, +0.075]).
- **When the answer is in the store.** Searching the whole store instead of only earlier sessions gives instore - none = +0.466 [+0.399, +0.526]. This is biased upward: each question was written from the stored passage and shares its wording.
- **On the other outcome.** Memory - none is +0.034 [+0.004, +0.071] on contains reference.
- **Context that misleads.** With nothing, the model said "I do not know" to 85% of questions and gave a wrong answer to 12% (0.116 [0.082, 0.153]). With recalled context it abstained on 39% and gave a wrong answer to 55% (0.552 [0.485, 0.619]). Recalled text that does not hold the answer still gets answered from.
- **Was the answer there to find?** For 28% of questions the reference answer appears word for word somewhere in the earlier chunks memory was allowed to search. Where it does (75 questions), memory - none is +0.111 [+0.022, +0.207]. Where it does not (193 questions), it is -0.006 [-0.033, +0.019].

## What these numbers do not show

- "Same project" is a coarse label. It shows the engine finds the right neighbourhood, not that it finds the passage that answers the prompt.
- One person, one corpus, one embedding model. Nothing here says how it behaves on anyone else's history.
- The A/B used one small local model (`llama3.2:3b`) to write the questions, answer them and judge the answers. A stronger model might use the same context better, or need it less.
- Nobody read the generated questions as part of the run. Their quality shows only indirectly, in how many passed the checks and in how far the oracle arm sits above the no-context arm. The owner can audit them in the private store.
- The no-context arm had no tools and no repository. A real agent without memory could still read the code, so this measures what recalled context adds to a model that has nothing else.
- Holding a session out measures whether earlier work carries over to a later session. It is a harder test than recalling something already in the store, which only the upward-biased instore arm touches.

## 4. How much do you repeat yourself?

Every prompt typed by hand is split into sentences; pasted code, logs and contentless go-aheads are dropped. Each sentence is embedded locally and every pair is compared exactly. Two sentences count as the same thing when their similarity is far above that of a typical unrelated pair, and a group counts as a habit when it shows up in several separate sessions on more than one day.

- 3,486 sentences from 1,304 prompts. 11 prompts were left out as program-written templates, and 2 groups as same-day bursts.
- Threshold: similarity 0.853. A typical cross-session pair scores 0.622 with a robust spread of 0.046, so the threshold sits 5.0 spreads above it.
- Cross-session pairs above the threshold: **1,130** out of 5,824,954 compared. If similarities of unrelated sentences followed a normal curve, about 1.7 would be expected by chance.
- **79 things were said in 3 or more sessions on at least two days**: 34 instructions, 7 recurring task openers, 3 questions and 35 other statements.
- Those groups hold 469 sentences, 13.5% of everything typed.

| sessions a group spans | groups |
|---|---:|
| 3 | 44 |
| 4 | 16 |
| 5 | 8 |
| 6 to 7 | 8 |
| 8 to 11 | 2 |
| 12 or more | 1 |

What this does not show: that a repeated instruction was needed each time, or that writing it down once as a standing rule would have saved the repetition. The kind labels (instruction, task, question) come from simple word patterns, not from reading for meaning. The chance figure assumes a normal curve, which real similarities only roughly follow, so read it as "far more than chance", not as an exact rate.
