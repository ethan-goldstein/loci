# Preregistration: does recalled context make an agent's answers better?

Written and committed on 2026-11-16, before the harness (`src/prove/ab.ts`) existed and before any answering arm was run on the private store. The git history is the evidence of that order: this file, then the harness, then the results.

Anything changed after this commit is listed under [Deviations](#deviations) at the bottom, with the reason.

## Question

`RESULTS.md` shows that the index is correct and that retrieval finds the right neighbourhood. It says nothing about whether handing the retrieved text to an agent makes the agent's answers better. This experiment tests that.

## Hypothesis

**H1.** A model answers questions about a session's content correctly more often when it is given context recalled by LOCI from *earlier* sessions than when it is given no context.

**H0.** There is no difference.

The test is two-sided. A result where recalled context makes answers worse is reported as a negative result, not folded into "no effect".

## Design

One held-out session at a time. For each eligible session, questions are written from that session's own text. The session is then hidden from retrieval, along with everything dated at or after its first chunk, and the same question is answered several times by the same model under different context conditions. Every comparison is paired: same question, same model, same instructions, only the context differs.

### Eligible sessions

A session is eligible when all of these hold, computed from the store at run time:

- it has a title of at least 8 characters (the same filter as the known-item test in `src/prove/eval.ts`);
- it has at least 8 chunks;
- at least 4 of them are assistant chunks of 400 characters or more (the candidate passages);
- at least 50 chunks from *other* sessions are dated strictly before its first chunk.

All eligible sessions are used. Counts taken before this file was written give 66.

### Questions

Up to 4 questions per session. No manual labelling.

1. The session's candidate passages are shuffled with a seeded generator (seed 20261116 plus the session's key).
2. For each passage in that order, a local model is asked for one question and a short answer copied from the passage, as structured JSON, at temperature 0.
3. The pair is kept only if it passes fixed, deterministic checks:
   - the answer appears verbatim in the passage (ignoring case and runs of whitespace), is 3 to 80 characters and at most 10 words, and contains a word of 3 or more letters or a digit;
   - the question is 20 to 300 characters, ends with a question mark, does not contain the answer, and does not point at the passage itself ("the passage", "the text", "the excerpt", "the conversation", "the assistant", "the speaker", "the author", "this session", "the above");
   - it is not a duplicate of an earlier question in the same session.
4. This stops at 4 kept questions or after 12 passages, whichever comes first. A session with no kept question contributes nothing and is counted as dropped.

Every arm sees the question in the same form, prefixed with the project name: `In the project <name>: <question>`. A real agent knows which project it is working in, and the prefix is identical across arms. If the prefixed question contains the answer, the pair is rejected. The session's title is **not** shown to any arm: it was written by a model that read the whole session, so it would hand every arm a summary of the held-out text.

### Arms

| arm | context given to the answering model |
|---|---|
| `none` | nothing |
| `memory` | LOCI hybrid search for the question, top 8, restricted to chunks dated before the session's first chunk, the session itself excluded |
| `recency` | the 8 most recent chunks under the same restriction, ignoring the question |
| `oracle` | the passage the question was written from |
| `instore` | LOCI hybrid search for the question, top 8, over the whole store with nothing excluded |

- `memory` against `none` is the experiment.
- `recency` is the control that beat retrieval in the retrieval eval. It tells apart "the right context helps" from "any recent context from the same work helps".
- `oracle` is a positive control and a ceiling. If the pipeline cannot show a difference between holding the answer and holding nothing, a null on `memory` means nothing.
- `instore` is the easy case: the answer is in the store and the question was written from it, so it shares the passage's wording. It is biased upward and is reported only to separate "the earlier history did not contain the answer" from "the engine could not surface it". No verdict rests on it.

Search uses the engine's own `before` and `excludeSessionPk` filters, mode `hybrid`, the MCP server's defaults (k = 8, snippets of 700 characters), and the same untrusted-data wrapper the MCP server puts around recalled text, so the model sees what an agent calling the `search` tool would see.

### Answering

One local model, the same system prompt in every arm, temperature 0, fixed seed. It is told to reply with one short sentence, not to mention notes or sources, and to say "I do not know" when it cannot tell. Answers are cut to 300 characters before judging.

### Judging, blind to the arm

Each answer is judged on its own. The judge receives the question, the reference answer and the answer to grade, and nothing else: not the arm, not the context, not the other arms' answers. It returns correct or not correct. The function that judges has no access to the arm by construction, and a unit test checks that.

A second outcome needs no model at all: **contains**, true when the reference answer appears in the answer after lower-casing and collapsing whitespace.

### Models

Generation, answering and judging all use `llama3.2:3b` through Ollama on loopback, the only generative model installed on this machine. Nothing is sent to a hosted model. This is a weak model for all three jobs, which is why the next section exists.

### Judge calibration

The harness carries 30 invented question-and-reference items, each with 5 candidate answers whose correctness is known by construction (two correct: one near-verbatim and one reworded; three wrong: a different value, an abstention, and a confident answer about something else). The text is invented and safe to publish. The first 10 items are for developing the judge prompt. The last 20 (100 judged answers) are never used for that and give the reported numbers: sensitivity, specificity and balanced accuracy.

## Metric

**Primary outcome:** judged correctness (0 or 1) of each answer.

**Primary statistic:** for each session, the mean correctness in `memory` minus the mean correctness in `none` over that session's questions; then the mean of those per-session differences. Each session counts once however many questions it produced.

**Interval:** 95% percentile interval from `clusterBootstrap` in `src/prove/bootstrap.ts`, resampling whole sessions, 2,000 resamples, its default seed. Sessions are resampled in a single stratum rather than within project: 4 of the 13 projects with eligible sessions have exactly one, and a one-session stratum contributes no spread, which would make the interval too narrow.

## Sample size

All eligible sessions, expected 66, with up to 4 questions each: at most 264 questions and 1,320 judged answers. The store is one person's history, so this is the whole population available, not a sample chosen for power.

What that can detect: if per-session differences have a standard deviation near 0.3, 66 sessions give a 95% half-width of about 0.07. Effects smaller than roughly 7 points of accuracy will probably not be distinguishable from zero.

There is one run. No interim looks, no stopping early, no adding sessions after seeing a result.

## Decision rule

Let Δ be the primary statistic with interval [lo, hi].

- **Positive** if lo > 0: recalled context made answers measurably better.
- **Negative** if hi < 0: recalled context made answers measurably worse.
- **Null** otherwise. A null is then qualified: if hi < 0.10 it "rules out a gain of 10 points or more"; if not, it "cannot rule out a useful gain". Ten points, one more correct answer in every ten questions, is the smallest gain treated here as worth having.

Three gates, fixed now, decide whether that verdict can be trusted:

1. **Judge.** If the judge's balanced accuracy on the 100 held-out calibration answers is below 0.90, the judged numbers are still reported but the verdict is taken from the **contains** outcome instead, and the report says so.
2. **Positive control.** If the interval for `oracle` minus `none` does not lie entirely above +0.20 on the outcome in use, the experiment is declared **not interpretable**: it could not detect the best case.
3. **Yield.** If fewer than 30 sessions contribute a question, the experiment is declared **not interpretable**.

## Secondary analyses

Reported with intervals, labelled secondary, and no verdict rests on any of them:

- `memory` minus `recency`, `recency` minus `none`, `oracle` minus `none`, `instore` minus `none`;
- the primary contrast on the outcome not used for the verdict;
- the primary contrast macro-averaged by project with the project-stratified bootstrap used in the retrieval eval, to show whether the largest project carries the result;
- the primary contrast split by whether the reference answer appears verbatim anywhere in the chunks the `memory` arm was allowed to search. Where it does not appear, memory should not help much; a large gain there would point to a leak or a biased judge;
- per arm: the share of answers that were wrong but not an abstention, since context that makes a model confidently wrong is a cost;
- how often the judge and the **contains** outcome agree on the real answers.

## Reported whatever happens

Every arm, both outcomes, the calibration numbers, the yield, the gates and the verdict go into `RESULTS.md` through `loci report`, whether the result is positive, null, negative or not interpretable.

## Privacy

Questions, references and answers are text derived from private conversations. They are written only to the private store directory, with owner-only permissions, so the owner can audit them. The terminal output, the JSON report and `RESULTS.md` hold counts and scores only. No conversation text was read by the author of this file, and none is printed by the harness.

## Limitations known in advance

- A 3B local model writes the questions, answers them and judges them. Its questions may be vague, and its judging is imperfect; the calibration set and the judge-free outcome measure the second problem, not the first.
- Nobody reads the generated questions as part of the run. Their quality shows up only indirectly, in the yield and in how far `oracle` sits above `none`.
- The `none` arm has no tools and no repository. A real agent without memory could still read the code. This measures what recalled context adds to a model that has nothing else.
- The held-out design measures whether earlier work transfers to a later session. It does not measure recall of something already in the store; `instore` touches that, with the bias noted above.
- One person, one corpus, one embedding model, one small generative model.

## What was known when this was written

- The retrieval results already in `RESULTS.md`, including that hybrid retrieval loses to recency on project-precision@10.
- Shape of the store, counts only: 96 sessions with chunks, 66 eligible under the rule above across 13 projects (21, 10, 9, 7, 7, 2, 2, 2, 2, 1, 1, 1, 1), 54 of the 66 with earlier work in the same project, median chunk length 685 characters, 36 sessions overlapping the previous one in time.
- Speed of the model on invented text (about 50 tokens per second out), and that a naive question-writing prompt returned an unusable pair on one invented passage, which is why the checks in "Questions" are strict.
- No arm had been run on private text and no question had been generated from it.

Prompts are developed on invented text only and are fixed in the harness commit that precedes the run. A smoke run on at most 2 sessions may be made to check that the harness does not crash on real data; its scores are not used to change anything here.

## Deviations

None at the time of this commit.

### Amendment 1: more than one way of averaging

Added 2026-11-16, while the run was in progress and before any score was read.

During the run, the retrieval eval on `main` was found to have changed its verdict between two runs because one project with a single query carried a full share of a project average. The primary statistic here was already built to avoid that: sessions are resampled in one stratum and each counts once. It and the decision rule are unchanged, and the verdict still comes from them alone.

What is added is a table, shown next to the verdict, of the same `memory` minus `none` difference under five ways of averaging:

1. mean over sessions (the preregistered headline);
2. mean over questions, with whole sessions still resampled;
3. mean over sessions that produced all 4 questions;
4. mean over projects, every project (already listed above as a secondary analysis);
5. mean over projects with at least 10 questions, the same floor the retrieval eval now uses.

If these disagree with the headline, the report says so in plain words.

State of the experiment when this was written (22:32 UTC): the full run had started at 22:26 UTC and was still writing questions. Its cache held 250 written questions, the 100 calibration judgements, and the 40 answers and 31 judgements left by the two-session smoke run. No score had been read from any of them, including the smoke run and the calibration. The summary is a pure function of the per-question rows, so adding a view cannot change the data it is computed from.

### Notes on the run

Written 2026-11-16 at 22:50 local, with the run still unfinished and no score read. None of these changes the hypothesis, the arms, the metric or the decision rule.

1. **The store was rebuilt between this file and the run.** The owner had the store re-ingested with a tighter email redaction rule, which also picked up that day's sessions. At run time it held 7,692 chunks and 67 eligible sessions, against the 7,661 and 66 counted above. The eligibility rule is the one stated above, applied to the store as it stood when the run started. The store has not changed since the run started.
2. **Identical answers are judged once.** When two arms give the same answer to the same question, the judge sees that (question, reference, answer) once and both arms get its verdict. This was not stated in advance. It cannot favour an arm, and it removes one way the same answer could be graded two ways.
3. **The held-out session's title is removed from result headers.** The search tool prints each result's session title. For a result that comes from the held-out session (possible only in `instore` and `oracle`), that title is left out, so "the title is not shown to any arm" holds exactly. Titles of other sessions are shown as the tool shows them.
4. **A smoke run on two sessions was made**, as allowed above. Only its counts and timing were read. Its generations are in the cache and are reused by the full run.
5. **The run was interrupted and resumed from its cache.** It stopped at 18:02 local, 140 questions into the answering phase, when the laptop slept. Since then the laptop has been asleep with its lid closed on battery, and the run has advanced only in the few seconds it wakes at a time. (It finished that way overnight; see below.) Every question, answer and judgement is generated once and stored; nothing is regenerated on resume. Requests that stall now retry on a longer schedule (commits `c008a00`, `75773a2`). The prompts did not change: the prompt hash is `f3ce4fe79e65` throughout.
6. **`main` was merged twice** (`820ad89`, `8a45fef`) while the run was stopped or in progress. Neither merge touched the harness, the prompts or the seeded generator it shuffles with.

### After the scores were read

The run finished on 2026-11-17 at 06:46 local (11:46 UTC): 67 sessions, 268 questions, 1,340 answers, 975 distinct real answers judged plus the 100 calibration cases. The store was the same from the first question to the last. The scores were first read after that.

The verdict was produced by the rule above with nothing changed: the judge cleared its gate (0.912 balanced accuracy), the positive control cleared its gate, 67 sessions contributed, and the interval for `memory` minus `none` on judged correctness includes zero.

Three changes were then made to how the report words the result. None changes a number, a table or the verdict, and each is listed here because it was made knowing the result:

1. **One sentence was made less favourable to memory.** The sentence under the averaging table used to end `Read that as "probably a gain", not as settled.` That over-read this table: three of its five rows are the same calculation (every session produced all 4 questions), and the one row whose interval excludes zero is the every-project average that this file set aside in advance for giving one-session projects a full share and too narrow an interval. The sentence now names the rows that exclude zero and says the verdict stays with the headline, and two sentences of explanation follow it.
2. **A disagreement between the two outcomes is stated next to the verdict.** On the model-free outcome the same difference has an interval just above zero. That was already in the tables and in the secondary list; it is now also said in the paragraph after the verdict, so a reader who stops there sees it.
3. **The "wrong, not an abstention" analysis got a sentence.** It was planned above and already in the arm table. Its result is large (12% without context, 55% with), so it is now also written out in the secondary list.
