import type { AbReport, Arm, View } from './ab.ts';
import type { RepeatsReport } from '../insight/repeats.ts';
import type { BenchReport, IndexBench } from './bench.ts';
import type { Interval } from './bootstrap.ts';
import type { EvalReport, SystemScore } from './eval.ts';

/**
 * Renders RESULTS.md from the benchmark, eval and A/B output. Numbers only: no conversation text, no
 * project names (projects are labelled P1, P2, ... by size), nothing that identifies the corpus.
 */

const f = (x: number, d = 3) => (Number.isFinite(x) ? x.toFixed(d) : 'n/a');
const ci = (i: Interval, d = 3) => `${f(i.value, d)} [${f(i.lo, d)}, ${f(i.hi, d)}]`;
const signed = (i: Interval, d = 3) => `${i.value >= 0 ? '+' : ''}${f(i.value, d)} [${i.lo >= 0 ? '+' : ''}${f(i.lo, d)}, ${i.hi >= 0 ? '+' : ''}${f(i.hi, d)}]`;
const mb = (bytes: number) => `${(bytes / 2 ** 20).toFixed(1)} MB`;

function indexTable(b: IndexBench): string {
  const lines = [
    `**${b.label}**: ${b.n.toLocaleString('en-US')} vectors, ${b.dim} dimensions, ${b.queries} queries.`,
    '',
    `- Build: ${f(b.buildSeconds, 1)} s. Graph: ${mb(b.indexBytes)} on top of ${mb(b.vectorBytes)} of vectors.`,
    `- Mean layer-0 degree ${f(b.meanDegree0, 1)}, top layer ${b.maxLevel}.`,
    `- Exact scan (the oracle): p50 ${f(b.exactP50Ms, 2)} ms, p95 ${f(b.exactP95Ms, 2)} ms.`,
    '',
    '| ef | recall@10 vs exact | p50 ms | p95 ms | queries/s | speed-up vs exact |',
    '|---:|---:|---:|---:|---:|---:|',
  ];
  for (const s of b.sweep) {
    lines.push(`| ${s.ef} | ${f(s.recall10, 4)} | ${f(s.p50Ms, 3)} | ${f(s.p95Ms, 3)} | ${Math.round(s.qps).toLocaleString('en-US')} | ${f(b.exactP50Ms / s.p50Ms, 1)}x |`);
  }
  return lines.join('\n');
}

function systemTable(systems: SystemScore[]): string {
  const lines = ['| system | project-precision@10 | MRR | precision lift over recency |', '|---|---:|---:|---:|'];
  for (const s of systems) {
    lines.push(`| ${s.name} | ${ci(s.precision)} | ${ci(s.mrr)} | ${s.liftOverRecency ? signed(s.liftOverRecency.precision) : 'baseline'} |`);
  }
  return lines.join('\n');
}

/** Rankings derived from hybrid after the first results were seen. Reported on their own, as exploratory. */
const DIVERSIFIED = new Set(['hybrid+fold', 'hybrid+spread']);

function diversitySection(ev: EvalReport): string[] {
  const d = ev.diversity!;
  const score = (name: string) => ev.systems.find((x) => x.name === name);
  const rep = (name: string) => d.systems.find((x) => x.name === name);
  const hybrid = score('hybrid');
  const lines = [
    '### Folding near-copies (exploratory)',
    '',
    'A history is full of near-copies: the same explanation given in three sessions, the same error pasted twice. A ranker scores each on its own, so they arrive together and push everything else off the page. Two rankings derived from hybrid deal with that. They were added after the results above were first seen, so read this part as exploratory.',
    '',
    `- **fold** skips a result that is a near-copy of one already shown and reorders nothing else. "Near-copy" means a similarity of ${f(d.nearCopy)} or more, a threshold set from the corpus itself (five robust spreads above a typical unrelated pair), not picked by hand.`,
    `- **spread** also trades rank for variety (maximal marginal relevance, strength ${d.spreadLambda}, fixed before it was run on this corpus).`,
    '',
    '| ranking | project-precision@10 | MRR | mean similarity among the top 10 | near-copy pairs in the top 10 (of 45) | of the plain top 10, replaced |',
    '|---|---:|---:|---:|---:|---:|',
  ];
  for (const name of ['hybrid', 'hybrid+fold', 'hybrid+spread']) {
    const sc = score(name);
    const r = rep(name);
    if (!sc || !r) continue;
    lines.push(`| ${name} | ${ci(sc.precision)} | ${ci(sc.mrr)} | ${f(r.meanSim)} | ${r.dupPairs.toFixed(2)} | ${name === 'hybrid' ? '' : r.folded.toFixed(2)} |`);
  }
  const fold = score('hybrid+fold');
  const foldRep = rep('hybrid+fold');
  const plainRep = rep('hybrid');
  if (hybrid && fold && foldRep && plainRep) {
    const dp = fold.precision.value - hybrid.precision.value;
    const dm = fold.mrr.value - hybrid.mrr.value;
    lines.push(
      '',
      `**What it costs and buys.** Folding changes project-precision@10 by ${dp >= 0 ? '+' : ''}${f(dp)} and MRR by ${dm >= 0 ? '+' : ''}${f(dm)} against plain hybrid (point estimates on the same queries), and takes near-copy pairs in the top 10 from ${plainRep.dupPairs.toFixed(2)} to ${foldRep.dupPairs.toFixed(2)} per query. The same-project label can only price diversity, never reward it: a different chunk from the same project scores exactly what the copy it replaced did. So this table shows what folding costs on this measure and how much repetition it removes. It does not show that the result list is more useful.`,
    );
  }
  lines.push('', 'Folding is on by default in the viewer, the command line and the MCP tool. Every other number in this file is from the plain ranking.', '');
  return lines;
}

/** One sentence on whether the hybrid-versus-recency verdict holds across ways of averaging. */
function stability(lifts: Interval[], verdicts: Set<string>): string {
  if (verdicts.size === 1) return `Hybrid ${[...verdicts][0]} recency under all three.`;
  const clear = lifts.filter((l) => l.lo > 0 || l.hi < 0).length;
  const side = lifts.every((l) => l.value < 0) ? 'below' : lifts.every((l) => l.value > 0) ? 'above' : null;
  if (side) {
    return `Hybrid's estimate is ${side} recency in all ${lifts.length} views, but the gap is outside the noise in only ${clear} of them. Read that as "probably ${side === 'below' ? 'behind' : 'ahead'}", not as settled.`;
  }
  return 'The direction of the difference changes between these views, so treat the comparison as unsettled.';
}

/** A plain-words verdict on whether a lift is distinguishable from zero. */
function verdict(i: Interval): string {
  if (i.lo > 0) return 'beats';
  if (i.hi < 0) return 'loses to';
  return 'is not distinguishable from';
}

const pct = (x: number) => (Number.isFinite(x) ? `${(x * 100).toFixed(0)}%` : 'n/a');

const ARM_CONTEXT: Record<Arm, string> = {
  none: 'nothing',
  memory: 'search over earlier sessions only',
  recency: 'the most recent earlier chunks, question ignored',
  oracle: 'the passage the question was written from',
  instore: 'search over the whole store',
};

/** Whether memory-versus-nothing reads the same under every way of averaging, and which views differ. */
function abStability(views: readonly View[]): string {
  const side = (l: Interval) => (l.lo > 0 ? 'above' : l.hi < 0 ? 'below' : 'within');
  const sides = new Set(views.map((v) => side(v.lift)));
  if (sides.size === 1) {
    const s = [...sides][0]!;
    return s === 'within' ? `The interval includes zero under all ${views.length}.` : `The interval lies ${s} zero under all ${views.length}.`;
  }
  const clear = views.filter((v) => side(v.lift) !== 'within');
  const sign = views.every((v) => v.lift.value > 0) ? 'above' : views.every((v) => v.lift.value < 0) ? 'below' : null;
  if (sign) return `The estimate is ${sign} zero in all ${views.length} views, but the interval excludes zero in only ${clear.length} of them: ${clear.map((v) => v.label).join('; ')}. The verdict stays with the preregistered headline.`;
  return 'The direction of the difference changes between these views, so treat the comparison as unsettled.';
}

/** Section 3: the agent A/B. Numbers only; the questions and answers never leave the private store. */
function abSection(ab: AbReport): string[] {
  const p = ab.params;
  const o = ab.outcome;
  const label = o === 'judged' ? 'judged correctness' : 'the contains-reference outcome';
  const arm = (name: Arm) => ab.arms.find((a) => a.arm === name)!;
  const contrast = (a: Arm, b: Arm) => ab.contrasts.find((c) => c.a === a && c.b === b)!;
  const control = contrast('oracle', 'none')[o];
  const vsRecency = contrast('memory', 'recency')[o];
  const instore = contrast('instore', 'none')[o];
  const other = o === 'judged' ? 'contains' : 'judged';
  const rejected = Object.entries(ab.rejected).sort((x, y) => y[1] - x[1]);
  const kept = ab.passagesTried ? ab.questions / ab.passagesTried : Number.NaN;

  const out: string[] = [
    "## 3. Does recalled context make an agent's answers better?",
    '',
    'The hypothesis, arms, metric, sample and decision rule were committed in [docs/memory-ab-prereg.md](docs/memory-ab-prereg.md) before the harness was written or run.',
    '',
    "One session is held out at a time. A local model writes questions from that session's own text; a question is kept only if its answer appears word for word in the passage and not in the question. The same model then answers each question once per arm, with nothing but the context changed. Recalled context is formatted as the MCP `search` tool returns it. Answers from all arms are pooled and shuffled, and the judge sees only the question, the reference answer and the answer to grade.",
    '',
    `${ab.questions} questions from ${ab.sessions} sessions across ${ab.projects} projects, over a corpus of ${ab.corpus.toLocaleString('en-US')} chunks. ${ab.eligibleSessions} sessions were eligible${ab.droppedSessions ? ` and ${ab.droppedSessions} produced no usable question` : ''}; ${ab.passagesTried} passages were tried and ${pct(kept)} gave a kept question${rejected.length ? ` (most common rejection: ${rejected[0]![0]}, ${rejected[0]![1]})` : ''}. Writing, answering and judging all used \`${ab.model}\`${ab.modelDigest ? ` (${ab.modelDigest})` : ''} on this machine at temperature 0. Each session counts once. Brackets are 95% intervals from a bootstrap that resamples whole sessions.`,
    '',
    '| arm | context | judged correct | contains reference | wrong, not an abstention |',
    '|---|---|---:|---:|---:|',
    ...ab.arms.map((a) => `| ${a.arm} | ${ARM_CONTEXT[a.arm]} | ${ci(a.judged)} | ${ci(a.contains)} | ${ci(a.wrongAnswered)} |`),
    '',
    '- **judged correct** is the model judge. **contains reference** needs no model: the reference answer appears in the answer.',
    `- **wrong, not an abstention** is the share of answers that were wrong (by ${label}) without saying "I do not know": the cost of context that misleads.`,
    '',
    '| difference | judged correct | contains reference |',
    '|---|---:|---:|',
    ...ab.contrasts.map((c) => `| ${c.a} - ${c.b}${c.a === 'memory' && c.b === 'none' ? ' (primary)' : ''} | ${signed(c.judged)} | ${signed(c.contains)} |`),
    '',
    '### Can the grading be trusted?',
    '',
    `The judge was checked on ${ab.calibration.cases} invented answers with known labels, none of them used while writing its prompt.`,
    '',
    '| grader | sensitivity | specificity | balanced accuracy |',
    '|---|---:|---:|---:|',
    `| judge (\`${ab.model}\`) | ${f(ab.calibration.judge.sensitivity)} | ${f(ab.calibration.judge.specificity)} | ${f(ab.calibration.judge.balanced)} |`,
    `| contains reference (no model) | ${f(ab.calibration.contains.sensitivity)} | ${f(ab.calibration.contains.specificity)} | ${f(ab.calibration.contains.balanced)} |`,
    '',
    ab.gates.judge
      ? `The bar set in advance was ${f(p.judgeGate, 2)} balanced accuracy. The judge cleared it, so the verdict rests on judged correctness.`
      : `The bar set in advance was ${f(p.judgeGate, 2)} balanced accuracy. The judge fell short of it, so, as preregistered, the verdict rests on the contains-reference outcome and the judged numbers are shown for information only.`,
    `On the ${ab.agreement.answers.toLocaleString('en-US')} real answers the two graders agreed on ${pct((ab.agreement.both + ab.agreement.neither) / ab.agreement.answers)}: ${ab.agreement.judgeOnly} were correct for the judge only and ${ab.agreement.containsOnly} for the string check only.`,
    '',
  ];

  const primary = `${signed(ab.primary)} on ${label}`;
  if (ab.verdict === 'positive') {
    out.push(`**Verdict: positive.** Recalled context made answers measurably better. Memory beat no context by ${primary}.`);
  } else if (ab.verdict === 'negative') {
    out.push(`**Verdict: negative.** Recalled context made answers measurably worse. Memory against no context: ${primary}.`);
  } else if (ab.verdict === 'null') {
    out.push(
      `**Verdict: null.** No measurable difference between answering with recalled context and answering without it: ${primary}. ${
        ab.rulesOutUsefulGain ? `The interval rules out a gain of ${f(p.usefulGain, 2)} or more, the smallest gain set in advance as worth having.` : `The interval is too wide to rule out a gain of ${f(p.usefulGain, 2)} or more, the smallest gain set in advance as worth having, so this is an absence of evidence, not evidence of absence.`
      }`,
    );
  } else {
    const why: string[] = [];
    if (!ab.gates.positiveControl) why.push(`the model was not clearly better with the answer in front of it (oracle - none = ${signed(control)}; the bar was an interval entirely above +${f(p.controlGate, 2)})`);
    if (!ab.gates.yield) why.push(`only ${ab.sessions} sessions produced a question (the bar was ${p.minSessions})`);
    out.push(`**Verdict: not interpretable.** A check set in advance failed: ${why.join(', and ')}. Memory against no context came out at ${primary}, but no conclusion about memory is drawn from it.`);
  }
  const otherLift = contrast('memory', 'none')[other];
  const sideOf = (l: Interval) => (l.lo > 0 ? 'above' : l.hi < 0 ? 'below' : 'across');
  if (ab.verdict !== 'not interpretable' && sideOf(otherLift) !== sideOf(ab.primary)) {
    const words = { above: 'lies above zero', below: 'lies below zero', across: 'includes zero' };
    out.push(
      '',
      `The two outcomes do not agree on that. On ${other === 'judged' ? 'judged correctness' : 'the contains-reference outcome'} the difference is ${signed(otherLift)}, an interval that ${words[sideOf(otherLift)]}, where the headline's ${words[sideOf(ab.primary)]}. The verdict follows the outcome chosen in advance; the other is a reason not to read it as more settled than it is.`,
    );
  }
  out.push('');

  const sameRows = ab.robustness.length >= 3 && ab.robustness.slice(1, 3).every((v) => v.questions === ab.questions && v.sessions === ab.sessions);
  const none = arm('none')[o];
  const oracle = arm('oracle')[o];
  out.push(
    `For scale: with the source passage in hand the model was right ${pct(oracle.value)} of the time, against ${pct(none.value)} with nothing (oracle - none = ${signed(control)}).${
      control.value > 0 && ab.primary.value > 0 ? ` Memory recovered ${pct(ab.primary.value / control.value)} of that gap.` : ''
    }`,
    '',
    '### How stable is that verdict?',
    '',
    `The headline counts each session once. Section 2 showed that a verdict can turn on how groups are weighted, so the same difference is given here under five ways of averaging, two of them with a floor on group size. The list was fixed while the run was in progress, before any score was read. Sessions are what is resampled in every row; in the two project rows they are resampled within their project.`,
    '',
    '| averaging | sessions | questions | projects | memory - none |',
    '|---|---:|---:|---:|---:|',
    ...ab.robustness.map((v) => `| ${v.label} | ${v.sessions} | ${v.questions} | ${v.projects} | ${signed(v.lift)} |`),
    '',
    abStability(ab.robustness),
    ...(sameRows ? [`Every session produced all ${p.questionsPerSession} questions, so the first three rows are the same calculation.`] : []),
    'The every-project row gives a project with one session the same weight as a project with twenty, and a one-session project adds no spread to its interval. That is why the preregistration did not use it as the headline, and why the last row sets a floor.',
    '',
    'Secondary results, reported as planned. No verdict rests on them.',
    '',
    `- **Against recency.** Memory ${verdict(vsRecency)} the most recent earlier chunks (${signed(vsRecency)}).`,
    `- **When the answer is in the store.** Searching the whole store instead of only earlier sessions gives instore - none = ${signed(instore)}. This is biased upward: each question was written from the stored passage and shares its wording.`,
    `- **On the other outcome.** Memory - none is ${signed(otherLift)} on ${other === 'judged' ? 'judged correctness' : 'contains reference'}.`,
    `- **Context that misleads.** With nothing, the model said "I do not know" to ${pct(arm('none').abstained)} of questions and gave a wrong answer to ${pct(arm('none').wrongAnswered.value)} (${ci(arm('none').wrongAnswered)}). With recalled context it abstained on ${pct(arm('memory').abstained)} and gave a wrong answer to ${pct(arm('memory').wrongAnswered.value)} (${ci(arm('memory').wrongAnswered)}). Recalled text that does not hold the answer still gets answered from.`,
  );
  const { present, absent, share } = ab.answerInMemory;
  out.push(
    `- **Was the answer there to find?** For ${pct(share)} of questions the reference answer appears word for word somewhere in the earlier chunks memory was allowed to search. ${present ? `Where it does (${present.questions} questions), memory - none is ${signed(present.lift)}. ` : ''}${absent ? `Where it does not (${absent.questions} questions), it is ${signed(absent.lift)}.` : ''}`,
    '',
  );
  return out;
}

/** Numbers only: how much of what was typed is a repeat. The sentences themselves stay private. */
function repeatsSection(r: RepeatsReport): string[] {
  const n = (x: number) => x.toLocaleString('en-US');
  const out = [
    '## 4. How much do you repeat yourself?',
    '',
    'Every prompt typed by hand is split into sentences; pasted code, logs and contentless go-aheads are dropped. Each sentence is embedded locally and every pair is compared exactly. Two sentences count as the same thing when their similarity is far above that of a typical unrelated pair, and a group counts as a habit when it shows up in several separate sessions on more than one day.',
    '',
    `- ${n(r.sentences)} sentences from ${n(r.prompts)} prompts. ${n(r.templatedPrompts)} prompts were left out as program-written templates, and ${n(r.sameDayBursts)} groups as same-day bursts.`,
    `- Threshold: similarity ${f(r.threshold.tau)}. A typical cross-session pair scores ${f(r.threshold.median)} with a robust spread of ${f(r.threshold.sigma)}, so the threshold sits ${f(r.threshold.z, 1)} spreads above it.`,
    `- Cross-session pairs above the threshold: **${n(r.pairs.above)}** out of ${n(r.pairs.crossSession)} compared. If similarities of unrelated sentences followed a normal curve, about ${f(r.pairs.expectedByChance, 1)} would be expected by chance.`,
    `- **${n(r.groups)} things were said in ${r.minSessions} or more sessions on at least two days**: ${r.byKind.instruction} instructions, ${r.byKind.task} recurring task openers, ${r.byKind.question} questions and ${r.byKind.statement} other statements.`,
    `- Those groups hold ${n(r.sentencesInGroups)} sentences, ${f(r.share * 100, 1)}% of everything typed.`,
    '',
    '| sessions a group spans | groups |',
    '|---|---:|',
    ...r.spread.map((b, i) => `| ${i === r.spread.length - 1 ? `${b.sessions} or more` : b.sessions === (r.spread[i + 1]?.sessions ?? 0) - 1 ? String(b.sessions) : `${b.sessions} to ${(r.spread[i + 1]?.sessions ?? 0) - 1}`} | ${b.groups} |`),
    '',
    'What this does not show: that a repeated instruction was needed each time, or that writing it down once as a standing rule would have saved the repetition. The kind labels (instruction, task, question) come from simple word patterns, not from reading for meaning. The chance figure assumes a normal curve, which real similarities only roughly follow, so read it as "far more than chance", not as an exact rate.',
    '',
  ];
  return out;
}

export function renderResults(bench: BenchReport | null, ev: EvalReport | null, generated: string, ab: AbReport | null = null, repeats: RepeatsReport | null = null): string {
  const out: string[] = [
    '# LOCI results',
    '',
    `Generated ${generated} by \`loci bench\`, \`loci eval\`${ab ? ', `loci ab`' : ''}${repeats ? ', `loci repeats`' : ''} and \`loci report\`. Every number below is measured on this machine; re-run the commands to reproduce.`,
    '',
    'Nothing here contains conversation text. Projects are labelled by size (P1 is the largest), never by name.',
    '',
  ];

  if (bench) {
    const at = (b: IndexBench, ef: number) => b.sweep.find((s) => s.ef === ef)!;
    out.push(
      '## 1. Is the index correct, and how fast is it?',
      '',
      `Machine: ${bench.machine.cpu}, ${bench.machine.cores} cores, ${bench.machine.memoryGb} GB, Node ${bench.machine.node}, ${bench.machine.platform}. Single thread, no SIMD, no native code.`,
      '',
      `HNSW parameters: M = ${bench.params.M}, M0 = ${bench.params.M0}, efConstruction = ${bench.params.efConstruction}. Recall is the share of the true 10 nearest neighbours (found by scanning every vector) that the graph returns.`,
      '',
      indexTable(bench.real),
      '',
      `On the real corpus at ef = 64 the graph returns ${f(at(bench.real, 64).recall10 * 100, 1)}% of the true top 10 in ${f(at(bench.real, 64).p50Ms, 2)} ms.`,
      '',
    );
    if (bench.synthetic) {
      out.push(
        'The real corpus is small, so the same code is also measured on a larger synthetic set. These vectors are random clusters, not text: they show how the index scales, not how it behaves on language.',
        '',
        indexTable(bench.synthetic),
        '',
      );
    }
    out.push(
      '### Lexical index against SQLite FTS5',
      '',
      `The from-scratch BM25 uses the same formula and tokeniser as SQLite FTS5, so FTS5 serves as its oracle. Over ${bench.bm25.ftsQueries} real prompts, ${f(bench.bm25.ftsTop10Agreement * 100, 2)}% of its top-10 results match FTS5's top 10 (ties at the cut-off counted as matches).`,
      '',
      `- ${bench.bm25.docs.toLocaleString('en-US')} documents, ${bench.bm25.vocabulary.toLocaleString('en-US')} distinct terms, built in ${f(bench.bm25.buildMs, 0)} ms.`,
      `- Query latency: p50 ${f(bench.bm25.p50Ms, 3)} ms, p95 ${f(bench.bm25.p95Ms, 3)} ms.`,
      `- Process memory with everything loaded: ${bench.rssMb} MB.`,
      '',
    );
  }

  if (ev) {
    const get = (name: string) => ev.systems.find((s) => s.name === name);
    const hybrid = get('hybrid')!;
    const fused = get('hybrid+recency');
    const lift = hybrid.liftOverRecency!;
    const chance = get('random')!;
    out.push(
      '## 2. Does retrieval find the right earlier work?',
      '',
      'No manual labels. Each query is a real prompt; the engine may only return chunks written **before** that prompt and never from the same session. A result counts as relevant when it comes from the same project as the prompt.',
      '',
      `${ev.queries} queries from ${ev.sessions} sessions across ${ev.projects} projects, over a corpus of ${ev.corpus.toLocaleString('en-US')} chunks (median ${ev.medianEligible.toLocaleString('en-US')} eligible candidates per query).`,
      '',
      `Scores are macro-averaged by project, over the ${ev.headlineProjects} projects with at least ${ev.minProjectQueries} queries (${ev.headlineQueries} queries). Brackets are 95% intervals from a bootstrap that resamples whole sessions within each project.`,
      '',
      systemTable(ev.systems.filter((x) => !DIVERSIFIED.has(x.name))),
      '',
      '- **random** is the closed-form expectation of a random ranking: what the corpus gives away for free.',
      '- **majority** always answers with the largest project.',
      '- **recency** returns the 10 most recent earlier chunks and ignores the prompt entirely. It is the baseline to beat, because the last thing you worked on is usually the same project.',
      '- **hybrid** fuses the dense and BM25 rankings by reciprocal rank.',
      ...(fused ? ['- **hybrid+recency** adds the recency list as a third ranking in the same fusion, with no tuned weights.'] : []),
      '',
      `**Verdict.** Hybrid retrieval is well above chance (${f(hybrid.precision.value)} against ${f(chance.precision.value)}) but ${verdict(lift.precision)} the recency baseline on project-precision@10 (${signed(lift.precision)}). On MRR it ${verdict(lift.mrr)} recency (${signed(lift.mrr)}).`,
      '',
    );
    if (ev.robustness?.length) {
      const cell = (sys: { precision: number; lift: Interval | null }) => (sys.lift ? `${f(sys.precision)} (${signed(sys.lift)})` : f(sys.precision));
      const verdicts = new Set(ev.robustness.map((v) => verdict(v.systems.find((x) => x.name === 'hybrid')!.lift!)));
      out.push(
        '### How stable is that verdict?',
        '',
        `The headline is a mean over projects. Projects with fewer than ${ev.minProjectQueries} queries are left out of it, because otherwise a project with one query weighs as much as a project with five hundred. That floor was added after exactly this happened: between two runs a few hours apart, one new single-query project moved the every-project average by about 0.03 and changed its verdict, while no other project's score moved. The same comparison under each way of averaging, with the lift over recency in brackets:`,
        '',
        '| averaging | queries | projects | recency | hybrid | hybrid+recency |',
        '|---|---:|---:|---:|---:|---:|',
        ...ev.robustness.map((v) => {
          const get = (name: string) => v.systems.find((x) => x.name === name);
          const fusedCell = get('hybrid+recency');
          return `| ${v.label} | ${v.queries} | ${v.projects} | ${f(get('recency')!.precision)} | ${cell(get('hybrid')!)} | ${fusedCell ? cell(fusedCell) : 'n/a'} |`;
        }),
        '',
        stability(ev.robustness.map((v) => v.systems.find((x) => x.name === 'hybrid')!.lift!), verdicts),
        '',
      );
    }
    if (fused) {
      const fl = fused.liftOverRecency!;
      out.push(
        `Adding recency as a third signal: hybrid+recency ${verdict(fl.precision)} recency on precision (${signed(fl.precision)}) and ${verdict(fl.mrr)} it on MRR (${signed(fl.mrr)}).`,
        '',
        'The recency-fused ranker was added after the first run showed plain hybrid losing to recency, and it is scored on the same queries. It has no tuned parameters, but it is still a post-hoc change: treat its numbers as exploratory until the eval is re-run on sessions recorded after this date.',
        '',
      );
    }
    if (ev.masked) {
      const mh = ev.masked.find((s) => s.name.startsWith('hybrid'))!;
      const delta = mh.precision.value - hybrid.precision.value;
      out.push(
        '### Ablation: project names and paths hidden',
        '',
        'A ranker could score well just by matching folder names. Here every project name and file path is replaced with a placeholder in both the prompts and the corpus, then the corpus is re-embedded and the eval repeated.',
        '',
        systemTable(ev.masked),
        '',
        Math.abs(delta) < 0.01
          ? `Hiding names and paths moves hybrid precision by ${Math.abs(delta) < 0.0005 ? '0.000' : `${delta > 0 ? '+' : ''}${f(delta)}`} (${f(hybrid.precision.value)} to ${f(mh.precision.value)}): the result does not depend on those surface cues.`
          : `Hiding names and paths moves hybrid precision by ${delta >= 0 ? '+' : ''}${f(delta)} (${f(hybrid.precision.value)} to ${f(mh.precision.value)}): that much of the result depended on those surface cues.`,
        '',
      );
    }
    out.push(
      '### By project',
      '',
      'Precision@10 per project. Recency wins where a project was worked in one concentrated burst; retrieval wins where work on a project was spread out or interleaved with others.',
      '',
      '| project | queries | random | recency | hybrid | hybrid+recency |',
      '|---|---:|---:|---:|---:|---:|',
    );
    for (const p of ev.perProject) out.push(`| ${p.label} | ${p.queries} | ${f(p.random)} | ${f(p.recency)} | ${f(p.hybrid)} | ${p.fused === undefined ? 'n/a' : f(p.fused)} |`);
    out.push('');
    if (ev.knownItem) {
      out.push(
        '### Known-item test: title to session',
        '',
        `Each session has a short title that is never indexed. Using only the title as the query, does a chunk of that session come back? ${ev.knownItem.queries} sessions, searching the whole corpus.`,
        '',
        '| system | hit@1 | hit@10 | MRR |',
        '|---|---:|---:|---:|',
        ...ev.knownItem.systems.map((s) => `| ${s.name} | ${f(s.hit1)} | ${f(s.hit10)} | ${f(s.mrr)} |`),
        `| random (expected) | | ${f(ev.knownItem.randomHit10)} | |`,
        '',
      );
    }
  }

  if (ev?.diversity) out.push(...diversitySection(ev));

  if (ab) out.push(...abSection(ab));

  if (ev || ab) {
    out.push('## What these numbers do not show', '');
    if (ev) {
      out.push(
        '- "Same project" is a coarse label. It shows the engine finds the right neighbourhood, not that it finds the passage that answers the prompt.',
        "- One person, one corpus, one embedding model. Nothing here says how it behaves on anyone else's history.",
      );
    }
    if (ab) {
      out.push(
        `- The A/B used one small local model (\`${ab.model}\`) to write the questions, answer them and judge the answers. A stronger model might use the same context better, or need it less.`,
        '- Nobody read the generated questions as part of the run. Their quality shows only indirectly, in how many passed the checks and in how far the oracle arm sits above the no-context arm. The owner can audit them in the private store.',
        '- The no-context arm had no tools and no repository. A real agent without memory could still read the code, so this measures what recalled context adds to a model that has nothing else.',
        '- Holding a session out measures whether earlier work carries over to a later session. It is a harder test than recalling something already in the store, which only the upward-biased instore arm touches.',
      );
    } else {
      out.push("- Whether recalled context makes an agent's answers better is a separate experiment that has not been run.");
    }
    out.push('');
  }
  // Placed after the caveats of the sections above: it is a separate measurement with its own.
  if (repeats && repeats.sentences > 0) out.push(...repeatsSection(repeats));
  return out.join('\n');
}
