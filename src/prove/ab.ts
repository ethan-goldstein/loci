import { createHash } from 'node:crypto';
import fs from 'node:fs';
import type { Embedder } from '../embed/ollama.ts';
import type { Engine, EngineHit } from '../engine.ts';
import { DEFAULT_K, SNIPPET, formatHits, wrap } from '../mcp/tools.ts';
import { CALIBRATION, DEV_ITEMS, calibrationCases } from './abcal.ts';
import { retrying } from './abmodel.ts';
import type { AbModel, JudgeItem } from './abmodel.ts';
import { clusterBootstrap, macroMean } from './bootstrap.ts';
import type { Interval } from './bootstrap.ts';
import { MIN_PROJECT_QUERIES } from './eval.ts';
import { rng } from './synth.ts';

/**
 * Agent A/B: does recalled context make a model's answers better than no context?
 *
 * The design, metric, sample and decision rule were fixed in docs/memory-ab-prereg.md before this
 * file existed. In short:
 *
 *   unit       one held-out session at a time
 *   question   written by a local model from a passage of that session, kept only if it passes
 *              deterministic checks (answer verbatim in the passage, question does not contain it)
 *   arms       the same question answered by the same model with different context:
 *                none     nothing
 *                memory   hybrid search, only chunks dated before the session, session excluded
 *                recency  the most recent chunks under the same restriction, ignoring the question
 *                oracle   the passage itself (positive control and ceiling)
 *                instore  hybrid search over the whole store (the answer is in the store)
 *   outcome    judged correct or not by a model that sees the question, the reference and the
 *              answer and nothing else; plus a model-free check that the answer contains the reference
 *   statistic  mean over sessions of (memory - none), with a session-clustered bootstrap interval
 *
 * Nothing in the report this returns contains conversation text. The questions and answers
 * themselves go to a separate structure that the CLI writes only into the private store.
 */

export const ARMS = ['none', 'memory', 'recency', 'oracle', 'instore'] as const;
export type Arm = (typeof ARMS)[number];
export type Outcome = 'judged' | 'contains';
export type Verdict = 'positive' | 'null' | 'negative' | 'not interpretable';

export interface AbParams {
  /** Search results handed to the model, and snippet length: the MCP server's defaults. */
  k: number;
  snippet: number;
  questionsPerSession: number;
  maxPassages: number;
  minTitle: number;
  minChunks: number;
  minPassages: number;
  minPassageChars: number;
  minEarlier: number;
  answerChars: number;
  seed: number;
  /** Smallest gain in accuracy treated as worth having. */
  usefulGain: number;
  /** Judge balanced accuracy below this moves the verdict to the model-free outcome. */
  judgeGate: number;
  /** The interval for oracle - none must lie entirely above this. */
  controlGate: number;
  minSessions: number;
}

/** The preregistered values. Tests override some of them to fit the small synthetic store. */
export const AB: AbParams = {
  k: DEFAULT_K,
  snippet: SNIPPET,
  questionsPerSession: 4,
  maxPassages: 12,
  minTitle: 8,
  minChunks: 8,
  minPassages: 4,
  minPassageChars: 400,
  minEarlier: 50,
  answerChars: 300,
  seed: 20261116,
  usefulGain: 0.1,
  judgeGate: 0.9,
  controlGate: 0.2,
  minSessions: 30,
};

// ---------------------------------------------------------------------------------------------
// Text checks. All deterministic: no model decides whether a question is kept or an answer
// "contains" its reference.

/** Lower-cased, compatibility-normalised, whitespace collapsed. */
export function norm(text: string): string {
  return text.normalize('NFKC').toLowerCase().replace(/\s+/g, ' ').trim();
}

export function containsReference(reference: string, answer: string): boolean {
  const r = norm(reference);
  return r.length > 0 && norm(answer).includes(r);
}

export function isAbstention(answer: string): boolean {
  return /\b(?:do(?:es)? not know|don['’]?t know|cannot tell|can['’]?t tell|not sure)\b/i.test(answer);
}

const POINTS_AT_PASSAGE = /\b(?:the passage|the text|the excerpt|the conversation|the assistant|the speaker|the author|this session|the above)\b/i;

export const questionFor = (project: string, question: string): string => `In the project ${project}: ${question}`;

/**
 * Why a written question is rejected, or null when it is kept. `shown` is the question exactly as
 * every arm will see it (with its project prefix), so an answer hidden in the prefix is caught too.
 */
export function rejectReason(question: string, answer: string, passage: string, shown: string, seen: ReadonlySet<string>): string | null {
  const a = norm(answer);
  if (a.length < 3 || a.length > 80 || a.split(' ').length > 10) return 'answer length';
  if (!/\p{L}{3}|\p{N}/u.test(a)) return 'answer has no word or number';
  if (!norm(passage).includes(a)) return 'answer not in passage';
  const q = question.trim();
  if (q.length < 20 || q.length > 300) return 'question length';
  if (!q.endsWith('?')) return 'question is not a question';
  if (norm(shown).includes(a)) return 'question contains answer';
  if (POINTS_AT_PASSAGE.test(q)) return 'question points at the passage';
  if (seen.has(norm(q))) return 'duplicate question';
  return null;
}

// ---------------------------------------------------------------------------------------------
// Sessions and contexts.

export interface AbSession {
  pk: number;
  project: number;
  /** Time of the session's first chunk: nothing dated at or after this may be recalled. */
  start: number;
  /** Candidate passages: assistant chunks long enough to hold a fact worth asking about. */
  passages: number[];
  earlier: number;
}

export function selectSessions(engine: Engine, p: AbParams = AB): AbSession[] {
  const members = new Map<number, number[]>();
  for (let i = 0; i < engine.n; i++) {
    const list = members.get(engine.sessionPk[i]!) ?? [];
    list.push(i);
    members.set(engine.sessionPk[i]!, list);
  }
  const out: AbSession[] = [];
  for (const [pk, nodes] of members) {
    const row = engine.sessions.get(pk);
    if (!row?.title || row.title.length < p.minTitle || nodes.length < p.minChunks) continue;
    const passages = nodes.filter((i) => engine.role[i] === 1 && engine.texts[i]!.length >= p.minPassageChars);
    if (passages.length < p.minPassages) continue;
    let start = Infinity;
    for (const i of nodes) start = Math.min(start, engine.ts[i]!);
    let earlier = 0;
    for (let i = 0; i < engine.n; i++) if (engine.ts[i]! < start && engine.sessionPk[i] !== pk) earlier++;
    if (earlier < p.minEarlier) continue;
    out.push({ pk, project: row.projectId, start, passages, earlier });
  }
  return out.sort((a, b) => a.start - b.start || a.pk - b.pk);
}

function shuffled<T>(items: readonly T[], seed: number): T[] {
  const out = [...items];
  const rand = rng(seed);
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [out[i], out[j]] = [out[j]!, out[i]!];
  }
  return out;
}

const asHit = (node: number): EngineHit => ({ node, chunkId: node + 1, score: 0, sim: null, denseRank: 0, bm25Rank: 0 });

/**
 * The context one arm gives the answering model, and the nodes it was built from. The wrapper and
 * the line format are the MCP search tool's own. The held-out session's title is never shown.
 */
export function contextFor(engine: Engine, arm: Arm, s: AbSession, node: number, shown: string, qvec: Float32Array, p: AbParams = AB): { text: string | null; nodes: number[] } {
  if (arm === 'none') return { text: null, nodes: [] };
  const guard = { before: s.start, excludeSessionPk: s.pk };
  let hits: EngineHit[];
  let snippet = p.snippet;
  if (arm === 'memory') hits = engine.searchWith(shown, qvec, { k: p.k, mode: 'hybrid', ...guard }).hits;
  else if (arm === 'instore') hits = engine.searchWith(shown, qvec, { k: p.k, mode: 'hybrid' }).hits;
  else if (arm === 'recency') {
    const allow = engine.allow(guard)!;
    hits = [];
    for (let i = 0; i < engine.n && hits.length < p.k; i++) {
      const cand = engine.newestFirst[i]!;
      if (allow(cand)) hits.push(asHit(cand));
    }
  } else {
    hits = [asHit(node)];
    snippet = Number.MAX_SAFE_INTEGER; // the whole passage: a cut could remove the answer
  }
  const body = hits.length === 0 ? 'No matches.' : formatHits(engine, hits, { snippet, hideTitleOfSession: s.pk });
  return { text: wrap(body), nodes: hits.map((h) => h.node) };
}

// ---------------------------------------------------------------------------------------------
// A resumable cache. A full run is about an hour of local generation; a crash should not cost it.

export interface AbCache {
  get(key: string): unknown;
  set(key: string, value: unknown): void;
}

export function memoryCache(): AbCache {
  const m = new Map<string, unknown>();
  return { get: (k) => m.get(k), set: (k, v) => void m.set(k, v) };
}

/**
 * An append-only JSON-lines file. It holds generated questions and answers, which are derived from
 * private conversations, so it must live in the private store directory; it is created owner-only.
 * Lines written under a different fingerprint (other model, prompts, parameters or corpus) are ignored.
 */
export function fileCache(file: string, fingerprint: string): AbCache {
  const m = new Map<string, unknown>();
  if (fs.existsSync(file)) {
    for (const line of fs.readFileSync(file, 'utf8').split('\n')) {
      if (!line) continue;
      try {
        const row = JSON.parse(line) as { fp?: string; key?: string; value?: unknown };
        if (row.fp === fingerprint && typeof row.key === 'string') m.set(row.key, row.value);
      } catch {
        // A half-written last line from an interrupted run: skip it.
      }
    }
  } else {
    fs.writeFileSync(file, '', { mode: 0o600 });
  }
  return {
    get: (k) => m.get(k),
    set: (k, v) => {
      m.set(k, v);
      fs.appendFileSync(file, `${JSON.stringify({ fp: fingerprint, key: k, value: v })}\n`);
    },
  };
}

const sha = (text: string): string => createHash('sha256').update(text).digest('hex').slice(0, 16);

export function fingerprint(engine: Engine, model: AbModel, embedModel: string, p: AbParams): string {
  return sha(JSON.stringify([model.name, model.promptHash, embedModel, engine.n, p]));
}

// ---------------------------------------------------------------------------------------------
// Judge calibration on invented cases with known labels.

export interface Confusion {
  tp: number;
  tn: number;
  fp: number;
  fn: number;
  sensitivity: number;
  specificity: number;
  balanced: number;
}

export interface Calibration {
  cases: number;
  judge: Confusion;
  contains: Confusion;
}

function confusion(pairs: readonly { truth: boolean; got: boolean }[]): Confusion {
  let tp = 0;
  let tn = 0;
  let fp = 0;
  let fn = 0;
  for (const { truth, got } of pairs) {
    if (truth && got) tp++;
    else if (!truth && !got) tn++;
    else if (!truth && got) fp++;
    else fn++;
  }
  const sensitivity = tp + fn ? tp / (tp + fn) : Number.NaN;
  const specificity = tn + fp ? tn / (tn + fp) : Number.NaN;
  return { tp, tn, fp, fn, sensitivity, specificity, balanced: (sensitivity + specificity) / 2 };
}

/** Grades the held-out calibration cases with the judge and with the model-free check. */
export async function calibrate(model: AbModel, cache: AbCache = memoryCache(), onProgress?: (done: number, total: number) => void): Promise<Calibration> {
  const cases = calibrationCases(CALIBRATION.slice(DEV_ITEMS));
  const judged: { truth: boolean; got: boolean }[] = [];
  const contained: { truth: boolean; got: boolean }[] = [];
  for (const [i, c] of cases.entries()) {
    const item: JudgeItem = { question: c.question, reference: c.reference, answer: c.answer };
    judged.push({ truth: c.correct, got: await judgeOnce(model, cache, item) });
    contained.push({ truth: c.correct, got: containsReference(c.reference, c.answer) });
    onProgress?.(i + 1, cases.length);
  }
  return { cases: cases.length, judge: confusion(judged), contains: confusion(contained) };
}

const judgeKey = (item: JudgeItem): string => `j:${sha(JSON.stringify([item.question, item.reference, item.answer]))}`;

/** One verdict per distinct (question, reference, answer): identical answers can never be graded differently. */
async function judgeOnce(model: AbModel, cache: AbCache, item: JudgeItem): Promise<boolean> {
  const key = judgeKey(item);
  const hit = cache.get(key);
  if (typeof hit === 'boolean') return hit;
  const got = await model.judge(item);
  cache.set(key, got);
  return got;
}

// ---------------------------------------------------------------------------------------------
// The run.

/** One kept question with everything that happened to it. Holds text: private store only. */
export interface AbItem {
  sessionPk: number;
  project: number;
  node: number;
  question: string;
  reference: string;
  answers: Record<Arm, string>;
  contextNodes: Record<Arm, number[]>;
  judged: Record<Arm, boolean>;
}

/** The same question as numbers only: safe to print and to keep in the report. */
export interface AbRow {
  sessionPk: number;
  project: number;
  node: number;
  /** The reference answer appears verbatim in a chunk the memory arm was allowed to search. */
  answerInMemory: boolean;
  judged: Record<Arm, number>;
  contains: Record<Arm, number>;
  abstained: Record<Arm, number>;
}

export interface ArmScore {
  arm: Arm;
  judged: Interval;
  contains: Interval;
  /** Share of answers that were wrong by the outcome in use and were not an abstention. */
  wrongAnswered: Interval;
  abstained: number;
}

export interface Contrast {
  a: Arm;
  b: Arm;
  judged: Interval;
  contains: Interval;
}

export interface Split {
  questions: number;
  sessions: number;
  lift: Interval;
}

/** One way of averaging the primary contrast. */
export interface View {
  label: string;
  sessions: number;
  questions: number;
  projects: number;
  lift: Interval;
}

export interface AbMeta {
  model: string;
  modelDigest: string | null;
  promptHash: string;
  embedModel: string;
  params: AbParams;
  corpus: number;
  eligibleSessions: number;
  passagesTried: number;
  rejected: Record<string, number>;
}

export interface AbReport extends AbMeta {
  sessions: number;
  droppedSessions: number;
  projects: number;
  questions: number;
  calibration: Calibration;
  arms: ArmScore[];
  contrasts: Contrast[];
  /** Which outcome carries the verdict: the judge's, unless it failed calibration. */
  outcome: Outcome;
  gates: { judge: boolean; positiveControl: boolean; yield: boolean };
  /** memory - none on the outcome in use. */
  primary: Interval;
  verdict: Verdict;
  rulesOutUsefulGain: boolean;
  /** The primary contrast macro-averaged by project, with the project-stratified bootstrap. */
  projectMacro: Interval;
  answerInMemory: { share: number; present: Split | null; absent: Split | null };
  /**
   * The primary contrast under other ways of averaging, so a verdict that depends on how groups are
   * weighted shows up as such. Added after the retrieval eval's verdict was seen to flip on one
   * single-query project; fixed before any A/B result was read (see the preregistration).
   */
  robustness: View[];
  /** Judge against the model-free outcome over every real answer. */
  agreement: { answers: number; both: number; neither: number; judgeOnly: number; containsOnly: number };
  rows: AbRow[];
}

export interface AbOptions {
  embedder: Embedder;
  model: AbModel;
  modelDigest?: string | null;
  params?: Partial<AbParams>;
  cache?: AbCache;
  /** Waits between attempts at a failed embedding request. Defaults to the model client's schedule. */
  retryDelaysMs?: readonly number[];
  /** Use only the first N eligible sessions. For smoke runs; a real run uses all of them. */
  maxSessions?: number;
  onProgress?: (stage: string, done: number, total: number) => void;
}

interface Kept {
  s: AbSession;
  node: number;
  question: string;
  shown: string;
  reference: string;
}

export async function runAb(engine: Engine, opts: AbOptions): Promise<{ report: AbReport; items: AbItem[] }> {
  const p: AbParams = { ...AB, ...opts.params };
  const { model, embedder } = opts;
  const cache = opts.cache ?? memoryCache();

  const calibration = await calibrate(model, cache, (d, t) => opts.onProgress?.('calibrate judge', d, t));

  const eligible = selectSessions(engine, p);
  const sessions = opts.maxSessions ? eligible.slice(0, opts.maxSessions) : eligible;

  // 1. Write questions.
  const kept: Kept[] = [];
  const rejected: Record<string, number> = {};
  let passagesTried = 0;
  for (const [si, s] of sessions.entries()) {
    const project = engine.projectName(s.project);
    const seen = new Set<string>();
    let got = 0;
    for (const node of shuffled(s.passages, p.seed + s.pk).slice(0, p.maxPassages)) {
      if (got === p.questionsPerSession) break;
      passagesTried++;
      const key = `q:${node}`;
      let pair = cache.get(key) as { question: string; answer: string } | null | undefined;
      if (pair === undefined) {
        pair = await model.writeQuestion(engine.texts[node]!);
        cache.set(key, pair);
      }
      const reason = pair === null ? 'no usable reply' : rejectReason(pair.question, pair.answer, engine.texts[node]!, questionFor(project, pair.question.trim()), seen);
      if (pair === null || reason !== null) {
        rejected[reason!] = (rejected[reason!] ?? 0) + 1;
        continue;
      }
      const question = pair.question.trim();
      seen.add(norm(question));
      kept.push({ s, node, question, shown: questionFor(project, question), reference: pair.answer.trim() });
      got++;
    }
    opts.onProgress?.('write questions', si + 1, sessions.length);
  }

  // 2. Answer every question in every arm.
  let normTexts: string[] | null = null;
  const items: AbItem[] = [];
  const inMemory: boolean[] = [];
  for (const [qi, k] of kept.entries()) {
    const [qvec] = await retrying(() => embedder.embed([k.shown], 'query'), opts.retryDelaysMs);
    if (!qvec || qvec.length !== engine.dim) throw new Error('Query embedding has the wrong dimension. Was the model changed? Run `loci embed`.');
    const answers = {} as Record<Arm, string>;
    const contextNodes = {} as Record<Arm, number[]>;
    await Promise.all(
      ARMS.map(async (arm) => {
        const ctx = contextFor(engine, arm, k.s, k.node, k.shown, qvec, p);
        contextNodes[arm] = ctx.nodes;
        const key = `a:${k.node}:${arm}`;
        let answer = cache.get(key) as string | undefined;
        if (answer === undefined) {
          answer = (await model.answer(k.shown, ctx.text)).slice(0, p.answerChars);
          cache.set(key, answer);
        }
        answers[arm] = answer;
      }),
    );
    normTexts ??= engine.texts.map(norm);
    const ref = norm(k.reference);
    let found = false;
    for (let i = 0; i < engine.n && !found; i++) found = engine.ts[i]! < k.s.start && engine.sessionPk[i] !== k.s.pk && normTexts[i]!.includes(ref);
    inMemory.push(found);
    items.push({ sessionPk: k.s.pk, project: k.s.project, node: k.node, question: k.shown, reference: k.reference, answers, contextNodes, judged: {} as Record<Arm, boolean> });
    opts.onProgress?.('answer', qi + 1, kept.length);
  }

  // 3. Judge, blind. Every answer becomes a bare (question, reference, answer) triple; the triples
  //    are pooled across arms and shuffled, so neither content nor order tells the judge the arm.
  const triples = new Map<string, JudgeItem>();
  for (const it of items) {
    for (const arm of ARMS) {
      const t: JudgeItem = { question: it.question, reference: it.reference, answer: it.answers[arm] };
      triples.set(judgeKey(t), t);
    }
  }
  const verdicts = new Map<string, boolean>();
  for (const [i, [key, t]] of shuffled([...triples.entries()], p.seed).entries()) {
    verdicts.set(key, await judgeOnce(model, cache, t));
    if (i % 10 === 0) opts.onProgress?.('judge', i + 1, triples.size);
  }
  for (const it of items) {
    for (const arm of ARMS) it.judged[arm] = verdicts.get(judgeKey({ question: it.question, reference: it.reference, answer: it.answers[arm] }))!;
  }

  // 4. Numbers only from here on.
  const rows: AbRow[] = items.map((it, i) => {
    const pick = (f: (arm: Arm) => boolean) => Object.fromEntries(ARMS.map((arm) => [arm, f(arm) ? 1 : 0])) as Record<Arm, number>;
    return {
      sessionPk: it.sessionPk,
      project: it.project,
      node: it.node,
      answerInMemory: inMemory[i]!,
      judged: pick((arm) => it.judged[arm]),
      contains: pick((arm) => containsReference(it.reference, it.answers[arm])),
      abstained: pick((arm) => isAbstention(it.answers[arm])),
    };
  });
  const meta: AbMeta = {
    model: model.name,
    modelDigest: opts.modelDigest ?? null,
    promptHash: model.promptHash,
    embedModel: embedder.model,
    params: p,
    corpus: engine.n,
    eligibleSessions: eligible.length,
    passagesTried,
    rejected,
  };
  return { report: summariseAb(rows, calibration, meta, sessions.length), items };
}

// ---------------------------------------------------------------------------------------------
// Statistics. Pure: takes rows of 0s and 1s, so it is tested without a store or a model.

interface Cell {
  sessionPk: number;
  project: number;
  v: number;
}

const mean = (xs: readonly number[]): number => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : Number.NaN);

/** One value per session: the mean over that session's questions. Each session then counts once. */
function perSession(rows: readonly AbRow[], value: (r: AbRow) => number): Cell[] {
  const acc = new Map<number, { project: number; s: number; n: number }>();
  for (const r of rows) {
    const a = acc.get(r.sessionPk) ?? { project: r.project, s: 0, n: 0 };
    a.s += value(r);
    a.n++;
    acc.set(r.sessionPk, a);
  }
  return [...acc.entries()].map(([sessionPk, a]) => ({ sessionPk, project: a.project, v: a.s / a.n }));
}

/** Mean over sessions, resampling whole sessions in a single stratum. */
function sessionMean(rows: readonly AbRow[], value: (r: AbRow) => number): Interval {
  const cells = perSession(rows, value);
  return clusterBootstrap([cells.map((c) => [c])], (sample) => mean(sample.map((c) => c.v)));
}

/** Mean over questions: every question counts once, and whole sessions are still what is resampled. */
function questionMean(rows: readonly AbRow[], value: (r: AbRow) => number): Interval {
  const bySession = new Map<number, AbRow[]>();
  for (const r of rows) bySession.set(r.sessionPk, [...(bySession.get(r.sessionPk) ?? []), r]);
  return clusterBootstrap([[...bySession.values()]], (sample) => mean(sample.map(value)));
}

function countBy(rows: readonly AbRow[], key: (r: AbRow) => number): Map<number, number> {
  const out = new Map<number, number>();
  for (const r of rows) out.set(key(r), (out.get(key(r)) ?? 0) + 1);
  return out;
}

/** The same quantity macro-averaged by project, resampling sessions within each project. */
function projectMacroMean(rows: readonly AbRow[], value: (r: AbRow) => number): Interval {
  const byProject = new Map<number, Cell[][]>();
  for (const c of perSession(rows, value)) {
    const list = byProject.get(c.project) ?? [];
    list.push([c]);
    byProject.set(c.project, list);
  }
  return clusterBootstrap([...byProject.values()], (sample) => macroMean(sample, (c) => c.project, (c) => c.v));
}

export function summariseAb(rows: readonly AbRow[], calibration: Calibration, meta: AbMeta, sessionsTried: number): AbReport {
  const p = meta.params;
  const judgeOk = calibration.judge.balanced >= p.judgeGate;
  const outcome: Outcome = judgeOk ? 'judged' : 'contains';
  const diff = (a: Arm, b: Arm, o: Outcome) => sessionMean(rows, (r) => r[o][a] - r[o][b]);

  const arms: ArmScore[] = ARMS.map((arm) => ({
    arm,
    judged: sessionMean(rows, (r) => r.judged[arm]),
    contains: sessionMean(rows, (r) => r.contains[arm]),
    wrongAnswered: sessionMean(rows, (r) => (r[outcome][arm] === 0 && r.abstained[arm] === 0 ? 1 : 0)),
    abstained: mean(rows.map((r) => r.abstained[arm])),
  }));
  const pairs: [Arm, Arm][] = [
    ['memory', 'none'],
    ['memory', 'recency'],
    ['recency', 'none'],
    ['oracle', 'none'],
    ['instore', 'none'],
  ];
  const contrasts: Contrast[] = pairs.map(([a, b]) => ({ a, b, judged: diff(a, b, 'judged'), contains: diff(a, b, 'contains') }));

  const primary = diff('memory', 'none', outcome);
  const control = diff('oracle', 'none', outcome);
  const sessions = new Set(rows.map((r) => r.sessionPk)).size;
  const gates = { judge: judgeOk, positiveControl: control.lo > p.controlGate, yield: sessions >= p.minSessions };
  const verdict: Verdict = !gates.positiveControl || !gates.yield ? 'not interpretable' : primary.lo > 0 ? 'positive' : primary.hi < 0 ? 'negative' : 'null';

  const split = (present: boolean): Split | null => {
    const sub = rows.filter((r) => r.answerInMemory === present);
    if (sub.length === 0) return null;
    return { questions: sub.length, sessions: new Set(sub.map((r) => r.sessionPk)).size, lift: sessionMean(sub, (r) => r[outcome].memory - r[outcome].none) };
  };

  // The same contrast under other ways of averaging. A group with a tiny sample should not be able
  // to carry a verdict, so two of the views set a floor on the group size.
  const d = (r: AbRow) => r[outcome].memory - r[outcome].none;
  const perSessionN = countBy(rows, (r) => r.sessionPk);
  const perProjectN = countBy(rows, (r) => r.project);
  const fullSessions = rows.filter((r) => perSessionN.get(r.sessionPk)! >= p.questionsPerSession);
  const bigProjects = rows.filter((r) => perProjectN.get(r.project)! >= MIN_PROJECT_QUERIES);
  const projectMacro = projectMacroMean(rows, d);
  const view = (label: string, sub: readonly AbRow[], lift: Interval): View => ({ label, sessions: new Set(sub.map((r) => r.sessionPk)).size, questions: sub.length, projects: new Set(sub.map((r) => r.project)).size, lift });
  const robustness: View[] = [
    view('mean over sessions (the preregistered headline)', rows, primary),
    view('mean over questions', rows, questionMean(rows, d)),
    view(`mean over sessions that produced all ${p.questionsPerSession} questions`, fullSessions, sessionMean(fullSessions, d)),
    view('mean over projects, every project', rows, projectMacro),
    view(`mean over projects with ${MIN_PROJECT_QUERIES}+ questions`, bigProjects, projectMacroMean(bigProjects, d)),
  ];

  const agreement = { answers: 0, both: 0, neither: 0, judgeOnly: 0, containsOnly: 0 };
  for (const r of rows) {
    for (const arm of ARMS) {
      agreement.answers++;
      if (r.judged[arm] && r.contains[arm]) agreement.both++;
      else if (!r.judged[arm] && !r.contains[arm]) agreement.neither++;
      else if (r.judged[arm]) agreement.judgeOnly++;
      else agreement.containsOnly++;
    }
  }

  return {
    ...meta,
    sessions,
    droppedSessions: sessionsTried - sessions,
    projects: new Set(rows.map((r) => r.project)).size,
    questions: rows.length,
    calibration,
    arms,
    contrasts,
    outcome,
    gates,
    primary,
    verdict,
    rulesOutUsefulGain: primary.hi < p.usefulGain,
    projectMacro,
    robustness,
    answerInMemory: { share: mean(rows.map((r) => (r.answerInMemory ? 1 : 0))), present: split(true), absent: split(false) },
    agreement,
    rows: [...rows],
  };
}
