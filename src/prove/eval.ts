import type { Engine, Mode } from '../engine.ts';
import type { Embedder } from '../embed/ollama.ts';
import { BM25 } from '../index/bm25.ts';
import { rrf } from '../index/fusion.ts';
import { dot, exactTopK } from '../index/hnsw.ts';
import { nearCopyThreshold } from '../search/core.ts';
import { redundancy } from '../search/diverse.ts';
import { clusterBootstrap, macroMean } from './bootstrap.ts';
import type { Interval } from './bootstrap.ts';

/**
 * Retrieval evaluation with no manual labels.
 *
 * Question asked: when you start typing a prompt, can the engine pull up earlier work from the
 * same project, using nothing but the words of the prompt?
 *
 *   query      a real prompt you typed (first part, 20+ characters)
 *   candidates only chunks written BEFORE the prompt, and never from the prompt's own session
 *   relevant   a candidate from the same project as the prompt
 *
 * The temporal guard matters: without it the engine could "retrieve" the answer to a question
 * from the conversation that followed it. Scores are macro-averaged by project so the largest
 * project cannot carry the result, and intervals come from a session-clustered bootstrap.
 *
 * The label is coarse (same project, not "the passage that answers this"), so read the numbers as
 * "finds the right neighbourhood", not "finds the right sentence". The baselines say how much of
 * that the corpus gives away for free.
 */

export const K = 10;
/** Strength of the "spread" system. Chosen before it was run on the real corpus, and not tuned on it. */
export const SPREAD_LAMBDA = 0.5;
/**
 * A project needs at least this many queries to count in the headline average. The headline is a
 * mean over projects, so without a floor one project with a single query weighs as much as one
 * with five hundred. (Added after exactly that happened: a one-query project moved the
 * all-projects average by 0.03 between two runs. All views are still reported.)
 */
export const MIN_PROJECT_QUERIES = 10;

export interface EvalQuery {
  node: number;
  project: number;
  sessionPk: number;
  ts: number;
  text: string;
  eligible: number;
  relevant: number;
}

interface Row {
  project: number;
  sessionPk: number;
  p: number;
  rr: number;
}

export interface SystemScore {
  name: string;
  precision: Interval;
  mrr: Interval;
  /** Difference from the most-recent-k baseline on the same queries. */
  liftOverRecency: { precision: Interval; mrr: Interval } | null;
}

export interface EvalReport {
  k: number;
  queries: number;
  sessions: number;
  projects: number;
  corpus: number;
  medianEligible: number;
  /** Projects with fewer queries than this are left out of the headline average. */
  minProjectQueries: number;
  headlineQueries: number;
  headlineProjects: number;
  /** The recency comparison under different ways of averaging, to show how stable the verdict is. */
  robustness: { label: string; queries: number; projects: number; systems: { name: string; precision: number; lift: Interval | null }[] }[];
  systems: SystemScore[];
  masked: SystemScore[] | null;
  perProject: { label: string; queries: number; random: number; recency: number; hybrid: number; fused: number }[];
  knownItem: { queries: number; systems: { name: string; hit1: number; hit10: number; mrr: number }[]; randomHit10: number } | null;
  /**
   * How repetitive each ranking's top results are, and what folding near-copies does to it.
   * Absent from reports written before this was measured.
   */
  diversity?: {
    /** Two chunks at or above this similarity count as near-copies (set from the corpus). */
    nearCopy: number;
    /** Strength used for the "spread" system (maximal marginal relevance). Fixed before the run. */
    spreadLambda: number;
    systems: { name: string; meanSim: number; dupPairs: number; folded: number }[];
  };
}

export function selectQueries(engine: Engine): EvalQuery[] {
  const firstParts = new Set(
    (engine.store.db.prepare("SELECT id FROM chunks WHERE role = 'human' AND part = 0").all() as { id: number }[]).map((r) => Number(r.id) - 1),
  );
  const out: EvalQuery[] = [];
  for (const node of firstParts) {
    if (node >= engine.n) continue;
    const text = engine.texts[node]!;
    if (text.length < 20) continue;
    const project = engine.project[node]!;
    const sessionPk = engine.sessionPk[node]!;
    const ts = engine.ts[node]!;
    let eligible = 0;
    let relevant = 0;
    for (let i = 0; i < engine.n; i++) {
      if (engine.ts[i]! >= ts || engine.sessionPk[i] === sessionPk) continue;
      eligible++;
      if (engine.project[i] === project) relevant++;
    }
    // No earlier same-project work means there is nothing to find: not a fair question.
    if (relevant === 0 || eligible < K) continue;
    out.push({ node, project, sessionPk, ts, text, eligible, relevant });
  }
  return out.sort((a, b) => a.ts - b.ts || a.node - b.node);
}

function score(ranked: readonly number[], q: EvalQuery, engine: Engine): { p: number; rr: number } {
  let rel = 0;
  let rr = 0;
  for (let i = 0; i < Math.min(K, ranked.length); i++) {
    if (engine.project[ranked[i]!] === q.project) {
      rel++;
      if (rr === 0) rr = 1 / (i + 1);
    }
  }
  return { p: rel / K, rr };
}

/** Expected precision@K and reciprocal rank of a uniformly random ranking, in closed form. */
export function randomExpectation(eligible: number, relevant: number, k = K): { p: number; rr: number } {
  const p = relevant / eligible;
  let rr = 0;
  let noneBefore = 1; // probability the first r-1 picks were all irrelevant
  for (let r = 1; r <= Math.min(k, eligible); r++) {
    const left = eligible - (r - 1);
    rr += (noneBefore * (relevant / left)) / r;
    noneBefore *= Math.max(0, (eligible - relevant - (r - 1)) / left);
  }
  return { p, rr };
}

function summarise(name: string, rows: readonly Row[], recency: readonly Row[] | null): SystemScore {
  type R = Row & { dp?: number; drr?: number };
  /** Rows grouped by project, then by session: the strata and clusters of the bootstrap. */
  const strata = (rs: readonly R[]): R[][][] => {
    const byProject = new Map<number, Map<number, R[]>>();
    for (const r of rs) {
      let sessions = byProject.get(r.project);
      if (!sessions) {
        sessions = new Map();
        byProject.set(r.project, sessions);
      }
      const list = sessions.get(r.sessionPk) ?? [];
      list.push(r);
      sessions.set(r.sessionPk, list);
    }
    return [...byProject.values()].map((m) => [...m.values()]);
  };
  const macro = (f: (r: R) => number) => (sample: readonly R[]) => macroMean(sample, (r) => r.project, f);
  let lift: SystemScore['liftOverRecency'] = null;
  if (recency) {
    const paired: R[] = rows.map((r, i) => ({ ...r, dp: r.p - recency[i]!.p, drr: r.rr - recency[i]!.rr }));
    const ps = strata(paired);
    lift = { precision: clusterBootstrap(ps, macro((r) => r.dp ?? 0)), mrr: clusterBootstrap(ps, macro((r) => r.drr ?? 0)) };
  }
  const st = strata(rows);
  return { name, precision: clusterBootstrap(st, macro((r) => r.p)), mrr: clusterBootstrap(st, macro((r) => r.rr)), liftOverRecency: lift };
}

/** Hides project names and file paths, so a ranker cannot win by matching folder names. */
export function maskText(text: string, projectNames: readonly string[]): string {
  let out = text.replace(/(?:~|\.{1,2})?\/?(?:[\w.@-]+\/)+[\w.@-]*/g, ' PATH ');
  for (const name of projectNames) {
    if (name.length < 3) continue;
    const pattern = name
      .split(/[^A-Za-z0-9]+/)
      .filter(Boolean)
      .map((part) => part.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
      .join('[\\s._-]*');
    if (pattern) out = out.replace(new RegExp(pattern, 'gi'), ' PROJECT ');
  }
  return out.replace(/[ \t]{2,}/g, ' ');
}

export interface EvalOptions {
  embedder: Embedder;
  /** Also run the ablation with project names and paths hidden (re-embeds the corpus in memory). */
  mask?: boolean;
  maxQueries?: number;
  onProgress?: (stage: string, done: number, total: number) => void;
}

async function embedMany(embedder: Embedder, texts: readonly string[], kind: 'document' | 'query', onProgress?: (done: number, total: number) => void): Promise<Float32Array[]> {
  const out: Float32Array[] = [];
  for (let i = 0; i < texts.length; i += 32) {
    out.push(...(await embedder.embed(texts.slice(i, i + 32), kind)));
    onProgress?.(Math.min(i + 32, texts.length), texts.length);
  }
  return out;
}

export async function runEval(engine: Engine, opts: EvalOptions): Promise<EvalReport> {
  let queries = selectQueries(engine);
  if (opts.maxQueries && queries.length > opts.maxQueries) {
    // Evenly spaced subsample, so the time range stays covered.
    const step = queries.length / opts.maxQueries;
    queries = Array.from({ length: opts.maxQueries }, (_, i) => queries[Math.floor(i * step)]!);
  }
  const qvecs = await embedMany(opts.embedder, queries.map((q) => q.text), 'query', (d, t) => opts.onProgress?.('embed queries', d, t));

  // Nodes in time order, for the recency baseline.
  const byTime = Array.from({ length: engine.n }, (_, i) => i).sort((a, b) => engine.ts[b]! - engine.ts[a]! || b - a);
  const projectSizes = (q: EvalQuery) => {
    const counts = new Map<number, number>();
    for (let i = 0; i < engine.n; i++) {
      if (engine.ts[i]! >= q.ts || engine.sessionPk[i] === q.sessionPk) continue;
      counts.set(engine.project[i]!, (counts.get(engine.project[i]!) ?? 0) + 1);
    }
    return counts;
  };

  const rows: Record<string, Row[]> = { random: [], majority: [], recency: [], bm25: [], dense: [], hybrid: [], 'hybrid+recency': [], 'hybrid+fold': [], 'hybrid+spread': [] };
  // Repetition among each ranking's top results, summed over queries.
  const { tau: nearCopy } = nearCopyThreshold(engine);
  const simOf = (a: number, b: number) => dot(engine.vectors, a * engine.dim, engine.vectors, b * engine.dim, engine.dim);
  const rep: Record<string, { meanSim: number; dupPairs: number; folded: number }> = {
    hybrid: { meanSim: 0, dupPairs: 0, folded: 0 },
    'hybrid+fold': { meanSim: 0, dupPairs: 0, folded: 0 },
    'hybrid+spread': { meanSim: 0, dupPairs: 0, folded: 0 },
  };
  const note = (name: string, nodes: number[], folded: number) => {
    const r = redundancy(nodes, simOf, nearCopy);
    rep[name]!.meanSim += r.meanSim;
    rep[name]!.dupPairs += r.dupPairs;
    rep[name]!.folded += folded;
  };
  queries.forEach((q, qi) => {
    const base = { project: q.project, sessionPk: q.sessionPk };
    rows.random!.push({ ...base, ...randomExpectation(q.eligible, q.relevant) });

    const sizes = [...projectSizes(q).entries()].sort((a, b) => b[1] - a[1] || a[0] - b[0]);
    const isMajority = sizes[0]![0] === q.project ? 1 : 0;
    rows.majority!.push({ ...base, p: isMajority, rr: isMajority });

    const recent: number[] = [];
    for (const node of byTime) {
      if (engine.ts[node]! >= q.ts || engine.sessionPk[node] === q.sessionPk) continue;
      recent.push(node);
      if (recent.length === K) break;
    }
    rows.recency!.push({ ...base, ...score(recent, q, engine) });

    for (const mode of ['bm25', 'dense', 'hybrid'] as Mode[]) {
      const res = engine.searchWith(q.text, qvecs[qi]!, { k: K, mode, before: q.ts, excludeSessionPk: q.sessionPk });
      rows[mode]!.push({ ...base, ...score(res.hits.map((h) => h.node), q, engine) });
      if (mode === 'hybrid') note('hybrid', res.hits.map((h) => h.node), 0);
    }
    // The same hybrid ranking with near-copies folded, and with rank traded for variety.
    for (const [name, diversify] of [['hybrid+fold', 1], ['hybrid+spread', SPREAD_LAMBDA]] as const) {
      const res = engine.searchWith(q.text, qvecs[qi]!, { k: K, mode: 'hybrid', before: q.ts, excludeSessionPk: q.sessionPk, diversify });
      const nodes = res.hits.map((h) => h.node);
      rows[name]!.push({ ...base, ...score(nodes, q, engine) });
      note(name, nodes, res.folded);
    }
    const fused = engine.searchWith(q.text, qvecs[qi]!, { k: K, mode: 'recent', before: q.ts, excludeSessionPk: q.sessionPk });
    rows['hybrid+recency']!.push({ ...base, ...score(fused.hits.map((h) => h.node), q, engine) });
    if (qi % 50 === 0) opts.onProgress?.('search', qi, queries.length);
  });

  // Which queries count towards the headline: those from projects with enough queries.
  const perProjectCount = new Map<number, number>();
  for (const q of queries) perProjectCount.set(q.project, (perProjectCount.get(q.project) ?? 0) + 1);
  const headline = queries.map((q, i) => ((perProjectCount.get(q.project) ?? 0) >= MIN_PROJECT_QUERIES ? i : -1)).filter((i) => i >= 0);
  const take = (list: readonly Row[], idx: readonly number[]) => idx.map((i) => list[i]!);

  const systems = ['random', 'majority', 'recency', 'bm25', 'dense', 'hybrid', 'hybrid+recency', 'hybrid+fold', 'hybrid+spread'].map((name) =>
    summarise(name, take(rows[name]!, headline), name === 'recency' ? null : take(rows.recency!, headline)),
  );

  // The same recency comparison under three ways of averaging.
  const everything = queries.map((_, i) => i);
  const views: { label: string; idx: number[]; micro: boolean }[] = [
    { label: `macro, projects with ${MIN_PROJECT_QUERIES}+ queries (headline)`, idx: headline, micro: false },
    { label: 'macro, every project', idx: everything, micro: false },
    { label: 'micro, every query counts equally', idx: everything, micro: true },
  ];
  const robustness = views.map((v) => {
    // For the micro view all queries share one group, so the macro mean becomes a plain mean.
    const pick = (name: string) => take(rows[name]!, v.idx).map((r) => (v.micro ? { ...r, project: 0 } : r));
    const recency = pick('recency');
    return {
      label: v.label,
      queries: v.idx.length,
      projects: new Set(v.idx.map((i) => queries[i]!.project)).size,
      systems: ['recency', 'hybrid', 'hybrid+recency'].map((name) => {
        const sc = summarise(name, pick(name), name === 'recency' ? null : recency);
        return { name, precision: sc.precision.value, lift: sc.liftOverRecency?.precision ?? null };
      }),
    };
  });

  // Ablation: the same eval with project names and paths hidden from both queries and documents.
  let masked: SystemScore[] | null = null;
  if (opts.mask) {
    const names = engine.projects.map((p) => p.name);
    const docs = engine.texts.map((t) => maskText(t, names));
    const mq = queries.map((q) => maskText(q.text, names));
    const bm = BM25.build(docs);
    const dvecs = await embedMany(opts.embedder, docs, 'document', (d, t) => opts.onProgress?.('re-embed masked corpus', d, t));
    const flat = new Float32Array(engine.n * engine.dim);
    dvecs.forEach((v, i) => flat.set(v, i * engine.dim));
    const mqv = await embedMany(opts.embedder, mq, 'query');
    const mrows: Record<string, Row[]> = { bm25: [], dense: [], hybrid: [] };
    queries.forEach((q, qi) => {
      const base = { project: q.project, sessionPk: q.sessionPk };
      const allow = (i: number) => engine.ts[i]! < q.ts && engine.sessionPk[i] !== q.sessionPk;
      const lexical = bm.search(mq[qi]!, 100, allow).map((s) => s.doc);
      const dense = exactTopK(flat, engine.dim, engine.n, mqv[qi]!, 100, allow).map((h) => h.id);
      mrows.bm25!.push({ ...base, ...score(lexical, q, engine) });
      mrows.dense!.push({ ...base, ...score(dense, q, engine) });
      mrows.hybrid!.push({ ...base, ...score(rrf([dense, lexical], K).map((f) => f.id), q, engine) });
    });
    masked = ['bm25', 'dense', 'hybrid'].map((name) => summarise(`${name} (masked)`, take(mrows[name]!, headline), take(rows.recency!, headline)));
  }

  // Per-project view, anonymised: projects are labelled by size, never by name.
  const order = [...new Set(queries.map((q) => q.project))]
    .map((p) => ({ p, n: queries.filter((q) => q.project === p).length }))
    .sort((a, b) => b.n - a.n);
  const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);
  const perProject = order.map(({ p, n }, i) => {
    const pick = (name: string) => mean(rows[name]!.filter((r) => r.project === p).map((r) => r.p));
    return { label: `P${i + 1}`, queries: n, random: pick('random'), recency: pick('recency'), hybrid: pick('hybrid'), fused: pick('hybrid+recency') };
  });

  const eligibleSorted = queries.map((q) => q.eligible).sort((a, b) => a - b);
  return {
    k: K,
    queries: queries.length,
    sessions: new Set(queries.map((q) => q.sessionPk)).size,
    projects: order.length,
    corpus: engine.n,
    medianEligible: eligibleSorted[Math.floor(eligibleSorted.length / 2)] ?? 0,
    minProjectQueries: MIN_PROJECT_QUERIES,
    headlineQueries: headline.length,
    headlineProjects: new Set(headline.map((i) => queries[i]!.project)).size,
    robustness,
    systems,
    masked,
    perProject,
    knownItem: await knownItem(engine, opts.embedder),
    diversity: {
      nearCopy,
      spreadLambda: SPREAD_LAMBDA,
      systems: Object.entries(rep).map(([name, r]) => ({
        name,
        meanSim: r.meanSim / Math.max(1, queries.length),
        dupPairs: r.dupPairs / Math.max(1, queries.length),
        folded: r.folded / Math.max(1, queries.length),
      })),
    },
  };
}

/**
 * Known-item test: a session's title (written by a model from the session, never stored in the
 * index) should retrieve a chunk of that same session.
 */
async function knownItem(engine: Engine, embedder: Embedder): Promise<EvalReport['knownItem']> {
  const sizes = new Map<number, number>();
  for (let i = 0; i < engine.n; i++) sizes.set(engine.sessionPk[i]!, (sizes.get(engine.sessionPk[i]!) ?? 0) + 1);
  const targets = [...engine.sessions.values()].filter((s) => s.title && s.title.length >= 8 && (sizes.get(s.pk) ?? 0) >= 3);
  if (targets.length === 0) return null;
  const qv = await embedMany(embedder, targets.map((s) => s.title!), 'query');
  const systems = (['bm25', 'dense', 'hybrid'] as Mode[]).map((mode) => {
    let hit1 = 0;
    let hit10 = 0;
    let mrr = 0;
    targets.forEach((s, i) => {
      const res = engine.searchWith(s.title!, qv[i]!, { k: 50, mode });
      const rank = res.hits.findIndex((h) => engine.sessionPk[h.node] === s.pk) + 1;
      if (rank === 1) hit1++;
      if (rank >= 1 && rank <= 10) hit10++;
      if (rank >= 1) mrr += 1 / rank;
    });
    return { name: mode, hit1: hit1 / targets.length, hit10: hit10 / targets.length, mrr: mrr / targets.length };
  });
  // Chance of a random top-10 containing any chunk of the target session, averaged over targets.
  let randomHit10 = 0;
  for (const s of targets) {
    const m = sizes.get(s.pk) ?? 0;
    let miss = 1;
    for (let r = 0; r < 10; r++) miss *= Math.max(0, (engine.n - m - r) / (engine.n - r));
    randomHit10 += 1 - miss;
  }
  return { queries: targets.length, systems, randomHit10: randomHit10 / targets.length };
}
