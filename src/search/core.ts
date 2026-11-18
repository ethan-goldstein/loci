import type { BM25 } from '../index/bm25.ts';
import { rrf } from '../index/fusion.ts';
import { dot } from '../index/hnsw.ts';
import type { Hnsw, TraceStep } from '../index/hnsw.ts';
import { calibrate } from '../index/simstats.ts';
import { pickDiverse } from './diverse.ts';

/**
 * The search itself, with no dependency on files, SQLite or Node. The local engine calls this
 * with data loaded from the private store; the public demo calls the very same function in the
 * browser with its invented data. One code path, so the demo shows the real thing.
 *
 * "Node" means a row of the vector array.
 */

/**
 * hybrid  dense + BM25, fused by reciprocal rank
 * recent  hybrid plus a third list, the most recent chunks, fused the same way (no tuned weights)
 */
export type Mode = 'hybrid' | 'dense' | 'bm25' | 'recent';

export interface Filter {
  projectId?: number;
  role?: 'human' | 'assistant';
  /** Only chunks strictly earlier than this time (ms). The eval's temporal guard. */
  before?: number;
  after?: number;
  excludeSessionPk?: number;
}

export interface SearchRequest extends Filter {
  k?: number;
  mode?: Mode;
  /** HNSW candidate list size. Larger is slower and more accurate. */
  ef?: number;
  trace?: boolean;
  /**
   * Keep near-copies from filling the list (see diverse.ts). Absent: off, the plain ranking.
   * 1: fold near-copies. Below 1: also trade rank for variety, more strongly the lower it is.
   */
  diversify?: number;
}

/** What the viewer, the command line and the MCP server use: fold near-copies, reorder nothing else. */
export const DIVERSIFY_DEFAULT = 1;

export interface EngineHit {
  node: number;
  chunkId: number;
  score: number;
  /** Cosine similarity to the query, when the dense ranker saw this chunk. */
  sim: number | null;
  denseRank: number;
  bm25Rank: number;
}

export interface SearchResult {
  hits: EngineHit[];
  trace: TraceStep[] | null;
  /** How the dense side was answered: the graph, an exact scan (restrictive filter), or not at all. */
  dense: 'hnsw' | 'exact' | 'none';
  /** Near-copies left out of the list. Always 0 when `diversify` is off. */
  folded: number;
}

/** Everything a search reads. All arrays are indexed by node. */
export interface CorpusView {
  n: number;
  dim: number;
  vectors: Float32Array;
  hnsw: Hnsw;
  bm25: BM25;
  /** Milliseconds since the epoch. */
  ts: ArrayLike<number>;
  project: ArrayLike<number>;
  /** 0 for a prompt the person typed, 1 for the assistant. */
  role: ArrayLike<number>;
  sessionPk: ArrayLike<number>;
  /** Nodes from newest to oldest, for the recency list. */
  newestFirst: ArrayLike<number>;
}

const POOL = 100;

/**
 * When are two chunks near-copies? Decided from the corpus itself, once per corpus: well above
 * the similarity of a typical unrelated pair (see simstats.ts). Cached, because it does not change.
 */
const nearCopy = new WeakMap<CorpusView, { tau: number; median: number }>();
export function nearCopyThreshold(view: CorpusView): { tau: number; median: number } {
  let t = nearCopy.get(view);
  if (!t) {
    const c = calibrate(view.vectors, view.dim, view.n, view.sessionPk, 5, 17, 20_000);
    t = { tau: c.tau, median: c.median };
    nearCopy.set(view, t);
  }
  return t;
}
/** Below this share of allowed nodes a filtered graph walk wastes effort; scan the few instead. */
const EXACT_BELOW = 0.2;

/** Builds a node predicate from a filter, or undefined when nothing is filtered. */
export function allowFor(view: CorpusView, f: Filter): ((node: number) => boolean) | undefined {
  const { projectId, role, before, after, excludeSessionPk } = f;
  if (projectId === undefined && role === undefined && before === undefined && after === undefined && excludeSessionPk === undefined) {
    return undefined;
  }
  const roleCode = role === undefined ? -1 : role === 'human' ? 0 : 1;
  return (node) =>
    (projectId === undefined || view.project[node] === projectId) &&
    (roleCode < 0 || view.role[node] === roleCode) &&
    (before === undefined || view.ts[node]! < before) &&
    (after === undefined || view.ts[node]! >= after) &&
    (excludeSessionPk === undefined || view.sessionPk[node] !== excludeSessionPk);
}

/** Ranks the corpus for a query. `qvec` may be null only for keyword-only search. */
export function searchCore(view: CorpusView, query: string, qvec: Float32Array | null, req: SearchRequest = {}): SearchResult {
  const k = req.k ?? 10;
  const mode = req.mode ?? 'hybrid';
  const allow = allowFor(view, req);
  const pool = Math.max(POOL, k);

  let dense: { id: number; sim: number }[] = [];
  let denseHow: SearchResult['dense'] = 'none';
  const trace: TraceStep[] | null = req.trace ? [] : null;
  if (mode !== 'bm25') {
    if (!qvec) throw new Error('dense search needs a query vector');
    let share = 1;
    if (allow) {
      let ok = 0;
      for (let i = 0; i < view.n; i++) if (allow(i)) ok++;
      share = ok / Math.max(view.n, 1);
    }
    if (share < EXACT_BELOW) {
      dense = view.hnsw.exact(qvec, pool, allow);
      denseHow = 'exact';
    } else {
      const ef = Math.min(Math.max(req.ef ?? 128, pool), 2048);
      dense = view.hnsw.search(qvec, pool, ef, { allow, trace: trace ?? undefined });
      denseHow = 'hnsw';
    }
  }
  const lexical = mode !== 'dense' ? view.bm25.search(query, pool, allow) : [];

  const simOf = new Map(dense.map((d) => [d.id, d.sim]));
  const denseRank = new Map(dense.map((d, i) => [d.id, i + 1]));
  const bm25Rank = new Map(lexical.map((x, i) => [x.doc, i + 1]));

  // Every mode produces one ranked list of candidates with a score each; the hits are built from it.
  let ranked: { id: number; score: number }[];
  if (mode === 'dense') ranked = dense.map((d) => ({ id: d.id, score: d.sim }));
  else if (mode === 'bm25') ranked = lexical.map((x) => ({ id: x.doc, score: x.score }));
  else {
    const lists = [dense.map((d) => d.id), lexical.map((x) => x.doc)];
    if (mode === 'recent') {
      const recent: number[] = [];
      for (let i = 0; i < view.n && recent.length < pool; i++) {
        const node = view.newestFirst[i]!;
        if (!allow || allow(node)) recent.push(node);
      }
      lists.push(recent);
    }
    ranked = rrf(lists, pool).map((f) => ({ id: f.id, score: f.score }));
  }

  let folded = 0;
  if (req.diversify === undefined) ranked = ranked.slice(0, k);
  else {
    // Only the top few times k are considered, so nothing ranked far down can be promoted.
    const candidates = ranked.slice(0, Math.max(3 * k, 30));
    const scoreOf = new Map(candidates.map((c) => [c.id, c.score]));
    const { tau, median } = nearCopyThreshold(view);
    const sim = (a: number, b: number) => dot(view.vectors, a * view.dim, view.vectors, b * view.dim, view.dim);
    const out = pickDiverse(candidates.map((c) => c.id), k, sim, { dupSim: tau, lambda: req.diversify, baseline: median });
    folded = out.folded;
    ranked = out.picked.map((id) => ({ id, score: scoreOf.get(id)! }));
  }

  const hits: EngineHit[] = ranked.map((r) => ({
    node: r.id,
    chunkId: r.id + 1,
    score: r.score,
    sim: simOf.get(r.id) ?? null,
    denseRank: denseRank.get(r.id) ?? 0,
    bm25Rank: bm25Rank.get(r.id) ?? 0,
  }));
  return { hits, trace, dense: denseHow, folded };
}
