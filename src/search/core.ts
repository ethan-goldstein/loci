import type { BM25 } from '../index/bm25.ts';
import { rrf } from '../index/fusion.ts';
import { exactTopK } from '../index/hnsw.ts';
import type { Hnsw, TraceStep } from '../index/hnsw.ts';

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
}

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
      dense = exactTopK(view.vectors, view.dim, view.n, qvec, pool, allow);
      denseHow = 'exact';
    } else {
      const ef = Math.min(Math.max(req.ef ?? 128, pool), 2048);
      dense = view.hnsw.search(qvec, pool, ef, { allow, trace: trace ?? undefined });
      denseHow = 'hnsw';
    }
  }
  const lexical = mode !== 'dense' ? view.bm25.search(query, pool, allow) : [];

  const simOf = new Map(dense.map((d) => [d.id, d.sim]));
  const hits: EngineHit[] = [];
  if (mode === 'dense') {
    dense.slice(0, k).forEach((d, i) => hits.push({ node: d.id, chunkId: d.id + 1, score: d.sim, sim: d.sim, denseRank: i + 1, bm25Rank: 0 }));
  } else if (mode === 'bm25') {
    lexical.slice(0, k).forEach((s, i) => hits.push({ node: s.doc, chunkId: s.doc + 1, score: s.score, sim: null, denseRank: 0, bm25Rank: i + 1 }));
  } else {
    const lists = [dense.map((d) => d.id), lexical.map((s) => s.doc)];
    if (mode === 'recent') {
      const recent: number[] = [];
      for (let i = 0; i < view.n && recent.length < pool; i++) {
        const node = view.newestFirst[i]!;
        if (!allow || allow(node)) recent.push(node);
      }
      lists.push(recent);
    }
    for (const f of rrf(lists, k)) {
      hits.push({ node: f.id, chunkId: f.id + 1, score: f.score, sim: simOf.get(f.id) ?? null, denseRank: f.ranks[0]!, bm25Rank: f.ranks[1]! });
    }
  }
  return { hits, trace, dense: denseHow };
}
