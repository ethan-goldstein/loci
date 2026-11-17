import type { Hnsw } from '../index/hnsw.ts';
import type { SearchResult } from '../search/core.ts';

/**
 * Turns engine results into what the viewer draws. Shared by the local server and by the public
 * demo's in-browser engine, so both answer in exactly the same shape. No Node dependencies.
 */

export interface Described {
  particle: number;
  chunkId: number;
  project: string;
  role: 'human' | 'assistant';
  date: string;
  session: string | null;
  text: string;
}

/** Describes a node for display, with its text cut to `chars` characters. */
export type Describe = (node: number, chars: number) => Described;

export function clip(text: string, chars: number): string {
  return text.length > chars ? `${text.slice(0, chars)}…` : text;
}

export function shapeSearch(res: SearchResult, describe: Describe, rankOf: ArrayLike<number>, info: { ms: number; mode: string; degraded: boolean }) {
  return {
    ms: Number(info.ms.toFixed(1)),
    mode: info.mode,
    degraded: info.degraded,
    dense: res.dense,
    hits: res.hits.map((h) => ({
      ...describe(h.node, 420),
      score: Number(h.score.toFixed(5)),
      similarity: h.sim === null ? null : Number(h.sim.toFixed(4)),
      denseRank: h.denseRank,
      bm25Rank: h.bm25Rank,
    })),
    // Trace in particle indices: [layer, particle, parent particle or -1, similarity].
    trace: (res.trace ?? []).map(([layer, node, parent, sim]): [number, number, number, number] => [layer, rankOf[node]!, parent < 0 ? -1 : rankOf[parent]!, Number(sim.toFixed(4))]),
  };
}

export function shapeChunk(node: number, hnsw: Hnsw, describe: Describe) {
  const neighbours = hnsw
    .neighbours(node)
    .slice(0, 6)
    .map((nb) => ({ ...describe(nb.id, 160), similarity: Number(nb.sim.toFixed(4)) }));
  return { ...describe(node, 2400), level: hnsw.level(node), neighbours };
}
