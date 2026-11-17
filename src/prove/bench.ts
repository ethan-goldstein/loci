import os from 'node:os';
import type { Engine } from '../engine.ts';
import type { Embedder } from '../embed/ollama.ts';
import { BM25, ftsQuery } from '../index/bm25.ts';
import { DEFAULT_PARAMS, Hnsw, exactTopK } from '../index/hnsw.ts';
import { percentile } from './bootstrap.ts';
import { selectQueries } from './eval.ts';
import { clusteredVectors } from './synth.ts';

/**
 * Index benchmarks. Every approximate answer is compared with an exact brute-force scan of the
 * same vectors, so "recall" here is measured against ground truth, not assumed.
 */

export interface SweepPoint {
  ef: number;
  recall10: number;
  p50Ms: number;
  p95Ms: number;
  qps: number;
}

export interface IndexBench {
  label: string;
  n: number;
  dim: number;
  queries: number;
  buildSeconds: number;
  indexBytes: number;
  vectorBytes: number;
  maxLevel: number;
  meanDegree0: number;
  exactP50Ms: number;
  exactP95Ms: number;
  sweep: SweepPoint[];
}

export interface BenchReport {
  machine: { cpu: string; cores: number; memoryGb: number; node: string; platform: string };
  params: typeof DEFAULT_PARAMS;
  real: IndexBench;
  synthetic: IndexBench | null;
  bm25: { docs: number; vocabulary: number; buildMs: number; p50Ms: number; p95Ms: number; ftsQueries: number; ftsTop10Agreement: number };
  rssMb: number;
}

const EFS = [16, 32, 64, 128, 256];

function timeEach(run: (qi: number) => void, count: number): number[] {
  const ms: number[] = [];
  for (let qi = 0; qi < count; qi++) {
    const t = performance.now();
    run(qi);
    ms.push(performance.now() - t);
  }
  return ms.sort((a, b) => a - b);
}

export function benchIndex(label: string, vectors: Float32Array, dim: number, n: number, queries: Float32Array, nq: number, onProgress?: (done: number, total: number) => void): IndexBench {
  const t0 = performance.now();
  const index = Hnsw.build(vectors, dim, DEFAULT_PARAMS, n, onProgress);
  const buildSeconds = (performance.now() - t0) / 1000;
  const q = (qi: number) => queries.subarray(qi * dim, (qi + 1) * dim);

  const truth: Set<number>[] = new Array<Set<number>>(nq);
  const exactMs = timeEach((qi) => {
    truth[qi] = new Set(exactTopK(vectors, dim, n, q(qi), 10).map((h) => h.id));
  }, nq);

  const sweep: SweepPoint[] = [];
  for (const ef of EFS) {
    for (let qi = 0; qi < Math.min(50, nq); qi++) index.search(q(qi), 10, ef); // warm up
    let hit = 0;
    const ms = timeEach((qi) => {
      for (const h of index.search(q(qi), 10, ef)) if (truth[qi]!.has(h.id)) hit++;
    }, nq);
    const total = ms.reduce((a, b) => a + b, 0);
    sweep.push({ ef, recall10: hit / (nq * 10), p50Ms: percentile(ms, 0.5), p95Ms: percentile(ms, 0.95), qps: (nq / total) * 1000 });
  }
  const stats = index.stats();
  return {
    label,
    n,
    dim,
    queries: nq,
    buildSeconds,
    indexBytes: index.toBuffer().length,
    vectorBytes: n * dim * 4,
    maxLevel: stats.maxLevel,
    meanDegree0: stats.meanDegree0,
    exactP50Ms: percentile(exactMs, 0.5),
    exactP95Ms: percentile(exactMs, 0.95),
    sweep,
  };
}

export interface BenchOptions {
  embedder: Embedder;
  maxQueries?: number;
  /** Size of the labelled synthetic scale test; 0 skips it. */
  syntheticN?: number;
  onProgress?: (stage: string, done: number, total: number) => void;
}

export async function runBench(engine: Engine, opts: BenchOptions): Promise<BenchReport> {
  const maxQueries = opts.maxQueries ?? 500;
  let queries = selectQueries(engine);
  if (queries.length > maxQueries) {
    const step = queries.length / maxQueries;
    queries = Array.from({ length: maxQueries }, (_, i) => queries[Math.floor(i * step)]!);
  }
  if (queries.length === 0) throw new Error('No usable queries in the corpus yet.');
  const flat = new Float32Array(queries.length * engine.dim);
  for (let i = 0; i < queries.length; i += 32) {
    const vecs = await opts.embedder.embed(queries.slice(i, i + 32).map((q) => q.text), 'query');
    vecs.forEach((v, j) => flat.set(v, (i + j) * engine.dim));
    opts.onProgress?.('embed queries', Math.min(i + 32, queries.length), queries.length);
  }

  const real = benchIndex('real corpus', engine.vectors, engine.dim, engine.n, flat, queries.length, (d, t) => opts.onProgress?.('build real index', d, t));

  let synthetic: IndexBench | null = null;
  const sn = opts.syntheticN ?? 50_000;
  if (sn > 0) {
    const sdim = 128;
    // Data and queries come from one draw, so queries fall inside the same clusters as the data.
    const all = clusteredVectors(sn + 500, sdim, 64, 0.5, 2026).data;
    synthetic = benchIndex('synthetic (64 Gaussian clusters)', all.subarray(0, sn * sdim), sdim, sn, all.subarray(sn * sdim), 500, (d, t) =>
      opts.onProgress?.('build synthetic index', d, t),
    );
  }

  const tb = performance.now();
  const bm = BM25.build(engine.texts);
  const buildMs = performance.now() - tb;
  const bmMs = timeEach((qi) => void bm.search(queries[qi]!.text, 10), queries.length);
  // Oracle 2 on real text: do the from-scratch ranker and SQLite FTS5 agree on the top 10?
  let agree = 0;
  let total = 0;
  for (const q of queries) {
    const expr = ftsQuery(q.text);
    if (!expr) continue;
    const mine = bm.search(q.text, 10);
    const theirs = engine.store.ftsTop(expr, 10).filter((r) => r.id <= engine.n);
    if (theirs.length === 0) continue;
    const cutoff = theirs[theirs.length - 1]!.score;
    const ids = new Set(theirs.map((r) => r.id - 1));
    for (const m of mine) {
      total++;
      // A document tied with the 10th score may legitimately swap in or out.
      if (ids.has(m.doc) || Math.abs(m.score - cutoff) < 1e-6 * Math.max(1, cutoff)) agree++;
    }
  }

  const cpu = os.cpus();
  return {
    machine: {
      cpu: cpu[0]?.model ?? 'unknown',
      cores: cpu.length,
      memoryGb: Math.round(os.totalmem() / 2 ** 30),
      node: process.version,
      platform: `${process.platform}-${process.arch}`,
    },
    params: DEFAULT_PARAMS,
    real,
    synthetic,
    bm25: {
      docs: bm.n,
      vocabulary: bm.vocabulary,
      buildMs,
      p50Ms: percentile(bmMs, 0.5),
      p95Ms: percentile(bmMs, 0.95),
      ftsQueries: queries.length,
      ftsTop10Agreement: total ? agree / total : Number.NaN,
    },
    rssMb: Math.round(process.memoryUsage().rss / 2 ** 20),
  };
}
