import { describe, expect, it } from 'vitest';
import { Heap } from '../../src/index/heap.ts';
import { DEFAULT_PARAMS, Hnsw, TRACE_CAP, exactTopK } from '../../src/index/hnsw.ts';
import type { TraceStep } from '../../src/index/hnsw.ts';
import { clusteredVectors, rng } from '../../src/prove/synth.ts';

const DIM = 48;
const N = 4000;
// Data and queries are drawn together so they share the same cluster centres.
const all = clusteredVectors(N + 200, DIM, 24, 0.35, 11).data;
const data = all.subarray(0, N * DIM);
const queries = all.subarray(N * DIM);
const index = Hnsw.build(data, DIM, DEFAULT_PARAMS);

function recallAt(k: number, ef: number, idx = index): number {
  let hit = 0;
  for (let qi = 0; qi < 200; qi++) {
    const q = queries.subarray(qi * DIM, (qi + 1) * DIM);
    const truth = new Set(idx.exact(q, k).map((h) => h.id));
    for (const h of idx.search(q, k, ef)) if (truth.has(h.id)) hit++;
  }
  return hit / (200 * k);
}

describe('Heap', () => {
  it('pops in order for both directions', () => {
    const rand = rng(3);
    const xs = Array.from({ length: 500 }, () => rand());
    const min = new Heap(false, 4);
    const max = new Heap(true, 4);
    xs.forEach((x, i) => {
      min.push(x, i);
      max.push(x, i);
    });
    const sorted = [...xs].sort((a, b) => a - b);
    expect([...min.drainAscending().keys]).toEqual(sorted);
    expect([...max.drainAscending().keys]).toEqual(sorted);
  });
});

describe('Hnsw against the brute-force oracle', () => {
  it('reaches high recall and recall rises with ef', () => {
    const r16 = recallAt(10, 16);
    const r64 = recallAt(10, 64);
    const r128 = recallAt(10, 128);
    const r256 = recallAt(10, 256);
    console.log(`recall@10 on ${N} synthetic vectors: ef16 ${r16}, ef64 ${r64}, ef128 ${r128}, ef256 ${r256}`);
    expect(r64).toBeGreaterThanOrEqual(0.9);
    expect(r128).toBeGreaterThanOrEqual(0.96);
    expect(r256).toBeGreaterThanOrEqual(0.99);
    expect(r256).toBeGreaterThanOrEqual(r128);
    expect(r128).toBeGreaterThanOrEqual(r64);
    expect(r64).toBeGreaterThanOrEqual(r16);
  });

  it('returns results sorted by similarity with correct similarity values', () => {
    const q = queries.subarray(0, DIM);
    const hits = index.search(q, 10, 64);
    expect(hits).toHaveLength(10);
    for (let i = 1; i < hits.length; i++) expect(hits[i - 1]!.sim).toBeGreaterThanOrEqual(hits[i]!.sim);
    const exact = new Map(exactTopK(data, DIM, N, q, N).map((h) => [h.id, h.sim]));
    for (const h of hits) expect(h.sim).toBeCloseTo(exact.get(h.id)!, 5);
  });

  it('finds an indexed vector as its own nearest neighbour', () => {
    for (const id of [0, 1, 777, N - 1]) {
      const hits = index.search(data.subarray(id * DIM, (id + 1) * DIM), 1, 64);
      expect(hits[0]!.sim).toBeCloseTo(1, 4);
    }
  });

  it('is deterministic for a given seed', () => {
    const again = Hnsw.build(data, DIM, DEFAULT_PARAMS);
    expect(again.toBuffer().equals(index.toBuffer())).toBe(true);
    const other = Hnsw.build(data, DIM, { ...DEFAULT_PARAMS, seed: 5 });
    expect(other.toBuffer().equals(index.toBuffer())).toBe(false);
  });

  it('respects link caps and keeps the graph connected enough to reach every node', () => {
    const s = index.stats();
    expect(s.n).toBe(N);
    expect(s.meanDegree0).toBeGreaterThan(4);
    expect(s.meanDegree0).toBeLessThanOrEqual(DEFAULT_PARAMS.M0);
    // Breadth-first over layer 0 from the entry point should reach (almost) everything.
    const seen = new Uint8Array(N);
    const queue = [index.entry];
    seen[index.entry] = 1;
    let reached = 0;
    while (queue.length) {
      const c = queue.pop()!;
      reached++;
      for (const nb of index.neighbours(c)) {
        expect(nb.id).toBeLessThan(N);
        if (!seen[nb.id]) {
          seen[nb.id] = 1;
          queue.push(nb.id);
        }
      }
    }
    expect(reached / N).toBeGreaterThan(0.999);
  });

  it('round-trips through its file format and searches identically', () => {
    const loaded = Hnsw.fromBuffer(index.toBuffer(), data, DIM);
    expect(loaded.stats()).toEqual(index.stats());
    for (let qi = 0; qi < 20; qi++) {
      const q = queries.subarray(qi * DIM, (qi + 1) * DIM);
      expect(loaded.search(q, 10, 64)).toEqual(index.search(q, 10, 64));
    }
  });

  it('extends incrementally to the same graph as a one-shot build', () => {
    const half = Hnsw.build(data, DIM, DEFAULT_PARAMS, N / 2);
    const resumed = Hnsw.fromBuffer(half.toBuffer(), data, DIM);
    resumed.extend(N);
    expect(resumed.toBuffer().equals(index.toBuffer())).toBe(true);
  });

  it('rejects corrupt and mismatched files', () => {
    const good = index.toBuffer();
    expect(() => Hnsw.fromBuffer(Buffer.from('nope'), data, DIM)).toThrow();
    expect(() => Hnsw.fromBuffer(good, data, DIM + 1)).toThrow(/dimension/);
    expect(() => Hnsw.fromBuffer(good.subarray(0, good.length - 64), data, DIM)).toThrow();
    expect(() => Hnsw.fromBuffer(good, data.subarray(0, 100 * DIM), DIM)).toThrow(/more nodes/);
    const bad = Buffer.from(good);
    // First layer-0 link of node 0 points far outside the graph.
    const linkOff = 48 + ((N + 3) & ~3) * 2;
    bad.writeUInt32LE(N + 12345, linkOff);
    expect(() => Hnsw.fromBuffer(bad, data, DIM)).toThrow(/corrupt/);
  });

  it('only returns allowed nodes when filtered, and matches the filtered oracle', () => {
    const allow = (id: number) => id % 3 === 0;
    let hit = 0;
    for (let qi = 0; qi < 100; qi++) {
      const q = queries.subarray(qi * DIM, (qi + 1) * DIM);
      const hits = index.search(q, 10, 128, { allow });
      expect(hits.every((h) => allow(h.id))).toBe(true);
      const truth = new Set(index.exact(q, 10, allow).map((h) => h.id));
      for (const h of hits) if (truth.has(h.id)) hit++;
    }
    expect(hit / 1000).toBeGreaterThanOrEqual(0.95);
  });

  it('records a trace that is a real walk over real edges', () => {
    const q = queries.subarray(0, DIM);
    const trace: TraceStep[] = [];
    const hits = index.search(q, 10, 64, { trace });
    expect(trace.length).toBeGreaterThan(5);
    expect(trace.length).toBeLessThanOrEqual(TRACE_CAP);
    expect(trace[0]![1]).toBe(index.entry);
    expect(trace[0]![2]).toBe(-1);
    const visited = new Set([index.entry]);
    for (const [layer, node, parent, sim] of trace.slice(1)) {
      expect(layer).toBeGreaterThanOrEqual(0);
      expect(visited.has(parent)).toBe(true); // every step starts from somewhere already reached
      expect(index.level(node)).toBeGreaterThanOrEqual(layer);
      expect(sim).toBeGreaterThanOrEqual(-1.001);
      expect(sim).toBeLessThanOrEqual(1.001);
      visited.add(node);
    }
    // Layers never go back up.
    for (let i = 1; i < trace.length; i++) expect(trace[i]![0]).toBeLessThanOrEqual(trace[i - 1]![0]);
    for (const h of hits) expect(visited.has(h.id)).toBe(true);
  });

  it('handles empty and tiny indexes', () => {
    const empty = new Hnsw(new Float32Array(0), DIM);
    expect(empty.search(queries.subarray(0, DIM), 5, 16)).toEqual([]);
    const one = Hnsw.build(data.subarray(0, DIM), DIM);
    expect(one.search(queries.subarray(0, DIM), 5, 16)).toHaveLength(1);
    const few = Hnsw.build(data.subarray(0, 5 * DIM), DIM);
    expect(few.search(queries.subarray(0, DIM), 10, 16)).toHaveLength(5);
    expect(() => index.search(new Float32Array(DIM + 1), 5, 16)).toThrow(/dimension/);
  });
});
