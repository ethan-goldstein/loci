import { describe, expect, it, vi } from 'vitest';
import { Hnsw, dot, exactTopK } from '../../src/index/hnsw.ts';
import { loadSimdKernel, simdAvailable } from '../../src/kernel/simd.ts';
import { kernelModule } from '../../src/kernel/wasm.ts';
import { clusteredVectors, rng } from '../../src/prove/synth.ts';

/**
 * The WebAssembly SIMD kernel against its oracle, the TypeScript loop it replaces. The two sum in
 * a different order and at a different precision (32-bit lanes against 64-bit numbers), so they
 * agree to a tolerance, not bit for bit.
 */

function randomRows(n: number, dim: number, seed: number): Float32Array {
  const rand = rng(seed);
  const v = new Float32Array(n * dim);
  for (let i = 0; i < n; i++) {
    let norm = 0;
    for (let d = 0; d < dim; d++) {
      const x = rand() * 2 - 1;
      v[i * dim + d] = x;
      norm += x * x;
    }
    norm = Math.sqrt(norm) || 1;
    for (let d = 0; d < dim; d++) v[i * dim + d] = v[i * dim + d]! / norm;
  }
  return v;
}

describe('the assembled module', () => {
  it('is valid WebAssembly and small', () => {
    const bytes = kernelModule();
    expect(WebAssembly.validate(bytes)).toBe(true);
    expect(bytes.length).toBeLessThan(600);
    // Assembled the same way every time: there is no hidden state in the assembler.
    expect([...kernelModule()]).toEqual([...bytes]);
    expect(simdAvailable()).toBe(true);
  });

  it('agrees with the TypeScript loop at every length, including ones that leave a remainder', () => {
    for (const dim of [1, 2, 3, 4, 5, 7, 15, 16, 17, 31, 63, 64, 65, 127, 128, 767, 768]) {
      const n = 64;
      const rows = randomRows(n, dim, dim);
      const q = randomRows(1, dim, dim + 1000);
      const kernel = loadSimdKernel(rows, dim, n)!;
      expect(kernel, `dim ${dim}`).not.toBeNull();
      kernel.setQuery(q);
      for (let i = 0; i < n; i++) expect(Math.abs(kernel.toRow(i) - dot(q, 0, rows, i * dim, dim)), `dim ${dim} row ${i}`).toBeLessThan(1e-5);
    }
  });

  it('scans every row to exactly what it returns one row at a time', () => {
    const dim = 100;
    const n = 500;
    const rows = randomRows(n, dim, 3);
    const kernel = loadSimdKernel(rows, dim, n)!;
    kernel.setQuery(randomRows(1, dim, 4));
    const all = new Float32Array(n);
    kernel.scan(n, all);
    for (let i = 0; i < n; i++) expect(all[i]).toBe(Math.fround(kernel.toRow(i)));
    // A shorter scan fills only what was asked for.
    const some = new Float32Array(n).fill(9);
    kernel.scan(10, some);
    expect([...some.subarray(0, 10)]).toEqual([...all.subarray(0, 10)]);
    expect(some[10]).toBe(9);
  });

  it('does not disturb its own copy when the caller changes the query or the rows', () => {
    const dim = 16;
    const rows = randomRows(8, dim, 5);
    const q = randomRows(1, dim, 6);
    const kernel = loadSimdKernel(rows, dim, 8)!;
    kernel.setQuery(q);
    const before = kernel.toRow(3);
    q.fill(0);
    rows.fill(0);
    expect(kernel.toRow(3)).toBe(before);
  });

  it('declines when there is nothing to load or WebAssembly is missing', () => {
    expect(loadSimdKernel(new Float32Array(0), 4, 0)).toBeNull();
    expect(loadSimdKernel(new Float32Array(8), 4, 3)).toBeNull(); // fewer floats than rows * dim
    vi.stubGlobal('WebAssembly', undefined);
    try {
      expect(simdAvailable()).toBe(false);
      expect(loadSimdKernel(randomRows(4, 8, 1), 8, 4)).toBeNull();
    } finally {
      vi.unstubAllGlobals();
    }
  });
});

describe('searching with the kernel', () => {
  const dim = 96;
  const n = 4000;
  const nq = 60;
  // Data and queries drawn together, so the queries come from the same clusters as the data.
  const all = clusteredVectors(n + nq, dim, 24, 0.35, 11).data;
  const data = all.subarray(0, n * dim);
  const query = (i: number) => all.subarray((n + i) * dim, (n + i + 1) * dim);

  it('leaves the graph itself untouched: same bytes with or without a kernel', () => {
    const plain = Hnsw.build(data, dim, undefined, 1500);
    const withKernel = new Hnsw(data, dim);
    withKernel.useKernel(loadSimdKernel(data, dim, 1500));
    withKernel.extend(1500);
    expect(withKernel.kernelName).toBe('wasm-simd');
    expect(Buffer.from(withKernel.toBytes()).equals(Buffer.from(plain.toBytes()))).toBe(true);
  });

  it('finds the same neighbours as the TypeScript loop, exactly and through the graph', () => {
    const index = Hnsw.build(data, dim, undefined, n);
    const kernel = loadSimdKernel(data, dim, n)!;
    let exactSame = 0;
    let graphOverlap = 0;
    let recallJs = 0;
    let recallSimd = 0;
    for (let qi = 0; qi < nq; qi++) {
      const q = query(qi);
      index.useKernel(null);
      expect(index.kernelName).toBe('js');
      const truth = exactTopK(data, dim, n, q, 10).map((h) => h.id);
      const jsGraph = index.search(q, 10, 64).map((h) => h.id);
      index.useKernel(kernel);
      const simdExact = index.exact(q, 10).map((h) => h.id);
      const simdGraph = index.search(q, 10, 64).map((h) => h.id);
      if (simdExact.join() === truth.join()) exactSame++;
      graphOverlap += simdGraph.filter((id) => jsGraph.includes(id)).length;
      recallJs += jsGraph.filter((id) => truth.includes(id)).length;
      recallSimd += simdGraph.filter((id) => truth.includes(id)).length;
    }
    // Exact results may differ only where two rows tie to within rounding; that should be rare.
    expect(exactSame / nq).toBeGreaterThanOrEqual(0.95);
    expect(graphOverlap / (nq * 10)).toBeGreaterThanOrEqual(0.99);
    expect(recallSimd / (nq * 10)).toBeGreaterThanOrEqual(0.96);
    expect(Math.abs(recallSimd - recallJs) / (nq * 10)).toBeLessThan(0.01);
  });

  it('honours a filter and a trace, and steps aside when the index outgrows it', () => {
    const index = Hnsw.build(data, dim, undefined, 2000);
    const small = loadSimdKernel(data, dim, 1000)!; // covers only half the index
    index.useKernel(small);
    expect(index.kernelName).toBe('js'); // ignored: it does not hold every row
    const kernel = loadSimdKernel(data, dim, 2000)!;
    index.useKernel(kernel);
    const q = query(0);
    const even = (id: number) => id % 2 === 0;
    const trace: [number, number, number, number][] = [];
    const hits = index.search(q, 10, 64, { allow: even, trace });
    expect(hits.length).toBe(10);
    for (const h of hits) expect(h.id % 2).toBe(0);
    expect(trace.length).toBeGreaterThan(10);
    const exact = index.exact(q, 10, even);
    expect(exact.map((h) => h.id)).toEqual(exactTopK(data, dim, 2000, q, 10, even).map((h) => h.id));
    for (const h of exact) expect(Math.abs(h.sim - dot(q, 0, data, h.id * dim, dim))).toBeLessThan(1e-5);
  });
});
