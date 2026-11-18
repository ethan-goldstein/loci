import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { Engine } from '../../src/engine.ts';
import { benchIndex } from '../../src/prove/bench.ts';
import { clusterBootstrap, macroMean, percentile } from '../../src/prove/bootstrap.ts';
import { buildDemoStore, hashEmbedder } from '../../src/prove/demo.ts';
import { MIN_PROJECT_QUERIES, maskText, randomExpectation, runEval, selectQueries } from '../../src/prove/eval.ts';
import type { EvalReport } from '../../src/prove/eval.ts';
import { renderResults } from '../../src/prove/report.ts';
import { clusteredVectors, rng } from '../../src/prove/synth.ts';

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'loci-eval-test-'));
afterAll(() => fs.rmSync(tmp, { recursive: true, force: true }));

describe('statistics', () => {
  it('random expectation matches simulation', () => {
    const rand = rng(4);
    for (const [eligible, relevant] of [
      [50, 5],
      [200, 60],
      [30, 1],
      [12, 12],
    ] as const) {
      let p = 0;
      let rr = 0;
      const trials = 20_000;
      for (let t = 0; t < trials; t++) {
        // Draw 10 without replacement; items below `relevant` are the relevant ones.
        const pool = Array.from({ length: eligible }, (_, i) => i);
        let first = 0;
        let hits = 0;
        for (let k = 0; k < 10; k++) {
          const j = k + Math.floor(rand() * (eligible - k));
          [pool[k], pool[j]] = [pool[j]!, pool[k]!];
          if (pool[k]! < relevant) {
            hits++;
            if (!first) first = k + 1;
          }
        }
        p += hits / 10;
        rr += first ? 1 / first : 0;
      }
      const exact = randomExpectation(eligible, relevant);
      expect(exact.p).toBeCloseTo(p / trials, 2);
      expect(exact.rr).toBeCloseTo(rr / trials, 2);
    }
  });

  it('macro mean weighs groups equally and the bootstrap brackets the estimate', () => {
    const items = [
      ...Array.from({ length: 100 }, () => ({ g: 1, v: 1 })),
      { g: 2, v: 0 }, // one item, but half the macro mean
    ];
    expect(macroMean(items, (x) => x.g, (x) => x.v)).toBe(0.5);
    expect(macroMean([], () => 0, () => 0)).toBeNaN();

    const rand = rng(8);
    // Two strata, ten clusters each, values around 0.3 and 0.7.
    const strata = [0.3, 0.7].map((mu) => Array.from({ length: 10 }, () => Array.from({ length: 5 }, () => ({ v: mu + (rand() - 0.5) * 0.4 }))));
    const mean = (xs: readonly { v: number }[]) => xs.reduce((a, x) => a + x.v, 0) / xs.length;
    const ci = clusterBootstrap(strata, mean, 1000, 3);
    expect(ci.lo).toBeLessThan(ci.value);
    expect(ci.hi).toBeGreaterThan(ci.value);
    expect(ci.value).toBeCloseTo(0.5, 1);
    expect(ci.hi - ci.lo).toBeLessThan(0.2);
    expect(clusterBootstrap(strata, mean, 200, 3)).toEqual(clusterBootstrap(strata, mean, 200, 3)); // seeded
    // A single cluster per stratum has no spread to resample.
    const single = clusterBootstrap([[[{ v: 1 }, { v: 3 }]]], mean, 100);
    expect([single.lo, single.value, single.hi]).toEqual([2, 2, 2]);
    expect(percentile([1, 2, 3, 4], 0.5)).toBe(2);
  });

  it('masks project names and paths', () => {
    const out = maskText('In acme-field-rover I edited ~/code/Acme_Field Rover/src/main.py and ./lib/x.ts for the Storefront site', ['acme-field-rover', 'Storefront', 'ab']);
    expect(out).not.toMatch(/acme.?field.?rover/i);
    expect(out).not.toContain('Storefront');
    expect(out).not.toContain('main.py');
    expect(out).toContain('PROJECT');
    expect(out).toContain('PATH');
    expect(maskText('a table about abacus', ['ab'])).toBe('a table about abacus'); // names under 3 characters are ignored
  });
});

describe('eval harness on the demo store', () => {
  let engine: Engine;
  let report: EvalReport;
  beforeAll(async () => {
    const cfg = await buildDemoStore(path.join(tmp, 'demo'), 1500, 5);
    engine = Engine.open(cfg, { embedder: hashEmbedder(), save: false });
    report = await runEval(engine, { embedder: hashEmbedder(), maxQueries: 180 });
  });
  afterAll(() => engine.close());

  it('only asks questions that have an earlier answer elsewhere', () => {
    const queries = selectQueries(engine);
    expect(queries.length).toBeGreaterThan(300);
    for (const q of queries.slice(0, 50)) {
      expect(engine.role[q.node]).toBe(0); // a prompt someone typed
      expect(q.relevant).toBeGreaterThan(0);
      expect(q.eligible).toBeGreaterThanOrEqual(10);
    }
  });

  it('is a positive control: where retrieval should clearly win, it says so', () => {
    // Demo topics are separable by vocabulary and shuffled in time, so retrieval should be near
    // perfect and recency near chance. If the harness could not show a win here, it could not
    // be trusted when it shows a loss on real data.
    const get = (name: string) => report.systems.find((s) => s.name === name)!;
    expect(report.queries).toBe(180);
    expect(report.projects).toBe(6);
    expect(get('random').precision.value).toBeGreaterThan(0.1);
    expect(get('random').precision.value).toBeLessThan(0.25);
    expect(get('recency').precision.value).toBeLessThan(0.3);
    for (const name of ['bm25', 'dense', 'hybrid']) {
      expect(get(name).precision.value).toBeGreaterThan(0.9);
      expect(get(name).liftOverRecency!.precision.lo).toBeGreaterThan(0.5); // the whole interval is a clear win
    }
    expect(get('recency').liftOverRecency).toBeNull();
    for (const s of report.systems) {
      expect(s.precision.lo).toBeLessThanOrEqual(s.precision.value + 1e-9);
      expect(s.precision.hi).toBeGreaterThanOrEqual(s.precision.value - 1e-9);
      expect(s.mrr.value).toBeGreaterThanOrEqual(0);
      expect(s.mrr.value).toBeLessThanOrEqual(1);
    }
  });

  it('reports the comparison under three ways of averaging', () => {
    expect(report.minProjectQueries).toBe(MIN_PROJECT_QUERIES);
    expect(report.robustness.map((v) => v.label.split(',')[0])).toEqual(['macro', 'macro', 'micro']);
    // Every demo project has plenty of queries, so the floor removes nothing here.
    expect(report.headlineQueries).toBe(report.queries);
    expect(report.headlineProjects).toBe(6);
    for (const view of report.robustness) {
      expect(view.systems.map((s) => s.name)).toEqual(['recency', 'hybrid', 'hybrid+recency']);
      expect(view.systems[0]!.lift).toBeNull();
      expect(view.systems[1]!.lift!.lo).toBeGreaterThan(0);
    }
    expect(report.knownItem!.systems.find((s) => s.name === 'dense')!.hit10).toBeGreaterThan(report.knownItem!.randomHit10);
  });

  it('renders a report with numbers only', () => {
    const md = renderResults(null, report, '2026-01-01');
    expect(md).toContain('## 2. Does retrieval find the right earlier work?');
    expect(md).toContain('### How stable is that verdict?');
    expect(md).toContain('Hybrid beats recency under all three.');
    expect(md).toContain('| P1 |');
    // Project names and conversation text never appear: projects are P1, P2, ...
    for (const topic of ['kitchen', 'sourdough', 'orbit', 'satellite', 'parser', 'tokenizer']) expect(md.toLowerCase()).not.toContain(topic);
    expect(renderResults(null, null, 'x')).toContain('# LOCI results');
  });

  it('does not call a mixed result a win or a loss', () => {
    // Same sign everywhere, but only one view clears the noise: "probably behind", not settled.
    const mixed = structuredClone(report);
    const lifts = [
      { value: -0.13, lo: -0.2, hi: -0.08 },
      { value: -0.02, lo: -0.08, hi: 0.02 },
      { value: -0.08, lo: -0.34, hi: 0.11 },
    ];
    mixed.robustness.forEach((v, i) => (v.systems[1]!.lift = lifts[i]!));
    const md = renderResults(null, mixed, 'x');
    expect(md).toContain('below recency in all 3 views, but the gap is outside the noise in only 1 of them');
    expect(md).toContain('"probably behind", not as settled');
    // Opposite signs: unsettled.
    mixed.robustness[1]!.systems[1]!.lift = { value: 0.05, lo: -0.01, hi: 0.09 };
    expect(renderResults(null, mixed, 'x')).toContain('direction of the difference changes');
  });
});

describe('index benchmark', () => {
  it('measures recall against exact search and reports a consistent sweep', () => {
    const all = clusteredVectors(2200, 32, 12, 0.3, 21).data;
    const b = benchIndex('test', all.subarray(0, 2000 * 32), 32, 2000, all.subarray(2000 * 32), 200);
    expect(b.n).toBe(2000);
    expect(b.sweep.map((s) => s.ef)).toEqual([16, 32, 64, 128, 256]);
    for (let i = 1; i < b.sweep.length; i++) expect(b.sweep[i]!.recall10).toBeGreaterThanOrEqual(b.sweep[i - 1]!.recall10 - 0.005);
    expect(b.sweep.at(-1)!.recall10).toBeGreaterThan(0.99);
    expect(b.indexBytes).toBeGreaterThan(0);
    expect(b.exactP50Ms).toBeGreaterThan(0);
    const md = renderResults({ machine: { cpu: 'test', cores: 1, memoryGb: 1, node: 'v0', platform: 'x' }, params: { M: 16, M0: 32, efConstruction: 200, seed: 1 }, real: b, synthetic: null, bm25: { docs: 1, vocabulary: 1, buildMs: 1, p50Ms: 1, p95Ms: 1, ftsQueries: 1, ftsTop10Agreement: 1 }, rssMb: 1 }, null, 'x');
    expect(md).toContain('## 1. Is the index correct, and how fast is it?');
    expect(md).toContain('| 64 |');

    // The same index searched with each inner loop, measured in interleaved rounds.
    const k = b.kernels!;
    expect(k).toBeTruthy();
    expect(k.rows.map((r) => r.name)).toEqual(['js', 'wasm-simd']);
    expect(k.rounds).toBeGreaterThanOrEqual(3);
    expect(k.exactTop10Agreement).toBeGreaterThanOrEqual(0.95);
    expect(k.maxAbsDiff).toBeLessThan(1e-5);
    for (const row of k.rows) {
      expect(row.exactP50Ms).toBeGreaterThan(0);
      expect(row.sweep.map((x) => x.ef)).toEqual([16, 32, 64, 128, 256]);
    }
    // Whichever loop does the arithmetic, the graph finds the same neighbours to within a hair.
    k.rows[0]!.sweep.forEach((x, i) => expect(Math.abs(x.recall10 - k.rows[1]!.sweep[i]!.recall10)).toBeLessThan(0.01));
    expect(md).toContain('TypeScript against WebAssembly SIMD');
    expect(md).toContain('| exact scan |');
    expect(md).not.toContain('with no SIMD');
    // A report written before the kernel existed still renders, and says so.
    const old = renderResults({ machine: { cpu: 'test', cores: 1, memoryGb: 1, node: 'v0', platform: 'x' }, params: { M: 16, M0: 32, efConstruction: 200, seed: 1 }, real: { ...b, kernels: undefined }, synthetic: null, bm25: { docs: 1, vocabulary: 1, buildMs: 1, p50Ms: 1, p95Ms: 1, ftsQueries: 1, ftsTop10Agreement: 1 }, rssMb: 1 }, null, '2026-01-01');
    expect(old).toContain('with no SIMD');
    expect(old).not.toContain('WebAssembly SIMD p50');
  });
});
