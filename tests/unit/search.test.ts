import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { Engine } from '../../src/engine.ts';
import { dot } from '../../src/index/hnsw.ts';
import { calibrate } from '../../src/index/simstats.ts';
import { buildDemoStore, hashEmbedder } from '../../src/prove/demo.ts';
import { DIVERSIFY_DEFAULT, nearCopyThreshold } from '../../src/search/core.ts';
import { pickDiverse, redundancy } from '../../src/search/diverse.ts';

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'loci-search-test-'));
afterAll(() => fs.rmSync(tmp, { recursive: true, force: true }));

/** Items 0, 1 and 3 are near-copies of each other; 2 and 4 are unlike everything. */
const group = [0, 0, 1, 0, 2];
const sim = (a: number, b: number) => (a === b ? 1 : group[a] === group[b] ? 0.95 : 0.1);

describe('folding near-copies', () => {
  it('keeps one of each set of near-copies and reorders nothing else', () => {
    const out = pickDiverse([0, 1, 2, 3, 4], 3, sim, { dupSim: 0.9 });
    expect(out.picked).toEqual([0, 2, 4]);
    // Of the plain top three (0, 1, 2), one was left out as a copy.
    expect(out.folded).toBe(1);
  });

  it('changes nothing when there are no near-copies', () => {
    const flat = (a: number, b: number) => (a === b ? 1 : 0.2);
    const out = pickDiverse([7, 3, 9, 1, 5], 4, flat, { dupSim: 0.9 });
    expect(out.picked).toEqual([7, 3, 9, 1]);
    expect(out.folded).toBe(0);
  });

  it('never changes the top result and only returns candidates it was given', () => {
    for (const lambda of [1, 0.7, 0.3]) {
      const ranked = [4, 0, 1, 3, 2];
      const out = pickDiverse(ranked, 4, sim, { dupSim: 0.9, lambda });
      expect(out.picked[0]).toBe(4);
      expect(new Set(out.picked).size).toBe(out.picked.length);
      for (const id of out.picked) expect(ranked).toContain(id);
    }
  });

  it('fills the list with the best near-copies rather than returning a short one', () => {
    const same = () => 0.99;
    const out = pickDiverse([5, 6, 7, 8], 3, same, { dupSim: 0.9 });
    expect(out.picked).toEqual([5, 6, 7]);
    // Nothing was displaced: the list is the plain top three.
    expect(out.folded).toBe(0);
  });

  it('handles empty input and a request for nothing', () => {
    expect(pickDiverse([], 5, sim, { dupSim: 0.9 })).toEqual({ picked: [], folded: 0 });
    expect(pickDiverse([1, 2], 0, sim, { dupSim: 0.9 })).toEqual({ picked: [], folded: 0 });
  });

  it('spread trades rank for variety, which fold alone does not', () => {
    // 0 and 1 are alike but below the near-copy line; 2 is unlike both and ranked last.
    const s = (a: number, b: number) => (a === b ? 1 : a + b === 1 ? 0.8 : 0);
    expect(pickDiverse([0, 1, 2], 2, s, { dupSim: 0.9 }).picked).toEqual([0, 1]);
    expect(pickDiverse([0, 1, 2], 2, s, { dupSim: 0.9, lambda: 0.3 }).picked).toEqual([0, 2]);
  });

  it('measures repetition in a list', () => {
    const r = redundancy([0, 1, 2], sim, 0.9);
    expect(r.dupPairs).toBe(1);
    expect(r.meanSim).toBeCloseTo((0.95 + 0.1 + 0.1) / 3, 10);
    expect(redundancy([3], sim, 0.9)).toEqual({ meanSim: 0, dupPairs: 0 });
  });
});

describe('search with folding, on the readable demo store', () => {
  let engine: Engine;
  beforeAll(async () => {
    const cfg = await buildDemoStore(path.join(tmp, 'demo'), 3000, 7, { prose: true });
    engine = Engine.open(cfg, { embedder: hashEmbedder(), save: false });
  });
  afterAll(() => engine.close());

  const simOf = (a: number, b: number) => dot(engine.vectors, a * engine.dim, engine.vectors, b * engine.dim, engine.dim);

  it('sets the near-copy threshold from the corpus, the same way every time', () => {
    const t = nearCopyThreshold(engine);
    expect(t.tau).toBeGreaterThanOrEqual(0.7);
    expect(t.tau).toBeLessThanOrEqual(0.97);
    expect(t.tau).toBeGreaterThan(t.median + 0.3);
    expect(nearCopyThreshold(engine)).toBe(t); // cached
    expect(calibrate(engine.vectors, engine.dim, engine.n, engine.sessionPk, 5, 17, 20_000).tau).toBe(t.tau);
  });

  it('leaves the plain ranking exactly as it was when folding is off', async () => {
    for (const mode of ['hybrid', 'dense', 'bm25', 'recent'] as const) {
      const plain = await engine.search('sourdough starter hydration', { k: 12, mode });
      expect(plain.folded).toBe(0);
      // With a threshold nothing can reach, folding must return the same list.
      const again = await engine.search('sourdough starter hydration', { k: 12, mode, diversify: undefined });
      expect(again.hits.map((h) => h.node)).toEqual(plain.hits.map((h) => h.node));
    }
  });

  it('removes repetition without leaving the topic or changing the top result', async () => {
    const { tau } = nearCopyThreshold(engine);
    let plainPairs = 0;
    let foldedPairs = 0;
    for (const [query, topic] of [
      ['sourdough starter hydration', 'kitchen'],
      ['why does the chord detector hear minor as major', 'music'],
      ['socket leak on failed handshake', 'network'],
      ['satellite orbit', 'orbit'],
    ] as const) {
      const plain = await engine.search(query, { k: 12, mode: 'hybrid' });
      const fold = await engine.search(query, { k: 12, mode: 'hybrid', diversify: DIVERSIFY_DEFAULT });
      expect(fold.hits).toHaveLength(12);
      expect(fold.hits[0]!.node).toBe(plain.hits[0]!.node);
      const onTopic = fold.hits.filter((h) => engine.projectName(engine.project[h.node]!) === topic).length;
      expect(onTopic, query).toBeGreaterThanOrEqual(11);
      plainPairs += redundancy(plain.hits.map((h) => h.node), simOf, tau).dupPairs;
      foldedPairs += redundancy(fold.hits.map((h) => h.node), simOf, tau).dupPairs;
      // Nothing is promoted from far down the ranking.
      const pool = await engine.search(query, { k: 36, mode: 'hybrid' });
      const allowed = new Set(pool.hits.map((h) => h.node));
      for (const h of fold.hits) expect(allowed.has(h.node)).toBe(true);
    }
    expect(plainPairs).toBeGreaterThan(0);
    expect(foldedPairs).toBeLessThan(plainPairs * 0.6);
  });

  it('works in every mode and with a project filter', async () => {
    const projectId = engine.projects[0]!.id;
    for (const mode of ['hybrid', 'dense', 'bm25', 'recent'] as const) {
      const res = await engine.search('add a test for the case', { k: 10, mode, projectId, diversify: DIVERSIFY_DEFAULT });
      expect(res.hits.length).toBeGreaterThan(0);
      for (const h of res.hits) expect(engine.project[h.node]).toBe(projectId);
      expect(new Set(res.hits.map((h) => h.node)).size).toBe(res.hits.length);
    }
  });
});
