import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { afterAll, describe, expect, it } from 'vitest';
import { Engine } from '../../src/engine.ts';
import { kmeans } from '../../src/index/kmeans.ts';
import { buildDemoStore, hashEmbedder } from '../../src/prove/demo.ts';
import { clusteredVectors, SYNTH_TOPICS } from '../../src/prove/synth.ts';
import { buildViewerData } from '../../src/serve/export.ts';
import { labelClusters, sentenceCount } from '../../src/serve/labels.ts';

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'loci-labels-test-'));
afterAll(() => fs.rmSync(tmp, { recursive: true, force: true }));

describe('kmeans', () => {
  const N = 900;
  const { data, labels } = clusteredVectors(N, 3, 5, 0.08, 4);

  it('recovers well-separated groups', () => {
    const out = kmeans(data, N, 3, 5);
    expect([...out.sizes].reduce((a, b) => a + b, 0)).toBe(N);
    // Each true group should map almost entirely onto one found cluster.
    const table = new Map<string, number>();
    for (let i = 0; i < N; i++) table.set(`${labels[i]}:${out.assign[i]}`, (table.get(`${labels[i]}:${out.assign[i]}`) ?? 0) + 1);
    let agree = 0;
    for (let g = 0; g < 5; g++) agree += Math.max(...Array.from({ length: 5 }, (_, c) => table.get(`${g}:${c}`) ?? 0));
    expect(agree / N).toBeGreaterThan(0.97);
    expect(out.iterations).toBeLessThan(40);
  });

  it('is deterministic for a seed and handles edge cases', () => {
    expect(kmeans(data, N, 3, 5, { seed: 3 })).toEqual(kmeans(data, N, 3, 5, { seed: 3 }));
    expect(kmeans(new Float32Array(0), 0, 3, 4).assign).toHaveLength(0);
    const few = kmeans(data.subarray(0, 6), 2, 3, 10); // more clusters asked for than points
    expect(few.sizes).toHaveLength(2);
    const same = kmeans(new Float32Array(30).fill(0.5), 10, 3, 3); // identical points
    expect([...same.sizes].reduce((a, b) => a + b, 0)).toBe(10);
    expect([...same.centroids].every(Number.isFinite)).toBe(true);
  });
});

describe('labelClusters', () => {
  it('names a cluster by words common inside it and rare outside', () => {
    const texts = [
      ...Array.from({ length: 30 }, (_, i) => `the tokenizer and the grammar need a fix, attempt ${i}`),
      ...Array.from({ length: 30 }, (_, i) => `the sourdough starter and the oven need a fix, attempt ${i}`),
    ];
    const assign = texts.map((_, i) => (i < 30 ? 0 : 1));
    const [a, b] = labelClusters(texts, assign, 2);
    expect(a!.words.sort()).toEqual(['grammar', 'tokenizer']);
    expect(b!.words).toHaveLength(2);
    expect(b!.words.every((w) => ['sourdough', 'starter', 'oven'].includes(w))).toBe(true);
    // Words shared by both clusters, stop words and bare numbers never appear.
    for (const l of [a!, b!]) for (const w of l.words) expect(['the', 'and', 'need', 'fix', 'attempt']).not.toContain(w);
    expect(a!.size).toBe(30);
  });

  it('gives no words rather than weak ones, and skips near-duplicates', () => {
    const same = Array.from({ length: 20 }, () => 'identical text everywhere about nothing specific');
    for (const l of labelClusters(same, same.map((_, i) => i % 2), 2)) expect(l.words).toEqual([]);
    const texts = [...Array.from({ length: 20 }, () => 'parse parser parsing compiler'), ...Array.from({ length: 20 }, () => 'bread flour water salt')];
    const [first] = labelClusters(texts, texts.map((_, i) => (i < 20 ? 0 : 1)), 2, 3);
    expect(first!.words.filter((w) => w.startsWith('pars'))).toHaveLength(1);
  });

  it('counts sentences roughly and within bounds', () => {
    expect(sentenceCount('One. Two! Three?')).toBe(3);
    expect(sentenceCount('no terminator at all')).toBe(1);
    expect(sentenceCount('line one\nline two\nline three')).toBe(3);
    expect(sentenceCount('a. '.repeat(50))).toBe(10);
    expect(sentenceCount('')).toBe(1);
  });
});

describe('viewer data', () => {
  it('ships consistent buffers and labels that name the demo topics', async () => {
    const cfg = await buildDemoStore(path.join(tmp, 'demo'), 3200, 6);
    const engine = Engine.open(cfg, { embedder: hashEmbedder(), save: false });
    const data = buildViewerData(engine, { demo: true, model: cfg.embedModel });
    const n = data.manifest.n;
    expect(n).toBe(3200);
    expect(data.manifest.clusters.length).toBe(10); // about one region per 320 chunks
    expect(Math.max(...data.sentences)).toBeGreaterThan(1); // prose-shaped text yields dust
    expect(data.pos).toHaveLength(n * 3);
    expect(data.home).toHaveLength(n * 3);
    expect(data.knn).toHaveLength(n * data.manifest.knn);
    expect(data.meta).toHaveLength(n * 3);
    expect(data.sentences).toHaveLength(n);
    expect(data.cluster).toHaveLength(n);
    expect(Math.max(...data.cluster)).toBeLessThan(data.manifest.clusters.length);
    expect(Math.min(...data.sentences)).toBeGreaterThanOrEqual(1);
    expect(data.manifest.clusters.reduce((a, c) => a + c.size, 0)).toBe(n);
    // Particles are in time order.
    for (let r = 1; r < n; r++) expect(data.meta[r * 3]!).toBeGreaterThanOrEqual(data.meta[(r - 1) * 3]!);

    // Each demo topic has its own vocabulary. Every label word must be a topic word, and most
    // regions should be named from a single topic (a region that straddles two topics is
    // honestly labelled with a word from each).
    const vocab: Record<string, string[]> = {
      orbit: ['satellite', 'ephemeris', 'kepler', 'perigee', 'propagate', 'orbit', 'tle', 'apogee', 'inclination', 'sgp4'],
      kitchen: ['sourdough', 'starter', 'hydration', 'crumb', 'proof', 'oven', 'flour', 'levain', 'bake', 'crust'],
      parser: ['tokenizer', 'grammar', 'ast', 'lexer', 'recursive', 'descent', 'precedence', 'parse', 'syntax', 'node'],
      garden: ['tomato', 'compost', 'mulch', 'seedling', 'prune', 'trellis', 'soil', 'harvest', 'water', 'bed'],
      network: ['socket', 'latency', 'packet', 'handshake', 'retry', 'timeout', 'backoff', 'stream', 'buffer', 'tcp'],
      music: ['chord', 'tempo', 'reverb', 'melody', 'synth', 'octave', 'rhythm', 'mix', 'bass', 'scale'],
    };
    expect(Object.keys(vocab).sort()).toEqual([...SYNTH_TOPICS].sort());
    const labelled = data.manifest.clusters.filter((c) => c.words.length > 0);
    expect(labelled.length).toBeGreaterThanOrEqual(4);
    let pure = 0;
    for (const c of labelled) {
      const topics = new Set(c.words.map((w) => Object.keys(vocab).find((t) => vocab[t]!.includes(w))));
      expect(topics.has(undefined), `label "${c.words.join(' ')}" uses a non-topic word`).toBe(false);
      if (topics.size === 1) pure++;
    }
    expect(pure / labelled.length).toBeGreaterThanOrEqual(0.7);
    engine.close();
  });
});
