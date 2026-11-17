import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import type { Config } from '../../src/config.ts';
import { Engine } from '../../src/engine.ts';
import { sha256 } from '../../src/ingest/run.ts';
import { Store } from '../../src/ingest/store.ts';
import { appendRows, prepareVecFile } from '../../src/index/vecfile.ts';
import { calibrate, candidateSentences, classify, draftRules, isCandidate, leaderClusters, neighboursAbove, normalTail, runRepeats, splitSentences, templatedPrompts } from '../../src/insight/repeats.ts';
import type { RepeatsReport } from '../../src/insight/repeats.ts';
import { DEMO_MODEL, PLANTED, buildDemoStore, hashEmbedder } from '../../src/prove/demo.ts';
import { renderResults } from '../../src/prove/report.ts';
import { clusteredVectors } from '../../src/prove/synth.ts';
import { repeatsView } from '../../src/serve/http.ts';
import { buildViewerData } from '../../src/serve/export.ts';

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'loci-repeats-test-'));
afterAll(() => fs.rmSync(tmp, { recursive: true, force: true }));

describe('sentence handling', () => {
  it('splits prompts into sentences and strips list markers and quotes', () => {
    expect(splitSentences('Fix the parser. Then run the tests!\n- no tabs please\n2) "keep it short"')).toEqual([
      'Fix the parser.',
      'Then run the tests!',
      'no tabs please',
      'keep it short',
    ]);
    expect(splitSentences('   \n\n ')).toEqual([]);
  });

  it('keeps readable instructions and drops code, logs, paths and empty go-aheads', () => {
    for (const keep of ['run the unit tests before you commit', 'no tabs, plain wording.', 'Make the table easier to read on phones', 'use short names.']) {
      expect(isCandidate(keep), keep).toBe(true);
    }
    for (const drop of [
      'ok go ahead', // nothing but filler
      'yes please do that now', // still nothing but filler
      'const x = compute(a, b);', // code
      'at Object.<anonymous> (/srv/app/index.js:12:5)', // stack trace
      'open src/index/hnsw.ts and look', // a path
      'see https://example.com/docs for more', // a link
      'token is [REDACTED:vendor-key] here ok', // redacted material
      '12345 67890 11111 22222', // not prose
      'short one', // too short
      `${'word '.repeat(60)}end`, // too long
    ]) {
      expect(isCandidate(drop), drop).toBe(false);
    }
  });

  it('sorts sentences into kinds by simple word patterns', () => {
    expect(classify('no semicolons, plain wording.')).toBe('instruction');
    expect(classify('Always run the tests before committing')).toBe('instruction');
    expect(classify('can you add tests for this?')).toBe('instruction'); // a request, despite the question mark
    expect(classify('please dont use emojis in commit messages')).toBe('instruction');
    expect(classify("let's update the changelog")).toBe('task');
    expect(classify('now release planning')).toBe('task');
    expect(classify('why is the build failing again?')).toBe('question');
    expect(classify('do we need a cache here?')).toBe('question');
    expect(classify('Do not touch the config files in this pass.')).toBe('instruction'); // an order, not a question
    expect(classify('the header looks great on mobile')).toBe('statement');
  });
});

describe('threshold and grouping', () => {
  it('calibrates from unrelated pairs and is not dragged up by the repeats themselves', () => {
    // 600 unrelated unit vectors, plus one tight family of 60 near-copies in different sessions.
    const dim = 128; // high enough that two unrelated vectors essentially never look alike
    const base = clusteredVectors(600, dim, 600, 1.0, 5).data; // one "cluster" per point: unrelated
    const family = clusteredVectors(60, dim, 1, 0.02, 9).data;
    const all = new Float32Array((600 + 60) * dim);
    all.set(base);
    all.set(family, 600 * dim);
    const sessionOf = Int32Array.from({ length: 660 }, (_, i) => i); // every sentence its own session
    const t = calibrate(all, dim, 660, sessionOf, 5);
    expect(Math.abs(t.median)).toBeLessThan(0.1); // unrelated high-dimensional vectors are near-orthogonal
    expect(t.sigma).toBeGreaterThan(0.04);
    expect(t.sigma).toBeLessThan(0.2);
    expect(t.tau).toBeGreaterThanOrEqual(0.7); // never below the floor
    expect(t.tau).toBeLessThanOrEqual(0.97);

    const { adjacency, above, crossSession } = neighboursAbove(all, dim, 660, t.tau, sessionOf);
    expect(crossSession).toBe((660 * 659) / 2);
    // Only the planted family matches: 60 choose 2 pairs, and nothing among the unrelated vectors.
    expect(above).toBe((60 * 59) / 2);
    const clusters = leaderClusters(adjacency);
    expect(clusters[0]).toHaveLength(60);
    expect(clusters.filter((c) => c.length > 1)).toHaveLength(1);
    expect(new Set(clusters.flat()).size).toBe(660); // every sentence lands in exactly one group
  });

  it('ignores same-session pairs when calibrating and counting', () => {
    const dim = 16;
    const v = clusteredVectors(40, dim, 1, 0.01, 2).data; // all near-identical
    const oneSession = new Int32Array(40); // all in session 0
    expect(calibrate(v, dim, 40, oneSession, 5).sampledPairs).toBe(0);
    const { above, crossSession, adjacency } = neighboursAbove(v, dim, 40, 0.9, oneSession);
    expect([above, crossSession]).toEqual([0, 0]);
    expect(adjacency[0]).toHaveLength(39); // still neighbours, just not counted as cross-session
  });

  it('computes the normal tail', () => {
    expect(normalTail(0)).toBeCloseTo(0.5, 6);
    expect(normalTail(1.96)).toBeCloseTo(0.025, 3);
    expect(normalTail(5)).toBeCloseTo(2.87e-7, 8);
    expect(normalTail(-1)).toBeCloseTo(0.8413, 3);
  });
});

/** A small hand-built store: `rows` are prompts as [session, day, text]. */
async function storeOf(name: string, rows: [session: string, day: number, text: string][]): Promise<Engine> {
  const dir = path.join(tmp, name);
  fs.mkdirSync(dir, { recursive: true });
  const cfg: Config = { home: dir, source: '', ollamaUrl: 'http://127.0.0.1:11434', embedModel: DEMO_MODEL, port: 4319 };
  const store = new Store(path.join(dir, 'loci.db'));
  const texts: string[] = [];
  store.transaction(() => {
    const project = store.upsertProject('p', 'project-one');
    rows.forEach(([session, day, text], i) => {
      const ts = Date.UTC(2026, 0, 1 + day, 12, 0, i);
      const pk = store.upsertSession(session, project, null, ts, ts);
      store.insertChunk({ uuid: `u-${i}`, part: 0, sessionPk: pk, projectId: project, role: 'human', ts, text, sha: sha256(`${i}${text}`) });
      texts.push(text);
    });
  });
  store.close();
  const embedder = hashEmbedder();
  const vecs = await embedder.embed(texts, 'document');
  const file = path.join(dir, `vectors.${DEMO_MODEL}.f32`);
  prepareVecFile(file, vecs[0]!.length, DEMO_MODEL);
  const flat = new Float32Array(vecs.length * vecs[0]!.length);
  vecs.forEach((v, i) => flat.set(v, i * v.length));
  appendRows(file, flat);
  return Engine.open(cfg, { embedder, save: false });
}

describe('runRepeats on a hand-built store', () => {
  // Unrelated prompts must look unrelated, as real ones do: each is built from its own invented
  // words, so two of them share nothing. (Prompts that all share a template would raise the
  // typical similarity, and with it the threshold.)
  const syllables = ['ba', 'ke', 'mi', 'lo', 'tu', 'ra', 'ne', 'so', 'vi', 'da', 'pu', 'ze'];
  const word = (k: number) => syllables[k % 12]! + syllables[Math.floor(k / 12) % 12]! + syllables[Math.floor(k / 144) % 12]!;
  const filler = (i: number) => `${Array.from({ length: 7 }, (_, j) => word(i * 7 + j + 1)).join(' ')}.`;
  let engine: Engine;
  let report: RepeatsReport;
  beforeAll(async () => {
    const rows: [string, number, string][] = [];
    for (let i = 0; i < 60; i++) rows.push([`s${i % 20}`, i % 15, filler(i)]);
    // A habit: the same instruction in five sessions over several days, phrased two ways.
    for (let s = 0; s < 5; s++) rows.push([`s${s}`, s * 2, `${filler(100 + s)} ${s % 2 ? 'Never use tabs in the final output.' : 'Never use tabs in the final output please.'}`]);
    // Emphasis: three times, but all in one session.
    for (let k = 0; k < 3; k++) rows.push(['s7', 3, `${filler(200 + k)} Remember the invoice totals must match exactly.`]);
    // A burst: four sessions, all on one day.
    for (let s = 10; s < 14; s++) rows.push([`s${s}`, 9, `${filler(300 + s)} Respond with compact output and nothing else.`]);
    // A program's template: the same long preamble opening four sessions on different days.
    const preamble = `You are the scheduling assistant for the autumn term. Review the list below and write one line about what matters most. ${'Course detail line that pads this preamble out. '.repeat(6)}`;
    for (let s = 14; s < 18; s++) rows.push([`s${s}`, s, `${preamble} Reference number ${s}.`]);
    engine = await storeOf('habits', rows);
    report = await runRepeats(engine, { embedder: hashEmbedder() });
  });
  afterAll(() => engine.close());

  it('finds the habit and only the habit', () => {
    expect(report.groups).toBe(1);
    const [g] = report.top;
    expect(g!.text).toMatch(/^Never use tabs in the final output/);
    expect(g!.kind).toBe('instruction');
    expect(g!.occurrences).toBe(5);
    expect(g!.sessions).toBe(5);
    expect(g!.projects).toEqual(['project-one']);
    expect(g!.nodes).toHaveLength(5);
    expect(g!.variants).toHaveLength(1); // the other phrasing
    expect(new Date(g!.first).toISOString().slice(0, 10)).toBe('2026-01-01');
    expect(new Date(g!.last).toISOString().slice(0, 10)).toBe('2026-01-09');
    expect(report.byKind).toEqual({ instruction: 1, task: 0, question: 0, statement: 0 });
    expect(report.share).toBeCloseTo(5 / report.sentences, 10);
  });

  it('leaves out one-session emphasis, same-day bursts and program templates', () => {
    const all = JSON.stringify(report.top);
    expect(all).not.toContain('invoice totals'); // three times, one session
    expect(all).not.toContain('compact output'); // four sessions, one day
    expect(all).not.toContain('scheduling assistant'); // a template
    expect(report.sameDayBursts).toBe(1);
    expect(report.templatedPrompts).toBe(4);
    expect(templatedPrompts(engine).size).toBe(4);
    const { sentences } = candidateSentences(engine);
    expect(sentences.some((s) => s.text.includes('scheduling assistant'))).toBe(false);
  });

  it('reports far more matching pairs than chance would give', () => {
    expect(report.pairs.above).toBeGreaterThanOrEqual(10); // 5 choose 2, plus the burst's pairs
    expect(report.pairs.expectedByChance).toBeLessThan(1);
    expect(report.threshold.tau).toBeGreaterThan(report.threshold.median);
    expect(report.corpus).toBe(engine.n);
  });

  it('respects the minimum number of sessions', async () => {
    const strict = await runRepeats(engine, { embedder: hashEmbedder(), minSessions: 6 });
    expect(strict.groups).toBe(0);
    expect(strict.top).toEqual([]);
  });

  it('drafts rules from instructions only, with the evidence beside each', () => {
    const md = draftRules(report);
    expect(md).toContain('- Never use tabs in the final output');
    expect(md).toContain('said 5 times in 5 sessions across 1 project (project-one), 2026-01-01 to 2026-01-09');
    expect(md).toContain('  - also: Never use tabs in the final output');
    expect(md).toContain('Review and reword');
    expect(draftRules({ ...report, top: [] })).toContain('No recurring instructions found.');
  });

  it('writes a results section with numbers and none of the sentences', () => {
    const md = renderResults(null, null, '2026-01-01', null, report);
    expect(md).toContain('## 4. How much do you repeat yourself?');
    expect(md).toContain('**1 things were said in 3 or more sessions');
    expect(md).not.toContain('tabs in the final');
    expect(md).not.toContain('project-one');
    // Nothing to say when nothing was analysed.
    expect(renderResults(null, null, 'x', null, { ...report, sentences: 0 })).not.toContain('## 4.');
  });

  it('hands the viewer particle indices, and refuses a report from a different store', () => {
    const data = buildViewerData(engine, { demo: true, model: DEMO_MODEL });
    const view = repeatsView(report, engine, data);
    expect(view.available).toBe(true);
    expect(view.items![0]!.particles).toHaveLength(5);
    expect(view.items![0]!.particles.every((p) => p >= 0 && p < engine.n)).toBe(true);
    expect(view.items![0]).not.toHaveProperty('nodes');
    expect(repeatsView({ ...report, corpus: engine.n + 1 }, engine, data)).toMatchObject({ available: false });
    expect(repeatsView(null, engine, data).reason).toContain('loci repeats');
  });
});

describe('runRepeats on the demo store', () => {
  it('recovers every planted instruction across many sessions', async () => {
    const cfg = await buildDemoStore(path.join(tmp, 'demo'), 3000, 4);
    const engine = Engine.open(cfg, { embedder: hashEmbedder(), save: false });
    const report = await runRepeats(engine, { embedder: hashEmbedder() });
    const found = report.top.map((g) => g.text);
    for (const family of ['run the unit tests before you commit', 'Keep the answer short', 'Never push to the remote']) {
      const group = report.top.find((g) => g.text.includes(family));
      expect(group, `missing "${family}"`).toBeDefined();
      expect(group!.sessions).toBeGreaterThanOrEqual(3);
      expect(group!.kind).toBe('instruction');
    }
    expect(new Set(PLANTED).size).toBe(6);
    // The planted instructions rank above incidental matches between invented sentences.
    expect(found.slice(0, 3).every((t) => PLANTED.some((p) => p === t))).toBe(true);
    engine.close();
  });
});
