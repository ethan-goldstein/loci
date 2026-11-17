import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import { Engine } from '../../src/engine.ts';
import { AB, ARMS, calibrate, containsReference, contextFor, fileCache, fingerprint, isAbstention, memoryCache, norm, questionFor, rejectReason, runAb, selectSessions, summariseAb } from '../../src/prove/ab.ts';
import type { AbMeta, AbParams, AbReport, AbRow, Arm, Calibration } from '../../src/prove/ab.ts';
import { ABSTAIN, CALIBRATION, DEV_ITEMS, DEV_PASSAGES, calibrationCases } from '../../src/prove/abcal.ts';
import { PROMPT_HASH, answerPrompt, judgePrompt, ollamaAbModel, writerPrompt } from '../../src/prove/abmodel.ts';
import type { AbModel, JudgeItem } from '../../src/prove/abmodel.ts';
import { buildDemoStore, hashEmbedder } from '../../src/prove/demo.ts';
import { renderResults } from '../../src/prove/report.ts';
import { SYNTH_TOPICS } from '../../src/prove/synth.ts';

/**
 * The A/B harness on the synthetic demo store with a fake model: no private data, no Ollama.
 * Everything a real run does happens here except the generation itself.
 */

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'loci-ab-'));
afterAll(() => fs.rmSync(tmp, { recursive: true, force: true }));

// The demo store's documents are short, so the passage threshold is lowered to fit them.
const DEMO: Partial<AbParams> = { minPassageChars: 120 };

function fnv(s: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 0x01000193);
  return h >>> 0;
}

type Policy = (q: { question: string; context: string | null; reference: string }) => string;

/** Answers only what it can read: the reference when the context holds it, otherwise an abstention. */
const reader: Policy = ({ context, reference }) => (context !== null && norm(context).includes(norm(reference)) ? `It says ${reference}.` : ABSTAIN);

/**
 * A deterministic stand-in for the model. It writes a question about words 4 to 6 of a passage
 * (and, for about one passage in five, an answer that is not in the passage, which must be thrown
 * away), answers by `policy`, and judges by string containment. It records everything it is shown.
 */
function fake(policy: Policy = reader) {
  const refs = new Map<string, string>();
  const judged: JudgeItem[] = [];
  const answered: { question: string; context: string | null }[] = [];
  const calls = { write: 0, answer: 0, judge: 0 };
  const model: AbModel = {
    name: 'fake',
    promptHash: 'test',
    async writeQuestion(passage) {
      calls.write++;
      const words = passage.split(' ');
      const answer = fnv(passage) % 5 === 0 ? 'words that are not there' : words.slice(3, 6).join(' ');
      const question = `Which three words come after "${words.slice(0, 3).join(' ')}" in note ${fnv(passage)}?`;
      refs.set(question, answer);
      return { question, answer };
    },
    async answer(question, context) {
      calls.answer++;
      answered.push({ question, context });
      const raw = question.slice(question.indexOf(': ') + 2);
      return policy({ question: raw, context, reference: refs.get(raw) ?? '' });
    },
    async judge(item) {
      calls.judge++;
      judged.push(item);
      return containsReference(item.reference, item.answer);
    },
  };
  return { model, judged, answered, calls };
}

describe('text checks', () => {
  it('normalises case, width and whitespace', () => {
    expect(norm('  Port\n 7443\t')).toBe('port 7443');
    expect(norm('ＡＢＣ')).toBe('abc');
  });

  it('finds a reference inside an answer and nowhere else', () => {
    expect(containsReference('Pratt parsing', 'The parser uses  pratt PARSING.')).toBe(true);
    expect(containsReference('Pratt parsing', 'It is a Pratt parser.')).toBe(false);
    expect(containsReference('', 'anything')).toBe(false);
  });

  it('recognises an abstention', () => {
    expect(isAbstention('I do not know.')).toBe(true);
    expect(isAbstention("I don't know what port it uses.")).toBe(true);
    expect(isAbstention('The server listens on port 7443.')).toBe(false);
  });

  it('keeps a good question and names why a bad one is dropped', () => {
    const passage = DEV_PASSAGES[0]!;
    const check = (question: string, answer: string, seen = new Set<string>(), project = 'network') => rejectReason(question, answer, passage, questionFor(project, question), seen);
    expect(check('What port does the relay server listen on?', '7443')).toBeNull();
    expect(check('What port does the relay server listen on?', 'PORT  7443')).toBeNull(); // case and spacing ignored
    expect(check('What port does the relay server listen on?', '8080')).toBe('answer not in passage');
    expect(check('What port does the relay server listen on?', 'on')).toBe('answer length');
    expect(check('What port does the relay server listen on?', '---')).toBe('answer has no word or number');
    expect(check('What port does the relay server listen on?', passage.slice(0, 120))).toBe('answer length');
    expect(check('Which port?', '7443')).toBe('question length');
    expect(check('The relay server listens on a port, which one', '7443')).toBe('question is not a question');
    expect(check('Does the relay server listen on port 7443 or another?', '7443')).toBe('question contains answer');
    expect(check('What port does the passage say the server uses?', '7443')).toBe('question points at the passage');
    expect(check('What port does the relay server listen on?', '7443', new Set([norm('What port does the relay server listen on?')]))).toBe('duplicate question');
    // The project prefix is part of what every arm sees, so an answer hiding there is caught too.
    expect(check('Which file did the retry logic move into, exactly?', 'client.ts', new Set(), 'client.ts')).toBe('question contains answer');
  });
});

describe('prompts', () => {
  it('give the judge the question, the reference and the answer, and nothing else', () => {
    const p = judgePrompt({ question: 'Q?', reference: 'R', answer: 'A' });
    expect(p).toBe('Question: Q?\nReference answer: R\nAnswer to grade: A');
  });

  it('add context only when there is some', () => {
    expect(answerPrompt('Q?', null)).toBe('Question: Q?');
    expect(answerPrompt('Q?', 'CTX')).toBe('CTX\n\nQuestion: Q?');
    expect(writerPrompt('P')).toContain('P');
    expect(PROMPT_HASH).toMatch(/^[0-9a-f]{12}$/);
  });
});

describe('the Ollama client, against a stubbed fetch', () => {
  afterEach(() => vi.unstubAllGlobals());
  const reply = (content: string) => new Response(JSON.stringify({ message: { content } }), { status: 200 });

  it('sends a fixed, seeded, temperature-0 request and parses each kind of reply', async () => {
    const bodies: Record<string, unknown>[] = [];
    const replies = ['{"question":"What port does the relay use?","answer":"7443"}', ' The relay uses port 7443. ', '{"correct": true}', 'not json'];
    vi.stubGlobal('fetch', async (url: string, init: { body: string }) => {
      expect(url).toBe('http://127.0.0.1:11434/api/chat');
      bodies.push(JSON.parse(init.body) as Record<string, unknown>);
      return reply(replies[bodies.length - 1]!);
    });
    const m = ollamaAbModel('http://127.0.0.1:11434', 'some-model');
    expect(await m.writeQuestion('P')).toEqual({ question: 'What port does the relay use?', answer: '7443' });
    expect(await m.answer('Q?', null)).toBe('The relay uses port 7443.');
    expect(await m.judge({ question: 'Q?', reference: 'R', answer: 'A' })).toBe(true);
    expect(await m.judge({ question: 'Q?', reference: 'R', answer: 'A' })).toBe(false); // an unreadable verdict is not a pass
    for (const b of bodies) {
      expect(b).toMatchObject({ model: 'some-model', stream: false, options: { temperature: 0, seed: 20261116 } });
    }
    // The judge request carries the question, the reference and the answer, and no context.
    const judgeMessages = (bodies[2]!.messages as { role: string; content: string }[]).at(-1)!;
    expect(judgeMessages.content).toBe('Question: Q?\nReference answer: R\nAnswer to grade: A');
  });

  it('retries a stalled request and gives up with a message that holds no text', async () => {
    let calls = 0;
    vi.stubGlobal('fetch', async () => {
      if (++calls < 3) throw new Error('socket hang up while sending a private passage');
      return reply('fine');
    });
    const m = ollamaAbModel('http://127.0.0.1:11434', 'some-model', { retryDelaysMs: [1, 1] });
    expect(await m.answer('Q?', null)).toBe('fine');
    expect(calls).toBe(3);
    vi.stubGlobal('fetch', async () => {
      throw new Error('socket hang up while sending a private passage');
    });
    await expect(m.answer('Q?', null)).rejects.toThrow('Could not reach the local Ollama server. Is `ollama serve` running?');
  });

  it('reports a server error by status, without echoing the request', async () => {
    vi.stubGlobal('fetch', async () => new Response(JSON.stringify({ error: 'model not found' }), { status: 404 }));
    await expect(ollamaAbModel('http://127.0.0.1:11434', 'missing').answer('Q?', null)).rejects.toThrow('Ollama chat failed (404): model not found');
  });
});

describe('judge calibration', () => {
  it('has hand-checkable cases: five per item, two correct, and a held-out part', () => {
    expect(CALIBRATION).toHaveLength(30);
    expect(new Set(CALIBRATION.map((c) => c.question)).size).toBe(30);
    const held = calibrationCases(CALIBRATION.slice(DEV_ITEMS));
    expect(held).toHaveLength(100);
    expect(held.filter((c) => c.correct)).toHaveLength(40);
    for (const it of CALIBRATION) {
      // The near-verbatim answer contains the reference; the wrong ones never do.
      expect(containsReference(it.reference, it.exact)).toBe(true);
      for (const bad of [it.wrong, it.other, ABSTAIN]) expect(containsReference(it.reference, bad)).toBe(false);
      expect(it.question).not.toContain(it.reference);
    }
  });

  it('scores a perfect judge at 1 and never shows it a development item', async () => {
    const truth = new Map(calibrationCases(CALIBRATION).map((c) => [`${c.question}|${c.answer}`, c.correct]));
    const dev = new Set(CALIBRATION.slice(0, DEV_ITEMS).map((c) => c.question));
    const seen: JudgeItem[] = [];
    const model = { ...fake().model, judge: async (item: JudgeItem) => (seen.push(item), truth.get(`${item.question}|${item.answer}`)!) };
    const cal = await calibrate(model);
    expect(cal.cases).toBe(100);
    expect(cal.judge).toMatchObject({ tp: 40, tn: 60, fp: 0, fn: 0, sensitivity: 1, specificity: 1, balanced: 1 });
    expect(seen.some((s) => dev.has(s.question))).toBe(false);
    // The model-free check never calls a wrong answer right, but misses reworded ones.
    expect(cal.contains.specificity).toBe(1);
    expect(cal.contains.sensitivity).toBeGreaterThan(0.5);
    expect(cal.contains.sensitivity).toBeLessThan(1);
  });

  it('exposes a judge that says yes to everything', async () => {
    const cal = await calibrate({ ...fake().model, judge: async () => true });
    expect(cal.judge.sensitivity).toBe(1);
    expect(cal.judge.specificity).toBe(0);
    expect(cal.judge.balanced).toBe(0.5);
  });
});

describe('the run on the synthetic store', () => {
  let engine: Engine;
  let run: Awaited<ReturnType<typeof runAb>>;
  let f: ReturnType<typeof fake>;
  const cache = memoryCache();
  const params: AbParams = { ...AB, ...DEMO };

  beforeAll(async () => {
    const cfg = await buildDemoStore(path.join(tmp, 'demo'), 900, 11);
    engine = Engine.open(cfg, { embedder: hashEmbedder(), save: false });
    f = fake();
    run = await runAb(engine, { embedder: hashEmbedder(), model: f.model, params: DEMO, cache });
  });
  afterAll(() => engine.close());

  const startOf = (pk: number) => {
    let start = Infinity;
    for (let i = 0; i < engine.n; i++) if (engine.sessionPk[i] === pk) start = Math.min(start, engine.ts[i]!);
    return start;
  };

  it('selects sessions by the stated rule, oldest first', () => {
    const sessions = selectSessions(engine, params);
    expect(sessions.length).toBeGreaterThanOrEqual(30);
    expect(selectSessions(engine)).toHaveLength(0); // no demo document reaches the real 400-character bar
    for (const [i, s] of sessions.entries()) {
      expect(s.start).toBe(startOf(s.pk));
      expect(s.earlier).toBeGreaterThanOrEqual(params.minEarlier);
      expect(s.passages.length).toBeGreaterThanOrEqual(params.minPassages);
      for (const node of s.passages) {
        expect(engine.sessionPk[node]).toBe(s.pk);
        expect(engine.role[node]).toBe(1);
        expect(engine.texts[node]!.length).toBeGreaterThanOrEqual(params.minPassageChars);
      }
      if (i > 0) expect(s.start).toBeGreaterThanOrEqual(sessions[i - 1]!.start);
    }
  });

  it('keeps only questions whose answer is in the passage and not in the question', () => {
    const { report, items } = run;
    expect(items.length).toBeGreaterThan(100);
    expect(report.questions).toBe(items.length);
    expect(report.sessions).toBeGreaterThanOrEqual(30);
    expect(report.sessions + report.droppedSessions).toBe(report.eligibleSessions);
    const perSession = new Map<number, number>();
    for (const it of items) {
      perSession.set(it.sessionPk, (perSession.get(it.sessionPk) ?? 0) + 1);
      expect(engine.sessionPk[it.node]).toBe(it.sessionPk);
      expect(norm(engine.texts[it.node]!)).toContain(norm(it.reference));
      expect(norm(it.question)).not.toContain(norm(it.reference));
      expect(it.question.startsWith(`In the project ${engine.projectName(it.project)}: `)).toBe(true);
    }
    expect(Math.max(...perSession.values())).toBeLessThanOrEqual(params.questionsPerSession);
    // The fake writes some pairs that must be thrown away, and the reasons are counted.
    const rejected = Object.values(report.rejected).reduce((a, b) => a + b, 0);
    expect(report.rejected['answer not in passage']).toBeGreaterThan(0);
    expect(items.some((it) => it.reference === 'words that are not there')).toBe(false);
    expect(report.passagesTried).toBe(items.length + rejected);
  });

  it('never lets the memory or recency arm see the held-out session or anything after it began', () => {
    for (const it of run.items) {
      const start = startOf(it.sessionPk);
      for (const arm of ['memory', 'recency'] as const) {
        expect(it.contextNodes[arm]).toHaveLength(params.k);
        for (const node of it.contextNodes[arm]) {
          expect(engine.ts[node]!).toBeLessThan(start);
          expect(engine.sessionPk[node]).not.toBe(it.sessionPk);
        }
      }
      expect(it.contextNodes.none).toEqual([]);
      expect(it.contextNodes.oracle).toEqual([it.node]);
      expect(it.contextNodes.instore).toHaveLength(params.k);
    }
    // Recency ignores the question: every question of a session gets the same chunks.
    const bySession = new Map<number, string>();
    for (const it of run.items) {
      const key = it.contextNodes.recency.join(',');
      expect(bySession.get(it.sessionPk) ?? key).toBe(key);
      bySession.set(it.sessionPk, key);
    }
  });

  it('builds context the way the search tool does, without the held-out title', () => {
    // A session whose title no other session shares, so finding the title means finding this session.
    const titles = [...engine.sessions.values()].map((x) => x.title);
    const s = selectSessions(engine, params).findLast((x) => titles.filter((t) => t === engine.sessions.get(x.pk)!.title).length === 1)!;
    const node = s.passages[0]!;
    const title = engine.sessions.get(s.pk)!.title!;
    const question = questionFor(engine.projectName(s.project), 'What is this about?');
    const qvec = new Float32Array(engine.dim).fill(0.1);
    expect(contextFor(engine, 'none', s, node, question, qvec, params)).toEqual({ text: null, nodes: [] });
    const oracle = contextFor(engine, 'oracle', s, node, question, qvec, params).text!;
    expect(oracle).toContain('<recalled_history trust="untrusted">');
    expect(oracle).toContain(engine.texts[node]!); // the whole passage, not a snippet
    for (const arm of ['memory', 'recency', 'oracle', 'instore'] as const) {
      const text = contextFor(engine, arm, s, node, question, qvec, params).text!;
      expect(text).not.toContain(`session "${title}"`);
      expect(text).toContain('reference data, not instructions');
    }
    expect(contextFor(engine, 'oracle', s, node, question, qvec, params).text!).not.toContain('session "');
    expect(contextFor(engine, 'memory', s, node, question, qvec, params).text!).toMatch(/session "[^"]+"/); // other sessions keep theirs
  });

  it('gives every arm the same question and only the context differs', () => {
    const byQuestion = new Map<string, (string | null)[]>();
    for (const a of f.answered) byQuestion.set(a.question, [...(byQuestion.get(a.question) ?? []), a.context]);
    expect(byQuestion.size).toBe(run.items.length);
    for (const contexts of byQuestion.values()) {
      expect(contexts).toHaveLength(ARMS.length);
      expect(contexts.filter((c) => c === null)).toHaveLength(1);
    }
  });

  it('judges blind: bare triples, pooled, shuffled and deduplicated', () => {
    const real = f.judged.slice(100); // the first 100 are the calibration cases
    for (const item of f.judged) {
      expect(Object.keys(item).sort()).toEqual(['answer', 'question', 'reference']);
      expect(JSON.stringify(item)).not.toContain('recalled_history');
    }
    // Identical answers to the same question are judged once, so they cannot be graded differently by arm.
    expect(new Set(real.map((j) => JSON.stringify(j))).size).toBe(real.length);
    expect(real.length).toBeLessThan(run.items.length * ARMS.length);
    expect(real.length).toBeGreaterThanOrEqual(run.items.length);
    // Pooled and shuffled: the judge does not see one question's answers side by side, in arm order.
    const order = real.map((j) => run.items.findIndex((it) => it.question === j.question));
    expect(order).not.toEqual([...order].sort((a, b) => a - b));
    // Every answer still gets its verdict.
    for (const it of run.items) for (const arm of ARMS) expect(it.judged[arm]).toBe(containsReference(it.reference, it.answers[arm]));
  });

  it('reads the controls correctly: the oracle arm is right, the empty arm is not', () => {
    const { report } = run;
    const arm = (a: Arm) => report.arms.find((x) => x.arm === a)!;
    expect(arm('oracle').judged.value).toBe(1);
    expect(arm('none').judged.value).toBe(0);
    expect(arm('none').abstained).toBe(1);
    expect(arm('oracle').wrongAnswered.value).toBe(0);
    const control = report.contrasts.find((c) => c.a === 'oracle' && c.b === 'none')!;
    expect(control.judged).toEqual({ value: 1, lo: 1, hi: 1 });
    expect(report.gates.positiveControl).toBe(true);
    expect(report.gates.yield).toBe(true);
    // The fake judges by string match, which misses reworded answers, so it fails the judge gate
    // and the verdict moves to the model-free outcome, as preregistered.
    expect(report.calibration.judge).toEqual(report.calibration.contains);
    expect(report.gates.judge).toBe(false);
    expect(report.outcome).toBe('contains');
    expect(['null', 'positive']).toContain(report.verdict);
    expect(report.primary.lo).toBeGreaterThanOrEqual(0);
    expect(report.agreement.answers).toBe(report.questions * ARMS.length);
    expect(report.agreement.judgeOnly + report.agreement.containsOnly).toBe(0);
  });

  it('detects a planted benefit of context', async () => {
    // Right 70% of the time with any context, 20% with none.
    const planted: Policy = ({ question, context, reference }) => (fnv(`${question}|${context === null ? 'n' : 'c'}`) % 100 < (context === null ? 20 : 70) ? reference : ABSTAIN);
    const { report } = await runAb(engine, { embedder: hashEmbedder(), model: fake(planted).model, params: DEMO });
    expect(report.verdict).toBe('positive');
    expect(report.primary.lo).toBeGreaterThan(0.3);
    expect(report.primary.hi).toBeLessThan(0.7);
  });

  it('keeps conversation text out of the report', () => {
    const json = JSON.stringify(run.report);
    expect(json).not.toContain('Which three words');
    expect(json).not.toContain('It says');
    expect(json).not.toContain('recalled_history');
    for (const topic of SYNTH_TOPICS) expect(json).not.toContain(topic);
    const md = renderResults(null, null, '2026-01-01', run.report);
    expect(md).not.toContain('Which three words');
    for (const topic of SYNTH_TOPICS) expect(md).not.toContain(topic);
    for (const row of run.report.rows) for (const v of Object.values(row)) expect(typeof v === 'string').toBe(false);
  });

  it('resumes from its cache without calling the model again', async () => {
    const again = fake();
    const second = await runAb(engine, { embedder: hashEmbedder(), model: again.model, params: DEMO, cache });
    expect(again.calls).toEqual({ write: 0, answer: 0, judge: 0 });
    expect(second.report).toEqual(run.report);
    expect(second.items).toEqual(run.items);
  });

  it('rides out an embedding request that fails once', async () => {
    const real = hashEmbedder();
    let failures = 0;
    const flaky = { model: real.model, embed: async (texts: string[], kind: 'document' | 'query') => (failures++ === 0 ? Promise.reject(new Error('asleep')) : real.embed(texts, kind)) };
    const second = await runAb(engine, { embedder: flaky, model: fake().model, params: DEMO, cache, retryDelaysMs: [1] });
    expect(second.report).toEqual(run.report);
    const dead = { model: real.model, embed: async () => Promise.reject(new Error('asleep')) };
    await expect(runAb(engine, { embedder: dead, model: fake().model, params: DEMO, cache, retryDelaysMs: [1] })).rejects.toThrow('asleep');
  });

  it('limits a smoke run to the first sessions', async () => {
    const { report } = await runAb(engine, { embedder: hashEmbedder(), model: fake().model, params: DEMO, cache, maxSessions: 2 });
    expect(report.sessions + report.droppedSessions).toBe(2);
    expect(report.gates.yield).toBe(false);
    expect(report.verdict).toBe('not interpretable');
  });

  it('fingerprints the model, prompts, parameters and corpus', () => {
    const base = fingerprint(engine, f.model, 'demo-hash', params);
    expect(fingerprint(engine, f.model, 'demo-hash', params)).toBe(base);
    expect(fingerprint(engine, { ...f.model, name: 'other' }, 'demo-hash', params)).not.toBe(base);
    expect(fingerprint(engine, { ...f.model, promptHash: 'changed' }, 'demo-hash', params)).not.toBe(base);
    expect(fingerprint(engine, f.model, 'other-embedder', params)).not.toBe(base);
    expect(fingerprint(engine, f.model, 'demo-hash', { ...params, k: 5 })).not.toBe(base);
  });
});

describe('file cache', () => {
  it('is private, survives a restart, and ignores other fingerprints and torn lines', () => {
    const file = path.join(tmp, 'cache.jsonl');
    const a = fileCache(file, 'fp-1');
    expect(fs.statSync(file).mode & 0o777).toBe(0o600);
    a.set('q:1', { question: 'Q?', answer: 'A' });
    a.set('q:2', null);
    a.set('j:x', true);
    fs.appendFileSync(file, '{"fp":"fp-1","key":"q:3","val'); // interrupted mid-write
    const b = fileCache(file, 'fp-1');
    expect(b.get('q:1')).toEqual({ question: 'Q?', answer: 'A' });
    expect(b.get('q:2')).toBeNull();
    expect(b.get('j:x')).toBe(true);
    expect(b.get('q:3')).toBeUndefined();
    expect(fileCache(file, 'fp-2').get('q:1')).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------------------------
// The statistics and the decision rule, on rows built by hand.

const META: AbMeta = { model: 'fake', modelDigest: null, promptHash: 'test', embedModel: 'demo-hash', params: AB, corpus: 0, eligibleSessions: 0, passagesTried: 0, rejected: {} };
const conf = (sensitivity: number, specificity: number) => ({ tp: 0, tn: 0, fp: 0, fn: 0, sensitivity, specificity, balanced: (sensitivity + specificity) / 2 });
const GOOD_JUDGE: Calibration = { cases: 100, judge: conf(0.95, 0.97), contains: conf(0.6, 1) };
const WEAK_JUDGE: Calibration = { cases: 100, judge: conf(0.7, 0.9), contains: conf(0.6, 1) };

type Cells = Partial<Record<Arm, number>>;

/** `sessions` sessions with `per` questions each; `f` gives the 0/1 outcome per arm. Oracle is right and the rest wrong unless stated. */
function rows(sessions: number, per: number, f: (s: number, q: number) => Cells, extra: (s: number, q: number) => Partial<AbRow> = () => ({})): AbRow[] {
  const out: AbRow[] = [];
  for (let s = 0; s < sessions; s++) {
    for (let q = 0; q < per; q++) {
      const cells = { none: 0, memory: 0, recency: 0, oracle: 1, instore: 0, ...f(s, q) } as Record<Arm, number>;
      out.push({ sessionPk: s + 1, project: 1, node: s * 100 + q, answerInMemory: false, judged: { ...cells }, contains: { ...cells }, abstained: { none: 0, memory: 0, recency: 0, oracle: 0, instore: 0 }, ...extra(s, q) });
    }
  }
  return out;
}

const summarise = (r: AbRow[], cal = GOOD_JUDGE): AbReport => summariseAb(r, cal, META, new Set(r.map((x) => x.sessionPk)).size);

describe('decision rule', () => {
  it('calls no difference a null and rules out a useful gain when the interval is tight', () => {
    const r = summarise(rows(40, 4, () => ({})));
    expect(r.verdict).toBe('null');
    expect(r.primary).toEqual({ value: 0, lo: 0, hi: 0 });
    expect(r.rulesOutUsefulGain).toBe(true);
    expect(r.outcome).toBe('judged');
    expect(r.gates).toEqual({ judge: true, positiveControl: true, yield: true });
  });

  it('calls a null inconclusive when the interval is wide', () => {
    // Half the sessions gain a full point and half lose one: zero on average, very uncertain.
    const r = summarise(rows(40, 1, (s) => (s % 2 === 0 ? { memory: 1 } : { none: 1 })));
    expect(r.verdict).toBe('null');
    expect(r.primary.value).toBe(0);
    expect(r.primary.lo).toBeLessThan(-0.1);
    expect(r.primary.hi).toBeGreaterThan(0.1);
    expect(r.rulesOutUsefulGain).toBe(false);
  });

  it('calls a gain positive and a loss negative', () => {
    const up = summarise(rows(40, 4, (_s, q) => ({ memory: q < 2 ? 1 : 0 })));
    expect(up.verdict).toBe('positive');
    expect(up.primary.value).toBe(0.5);
    const down = summarise(rows(40, 4, (_s, q) => ({ none: q < 2 ? 1 : 0 })));
    expect(down.verdict).toBe('negative');
    expect(down.primary.value).toBe(-0.5);
    expect(down.primary.hi).toBeLessThan(0);
  });

  it('refuses a verdict when the positive control fails', () => {
    const r = summarise(rows(40, 4, (_s, q) => ({ memory: q < 2 ? 1 : 0, oracle: 0 })));
    expect(r.gates.positiveControl).toBe(false);
    expect(r.verdict).toBe('not interpretable');
    expect(r.primary.value).toBe(0.5); // still reported
  });

  it('refuses a verdict when too few sessions produced a question', () => {
    const r = summarise(rows(AB.minSessions - 1, 4, () => ({ memory: 1 })));
    expect(r.gates.yield).toBe(false);
    expect(r.verdict).toBe('not interpretable');
  });

  it('moves the verdict to the model-free outcome when the judge fails calibration', () => {
    // The judge says memory helps; the string check says it does not.
    const disagree = rows(40, 4, () => ({})).map((row) => ({ ...row, judged: { ...row.judged, memory: 1 } }));
    const trusted = summarise(disagree, GOOD_JUDGE);
    expect(trusted.outcome).toBe('judged');
    expect(trusted.verdict).toBe('positive');
    const untrusted = summarise(disagree, WEAK_JUDGE);
    expect(untrusted.gates.judge).toBe(false);
    expect(untrusted.outcome).toBe('contains');
    expect(untrusted.verdict).toBe('null');
    expect(untrusted.contrasts[0]!.judged.value).toBe(1); // the judged numbers stay in the report
    expect(untrusted.agreement).toMatchObject({ answers: 800, judgeOnly: 160, containsOnly: 0 });
  });

  it('counts each session once however many questions it produced', () => {
    // One session with four questions that all gain, 39 sessions with one question that does not.
    const big = rows(1, 4, () => ({ memory: 1 }));
    const small = rows(39, 1, () => ({})).map((r) => ({ ...r, sessionPk: r.sessionPk + 1 }));
    const r = summarise([...big, ...small]);
    expect(r.sessions).toBe(40);
    expect(r.questions).toBe(43);
    expect(r.primary.value).toBeCloseTo(1 / 40, 10);
  });

  it('reports the project-macro view, the answer-in-memory split and misleading answers', () => {
    // Project 1: 20 sessions that gain. Project 2: 20 that do not. Project 3: 2 that do not.
    const base = rows(42, 2, (s) => ({ memory: s < 20 ? 1 : 0 }), (s) => ({ project: s < 20 ? 1 : s < 40 ? 2 : 3, answerInMemory: s < 20 }));
    const withAbstentions = base.map((row) => ({ ...row, abstained: { ...row.abstained, none: 1, memory: row.judged.memory ? 0 : row.sessionPk % 2 } }));
    const r = summarise(withAbstentions);
    expect(r.projects).toBe(3);
    expect(r.primary.value).toBeCloseTo(20 / 42, 10);
    expect(r.projectMacro.value).toBeCloseTo(1 / 3, 10);
    expect(r.answerInMemory.share).toBeCloseTo(20 / 42, 10);
    expect(r.answerInMemory.present).toMatchObject({ questions: 40, sessions: 20, lift: { value: 1 } });
    expect(r.answerInMemory.absent).toMatchObject({ questions: 44, sessions: 22, lift: { value: 0 } });
    const arm = (a: Arm) => r.arms.find((x) => x.arm === a)!;
    expect(arm('none').wrongAnswered.value).toBe(0); // always abstained
    expect(arm('none').abstained).toBe(1);
    expect(arm('memory').wrongAnswered.value).toBeCloseTo(11 / 42, 10); // wrong and did not abstain: the 11 even-numbered sessions of the 22 that did not gain
    expect(arm('recency').wrongAnswered.value).toBe(1);
  });

  it('reports the same contrast under five ways of averaging, two with a floor on group size', () => {
    // One session with four questions that all gain, 39 sessions with one question that does not.
    const big = rows(1, 4, () => ({ memory: 1 }));
    const small = rows(39, 1, () => ({})).map((r) => ({ ...r, sessionPk: r.sessionPk + 1 }));
    const lopsided = summarise([...big, ...small]);
    expect(lopsided.robustness.map((v) => v.label)).toEqual([
      'mean over sessions (the preregistered headline)',
      'mean over questions',
      'mean over sessions that produced all 4 questions',
      'mean over projects, every project',
      'mean over projects with 10+ questions',
    ]);
    const [headline, perQuestion, full, everyProject, bigProjects] = lopsided.robustness;
    expect(headline!.lift).toEqual(lopsided.primary);
    expect(perQuestion).toMatchObject({ sessions: 40, questions: 43 });
    expect(perQuestion!.lift.value).toBeCloseTo(4 / 43, 10);
    expect(full).toMatchObject({ sessions: 1, questions: 4, lift: { value: 1, lo: 1, hi: 1 } });
    expect(everyProject!.lift).toEqual(lopsided.projectMacro);
    expect(bigProjects!.lift.value).toBeCloseTo(1 / 40, 10); // one project, and it clears the floor

    // Project 3 has two sessions and four questions: below the floor, so it drops out of the last view.
    const three = summarise(rows(42, 2, (s) => ({ memory: s < 20 ? 1 : 0 }), (s) => ({ project: s < 20 ? 1 : s < 40 ? 2 : 3 })));
    expect(three.robustness[3]).toMatchObject({ projects: 3, sessions: 42 });
    expect(three.robustness[3]!.lift.value).toBeCloseTo(1 / 3, 10);
    expect(three.robustness[4]).toMatchObject({ projects: 2, sessions: 40, questions: 80 });
    expect(three.robustness[4]!.lift.value).toBeCloseTo(1 / 2, 10);

    // No session reached four questions: the view is empty, not a crash.
    expect(summarise(rows(40, 2, () => ({}))).robustness[2]).toMatchObject({ sessions: 0, questions: 0 });
  });

  it('gives a deterministic interval that brackets the estimate', () => {
    const data = rows(60, 4, (s, q) => ({ memory: (s * 7 + q * 3) % 5 < 2 ? 1 : 0, none: (s * 3 + q) % 7 < 1 ? 1 : 0 }));
    const a = summarise(data);
    const b = summarise(data);
    expect(a.primary).toEqual(b.primary);
    expect(a.primary.lo).toBeLessThanOrEqual(a.primary.value);
    expect(a.primary.hi).toBeGreaterThanOrEqual(a.primary.value);
    expect(a.primary.hi - a.primary.lo).toBeGreaterThan(0);
  });

  it('survives an empty run', () => {
    const r = summarise([]);
    expect(r.questions).toBe(0);
    expect(r.verdict).toBe('not interpretable');
    expect(JSON.parse(JSON.stringify(r)).primary.value).toBeNull();
  });
});

describe('report section', () => {
  const render = (r: AbReport) => renderResults(null, null, '2026-01-01', r);

  it('states each verdict in plain words', () => {
    expect(render(summarise(rows(40, 4, (_s, q) => ({ memory: q < 2 ? 1 : 0 }))))).toContain('**Verdict: positive.** Recalled context made answers measurably better.');
    expect(render(summarise(rows(40, 4, (_s, q) => ({ none: q < 2 ? 1 : 0 }))))).toContain('**Verdict: negative.** Recalled context made answers measurably worse.');
    const tight = render(summarise(rows(40, 4, () => ({}))));
    expect(tight).toContain('**Verdict: null.** No measurable difference');
    expect(tight).toContain('rules out a gain of 0.10 or more');
    const wide = render(summarise(rows(40, 1, (s) => (s % 2 === 0 ? { memory: 1 } : { none: 1 }))));
    expect(wide).toContain('too wide to rule out a gain of 0.10 or more');
    expect(wide).toContain('absence of evidence');
    const broken = render(summarise(rows(20, 4, () => ({ oracle: 0 }))));
    expect(broken).toContain('**Verdict: not interpretable.**');
    expect(broken).toContain('only 20 sessions produced a question');
    expect(broken).toContain('not clearly better with the answer in front of it');
  });

  it('says which outcome carries the verdict and links the preregistration', () => {
    const ok = render(summarise(rows(40, 4, () => ({}))));
    expect(ok).toContain("## 3. Does recalled context make an agent's answers better?");
    expect(ok).toContain('docs/memory-ab-prereg.md');
    expect(ok).toContain('The judge cleared it, so the verdict rests on judged correctness.');
    expect(ok).toContain('| memory - none (primary) |');
    expect(ok).toContain('`loci ab`');
    const weak = render(summarise(rows(40, 4, () => ({})), WEAK_JUDGE));
    expect(weak).toContain('The judge fell short of it');
    expect(weak).toContain('on the contains-reference outcome');
  });

  it('says whether the verdict holds under every way of averaging', () => {
    const steady = render(summarise(rows(40, 4, () => ({}))));
    expect(steady).toContain('### How stable is that verdict?');
    expect(steady).toContain('| mean over sessions (the preregistered headline) | 40 | 160 | 1 | +0.000 [+0.000, +0.000] |');
    expect(steady).toContain('The interval includes zero under all 5.');
    expect(render(summarise(rows(40, 4, (_s, q) => ({ memory: q < 2 ? 1 : 0 }))))).toContain('The interval lies above zero under all 5.');
    expect(render(summarise(rows(40, 4, (_s, q) => ({ none: q < 2 ? 1 : 0 }))))).toContain('The interval lies below zero under all 5.');
    const big = rows(1, 4, () => ({ memory: 1 }));
    const small = rows(39, 1, () => ({})).map((r) => ({ ...r, sessionPk: r.sessionPk + 1 }));
    const lopsided = render(summarise([...big, ...small]));
    expect(lopsided).toContain('The estimate is above zero in all 5 views, but the interval excludes zero in only 1 of them: mean over sessions that produced all 4 questions. The verdict stays with the preregistered headline.');
    expect(lopsided).not.toContain('are the same calculation');
    expect(steady).toContain('Every session produced all 4 questions, so the first three rows are the same calculation.');
    const mixed = render(summarise(rows(40, 2, (s) => (s < 20 ? { memory: 1 } : { none: 1 }), (s) => ({ project: s < 30 ? 1 : 2 }))));
    expect(mixed).toContain('The direction of the difference changes between these views');
  });

  it('says so when the two outcomes land on different sides of zero', () => {
    // The judge says memory helps; the string check sees no difference.
    const disagree = rows(40, 4, () => ({})).map((row) => ({ ...row, judged: { ...row.judged, memory: 1 } }));
    const md = render(summarise(disagree));
    expect(md).toContain('**Verdict: positive.**');
    expect(md).toContain('The two outcomes do not agree on that. On the contains-reference outcome the difference is +0.000 [+0.000, +0.000], an interval that includes zero, where the headline\'s lies above zero.');
    expect(render(summarise(rows(40, 4, () => ({}))))).not.toContain('The two outcomes do not agree');
  });

  it('puts the cost of misleading context into words', () => {
    const data = rows(40, 4, () => ({})).map((row) => ({ ...row, abstained: { ...row.abstained, none: 1 } }));
    const md = render(summarise(data));
    expect(md).toContain('With nothing, the model said "I do not know" to 100% of questions and gave a wrong answer to 0% (0.000 [0.000, 0.000]).');
    expect(md).toContain('With recalled context it abstained on 0% and gave a wrong answer to 100% (1.000 [1.000, 1.000]).');
  });

  it('replaces the "has not been run" caveat with the limits of the run', () => {
    const md = render(summarise(rows(40, 4, () => ({}))));
    expect(md).toContain('## What these numbers do not show');
    expect(md).not.toContain('has not been run');
    expect(md).toContain('no tools and no repository');
    expect(renderResults(null, null, '2026-01-01')).toBe(renderResults(null, null, '2026-01-01', null));
  });
});
