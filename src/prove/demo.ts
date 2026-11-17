import fs from 'node:fs';
import path from 'node:path';
import type { Config } from '../config.ts';
import { modelTag } from '../config.ts';
import { sha256 } from '../ingest/run.ts';
import { Store } from '../ingest/store.ts';
import { tokenize } from '../index/tokenize.ts';
import { appendRows, prepareVecFile } from '../index/vecfile.ts';
import { DEMO_DIM, DEMO_MODEL, FILLER, hashEmbedder } from './hash-embed.ts';
import { proseDocs } from './prose.ts';
import { syntheticDocs } from './synth.ts';
import type { SynthDoc } from './synth.ts';

export { DEMO_MODEL, hashEmbedder };

/**
 * Demo mode: a complete LOCI store built from invented text, with a deterministic embedder that
 * needs no model. It runs the real ingest-to-search path, so it doubles as the fixture for
 * end-to-end tests and as something safe to screen-record: no private conversation is involved.
 */


/** Standing instructions planted in the demo: three families, each phrased a few ways. */
export const PLANTED = [
  'Always run the unit tests before you commit anything.',
  'Run the unit tests before you commit anything please.',
  'Always run the unit tests before you commit anything.',
  'Keep the answer short and skip the long summary.',
  'Keep the answer short and skip the long summary please.',
  'Keep the answer short and skip the long summary.',
  'Never push to the remote without asking me first.',
  'Never push to the remote without asking me first okay.',
  'Never push to the remote without asking me first.',
] as const;

/** Builds a throwaway store of invented documents under `dir` and returns a config pointing at it. */
export interface DemoOptions {
  /** Readable invented conversations (for the public demo) instead of the unit-test word salad. */
  prose?: boolean;
}

export async function buildDemoStore(dir: string, n = 6000, seed = 7, opts: DemoOptions = {}): Promise<Config> {
  fs.mkdirSync(dir, { recursive: true, mode: 0o700 });
  const cfg: Config = { home: dir, source: path.join(dir, 'none'), ollamaUrl: 'http://127.0.0.1:11434', embedModel: DEMO_MODEL, port: 4319 };
  const dbFile = path.join(dir, 'loci.db');
  const vecFile = path.join(dir, `vectors.${modelTag(DEMO_MODEL)}.f32`);
  for (const f of [dbFile, `${dbFile}-wal`, `${dbFile}-shm`, vecFile, path.join(dir, `hnsw.${modelTag(DEMO_MODEL)}.bin`)]) fs.rmSync(f, { force: true });

  const docs: (SynthDoc & { role?: 'human' | 'assistant' })[] = opts.prose ? proseDocs(n, seed, PLANTED) : syntheticDocs(n, seed);
  const store = new Store(dbFile);
  const embedder = hashEmbedder();
  const texts: string[] = [];
  store.transaction(() => {
    const sessionOf = new Map<string, { pk: number; left: number }>();
    const sessionCount = new Map<string, number>();
    docs.forEach((d, i) => {
      const projectId = store.upsertProject(`demo-${d.topic}`, d.topic);
      // Group each topic's documents into sessions of about 16 chunks.
      let s = sessionOf.get(d.topic);
      if (!s || s.left === 0) {
        const k = (sessionCount.get(d.topic) ?? 0) + 1;
        sessionCount.set(d.topic, k);
        // Titled from the session's own opening words, the way a real title summarises a real session.
        const opening = [...new Set(tokenize(d.text).filter((t) => !FILLER.has(t)))].slice(0, 4).join(' ');
        s = { pk: store.upsertSession(`demo-${d.topic}-${k}`, projectId, opening || `${d.topic} notes ${k}`, d.ts, d.ts), left: 16 };
        sessionOf.set(d.topic, s);
      }
      s.left--;
      const role = d.role ?? (i % 3 === 0 ? 'human' : 'assistant');
      // Every ninth prompt also carries a standing instruction, the kind of thing people keep
      // repeating, so the repeat-finder has something real-shaped to find in the demo. (Readable
      // demo text plants its own; see prose.ts.)
      const planted = !opts.prose && role === 'human' && (i / 3) % 9 === 4 ? ` ${PLANTED[Math.floor(i / 27) % PLANTED.length]!}` : '';
      const text = `${d.text}${planted}`;
      if (store.insertChunk({ uuid: `demo-${i}`, part: 0, sessionPk: s.pk, projectId, role, ts: d.ts, text, sha: sha256(`${role}\n${i}\n${text}`) })) {
        texts.push(text);
      }
    });
  });
  store.close();

  prepareVecFile(vecFile, DEMO_DIM, DEMO_MODEL);
  for (let i = 0; i < texts.length; i += 256) {
    const vecs = await embedder.embed(texts.slice(i, i + 256), 'document');
    const rows = new Float32Array(vecs.length * DEMO_DIM);
    vecs.forEach((v, j) => rows.set(v, j * DEMO_DIM));
    appendRows(vecFile, rows);
  }
  return cfg;
}
