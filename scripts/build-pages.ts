import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Engine } from '../src/engine.ts';
import { runRepeats } from '../src/insight/repeats.ts';
import { buildDemoStore, hashEmbedder } from '../src/prove/demo.ts';
import { buildViewerData } from '../src/serve/export.ts';
import { repeatsView } from '../src/serve/http.ts';

/**
 * Builds the data for the public demo: a complete LOCI store of invented conversations, run
 * through the real ingest, index and export code, then written out as static files the viewer
 * can load without a server.
 *
 * Only invented text goes in. This script never reads the private store (~/.loci) or anyone's
 * history; it builds its store in a temporary directory and deletes it afterwards.
 *
 * Output: web/public-pages/demo/ (ignored by git; rebuilt on every deploy).
 */

const CHUNKS = 6000;
const SEED = 7;

const root = fileURLToPath(new URL('..', import.meta.url));
const out = path.join(root, 'web', 'public-pages', 'demo');

const bytes = (a: ArrayBufferView): Uint8Array => new Uint8Array(a.buffer, a.byteOffset, a.byteLength);

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'loci-pages-'));
try {
  const t0 = performance.now();
  const cfg = await buildDemoStore(tmp, CHUNKS, SEED, { prose: true });
  const engine = Engine.open(cfg, { embedder: hashEmbedder(), save: false });
  const data = buildViewerData(engine, { demo: true, model: cfg.embedModel });
  const repeats = repeatsView(await runRepeats(engine, { embedder: hashEmbedder() }), engine, data);

  const n = engine.n;
  const corpus = {
    texts: engine.texts,
    ts: Array.from(engine.ts),
    project: Array.from(engine.project),
    role: Array.from(engine.role),
    session: Array.from(engine.sessionPk),
    sessionTitles: Object.fromEntries([...engine.sessions].map(([pk, s]) => [String(pk), s.title ?? ''])),
    projectNames: Object.fromEntries(engine.projects.map((p) => [String(p.id), p.name])),
    projectIds: data.manifest.projects.map((p) => {
      const id = engine.projects.find((x) => x.name === p.name)?.id;
      if (id === undefined) throw new Error(`No project id for ${p.name}`);
      return id;
    }),
    rankOf: Array.from(data.rankOf),
    nodeAt: Array.from(data.nodeAt),
    repeats,
  };

  fs.rmSync(out, { recursive: true, force: true });
  fs.mkdirSync(out, { recursive: true });
  const files: [string, Uint8Array | string][] = [
    ['manifest.json', JSON.stringify(data.manifest)],
    ['corpus.json', JSON.stringify(corpus)],
    ['pos.f32', bytes(data.pos)],
    ['home.f32', bytes(data.home)],
    ['knn.u32', bytes(data.knn)],
    ['meta.u32', bytes(data.meta)],
    ['sentences.u8', bytes(data.sentences)],
    ['cluster.u8', bytes(data.cluster)],
    // Node order, the order the index refers to.
    ['vectors.f32', bytes(engine.vectors.subarray(0, n * engine.dim))],
    ['hnsw.bin', engine.hnsw.toBytes()],
  ];
  let total = 0;
  for (const [name, body] of files) {
    fs.writeFileSync(path.join(out, name), body);
    total += typeof body === 'string' ? Buffer.byteLength(body) : body.byteLength;
  }
  engine.close();
  console.log(
    `pages: ${n} invented chunks, ${data.manifest.projects.length} projects, ${data.manifest.sessions} sessions, ` +
      `${repeats.groups ?? 0} repeated habits; ${files.length} files, ${(total / 1e6).toFixed(1)} MB, ${Math.round(performance.now() - t0)} ms`,
  );
} finally {
  fs.rmSync(tmp, { recursive: true, force: true });
}
