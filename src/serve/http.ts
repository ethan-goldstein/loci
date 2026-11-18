import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import type { Config } from '../config.ts';
import { ollamaEmbedder } from '../embed/ollama.ts';
import { Engine } from '../engine.ts';
import type { Mode } from '../engine.ts';
import { DIVERSIFY_DEFAULT } from '../search/core.ts';
import { runProfile } from '../insight/profile.ts';
import type { ProfileReport } from '../insight/profile.ts';
import { runRepeats, templatedPrompts } from '../insight/repeats.ts';
import type { RepeatGroup, RepeatsReport } from '../insight/repeats.ts';
import { buildDemoStore, hashEmbedder } from '../prove/demo.ts';
import { buildViewerData } from './export.ts';
import type { ViewerData } from './export.ts';
import { shapeChunk, shapeSearch } from './shape.ts';
import type { Describe } from './shape.ts';

/**
 * The local viewer's server. It exists to show one person their own data on their own machine,
 * and is built so it cannot do anything else (playbook section 3):
 *
 *   - binds 127.0.0.1 only, never a routable interface
 *   - checks Host and Origin, so a web page elsewhere cannot reach it through the browser
 *     (DNS rebinding and cross-site requests are both refused)
 *   - GET and HEAD only; there is nothing to write
 *   - every parameter is validated; SQL is never built from input
 *   - static files are confined to the build directory
 *   - errors are short messages, never stack traces or paths
 */

const DIST = fileURLToPath(new URL('../../dist/', import.meta.url));

const TYPES: Record<string, string> = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.ico': 'image/x-icon',
  '.wasm': 'application/wasm',
  '.task': 'application/octet-stream',
  '.binarypb': 'application/octet-stream',
  '.woff2': 'font/woff2',
};

const CSP = [
  "default-src 'self'",
  "script-src 'self' 'wasm-unsafe-eval'",
  "style-src 'self' 'unsafe-inline'",
  "img-src 'self' data: blob:",
  "connect-src 'self'",
  "worker-src 'self' blob:",
  "media-src 'self' blob:",
  "object-src 'none'",
  "base-uri 'none'",
  "form-action 'none'",
  "frame-ancestors 'none'",
].join('; ');

class HttpError extends Error {
  readonly status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

export interface ServerOptions {
  demo?: boolean;
  /** Extra origins allowed to call the API (the Vite dev server). */
  devOrigins?: string[];
  quiet?: boolean;
}

export interface RunningServer {
  port: number;
  url: string;
  close: () => Promise<void>;
}

function intParam(q: URLSearchParams, name: string, fallback: number, min: number, max: number): number {
  const raw = q.get(name);
  if (raw === null || raw === '') return fallback;
  if (!/^\d{1,9}$/.test(raw)) throw new HttpError(400, `${name} must be a whole number`);
  const n = Number(raw);
  if (n < min || n > max) throw new HttpError(400, `${name} must be between ${min} and ${max}`);
  return n;
}

/** What the viewer needs of a repeats report: groups as particle indices, or why there are none. */
export interface RepeatsView {
  available: boolean;
  reason?: string;
  minSessions?: number;
  groups?: number;
  share?: number;
  items?: (Omit<RepeatGroup, 'nodes'> & { particles: number[] })[];
}

export function repeatsView(report: RepeatsReport | null, engine: Engine, data: ViewerData): RepeatsView {
  if (!report) return { available: false, reason: 'Run `loci repeats` to compute this.' };
  // Chunk numbers in an old report may point at different text after a rebuild.
  if (report.corpus !== engine.n) return { available: false, reason: 'The store has changed since this was computed. Run `loci repeats` again.' };
  return {
    available: true,
    minSessions: report.minSessions,
    groups: report.groups,
    share: report.share,
    items: report.top.map(({ nodes, ...g }) => ({ ...g, particles: nodes.filter((nd) => nd >= 0 && nd < engine.n).map((nd) => data.rankOf[nd]!) })),
  };
}

/** What the viewer needs of a profile: the report, or why there is none. */
export interface ProfileView {
  available: boolean;
  reason?: string;
  profile?: ProfileReport;
}

export function profileView(report: ProfileReport | null, engine: Engine): ProfileView {
  if (!report) return { available: false, reason: 'Run `loci profile` to compute this.' };
  if (report.corpus !== engine.n) return { available: false, reason: 'The store has changed since this was computed. Run `loci profile` again.' };
  return { available: true, profile: report };
}

export function createHandler(engine: Engine, data: ViewerData, port: number, opts: ServerOptions & { repeats?: RepeatsView; profile?: ProfileView } = {}) {
  const hosts = new Set([`127.0.0.1:${port}`, `localhost:${port}`]);
  const origins = new Set([`http://127.0.0.1:${port}`, `http://localhost:${port}`, ...(opts.devOrigins ?? [])]);
  for (const o of opts.devOrigins ?? []) hosts.add(new URL(o).host);

  const buffers: Record<string, Buffer> = {
    pos: Buffer.from(data.pos.buffer, data.pos.byteOffset, data.pos.byteLength),
    home: Buffer.from(data.home.buffer, data.home.byteOffset, data.home.byteLength),
    knn: Buffer.from(data.knn.buffer, data.knn.byteOffset, data.knn.byteLength),
    meta: Buffer.from(data.meta.buffer, data.meta.byteOffset, data.meta.byteLength),
    sentences: Buffer.from(data.sentences.buffer, data.sentences.byteOffset, data.sentences.byteLength),
    cluster: Buffer.from(data.cluster.buffer, data.cluster.byteOffset, data.cluster.byteLength),
    level: Buffer.from(data.level.buffer, data.level.byteOffset, data.level.byteLength),
  };
  const projectByIndex = data.manifest.projects;

  const describe: Describe = (node, chars) => {
    const d = engine.describe({ node, chunkId: node + 1, score: 0, sim: null, denseRank: 0, bm25Rank: 0 }, chars);
    return { particle: data.rankOf[node]!, chunkId: d.chunkId, project: d.project, role: d.role, date: d.date, session: d.session, text: d.text };
  };

  async function api(url: URL): Promise<{ type: string; body: Buffer | string }> {
    const q = url.searchParams;
    const route = url.pathname.slice('/api/'.length);
    const json = (v: unknown) => ({ type: TYPES['.json']!, body: JSON.stringify(v) });

    if (route === 'health') return json({ ok: true });
    if (route === 'manifest') return json(data.manifest);
    if (route.startsWith('buffers/')) {
      const buf = buffers[route.slice('buffers/'.length)];
      if (!buf) throw new HttpError(404, 'No such buffer');
      return { type: 'application/octet-stream', body: buf };
    }
    if (route === 'search') {
      const query = (q.get('q') ?? '').trim();
      if (query.length === 0) throw new HttpError(400, 'q is required');
      if (query.length > 512) throw new HttpError(400, 'q is too long (512 characters max)');
      const mode = (q.get('mode') ?? 'hybrid') as Mode;
      if (mode !== 'hybrid' && mode !== 'dense' && mode !== 'bm25' && mode !== 'recent') throw new HttpError(400, 'mode must be hybrid, dense, bm25 or recent');
      const k = intParam(q, 'k', 12, 1, 50);
      // Near-copies are folded unless the caller asks for the plain ranking with diverse=0.
      const diversify = intParam(q, 'diverse', 1, 0, 1) === 1 ? DIVERSIFY_DEFAULT : undefined;
      let projectId: number | undefined;
      if (q.has('project')) {
        const p = projectByIndex[intParam(q, 'project', 0, 0, Math.max(0, projectByIndex.length - 1))];
        projectId = engine.projects.find((x) => x.name === p?.name)?.id;
      }
      const t0 = performance.now();
      let res;
      let degraded = false;
      try {
        res = await engine.search(query, { k, mode, projectId, trace: true, diversify });
      } catch {
        if (mode === 'bm25') throw new HttpError(500, 'Search failed');
        res = await engine.search(query, { k, mode: 'bm25', projectId, diversify });
        degraded = true; // embedding model unreachable: keyword search still answers
      }
      return json(shapeSearch(res, describe, data.rankOf, { ms: performance.now() - t0, mode: degraded ? 'bm25' : mode, degraded }));
    }
    if (route === 'repeats') return json(opts.repeats ?? { available: false, reason: 'Not computed.' });
    if (route === 'profile') return json(opts.profile ?? { available: false, reason: 'Not computed.' });
    if (route === 'chunk') {
      const particle = intParam(q, 'particle', -1, 0, Math.max(0, engine.n - 1));
      if (particle < 0 || !q.has('particle')) throw new HttpError(400, 'particle is required');
      return json(shapeChunk(data.nodeAt[particle]!, engine.hnsw, describe));
    }
    throw new HttpError(404, 'Not found');
  }

  function staticFile(pathname: string): { type: string; body: Buffer } {
    let rel: string;
    try {
      rel = decodeURIComponent(pathname);
    } catch {
      throw new HttpError(400, 'Bad path');
    }
    if (rel.includes('\0')) throw new HttpError(400, 'Bad path');
    if (rel === '/' || rel === '') rel = '/index.html';
    const full = path.normalize(path.join(DIST, rel));
    // Confinement: the resolved path must stay inside the build directory.
    if (!full.startsWith(DIST)) throw new HttpError(403, 'Forbidden');
    let st: fs.Stats;
    try {
      st = fs.statSync(full);
    } catch {
      throw new HttpError(404, 'Not found');
    }
    if (!st.isFile()) throw new HttpError(404, 'Not found');
    const type = TYPES[path.extname(full).toLowerCase()];
    if (!type) throw new HttpError(404, 'Not found');
    return { type, body: fs.readFileSync(full) };
  }

  return async (req: http.IncomingMessage, res: http.ServerResponse): Promise<void> => {
    const send = (status: number, type: string, body: Buffer | string, extra: Record<string, string> = {}) => {
      const buf = typeof body === 'string' ? Buffer.from(body) : body;
      res.writeHead(status, {
        'content-type': type,
        'content-length': String(buf.length),
        'content-security-policy': CSP,
        'x-content-type-options': 'nosniff',
        'x-frame-options': 'DENY',
        'referrer-policy': 'no-referrer',
        'cross-origin-resource-policy': 'same-origin',
        'cross-origin-opener-policy': 'same-origin',
        'permissions-policy': 'camera=(self), microphone=(), geolocation=()',
        'cache-control': 'no-store',
        ...extra,
      });
      res.end(req.method === 'HEAD' ? undefined : buf);
    };
    const fail = (status: number, message: string) => send(status, TYPES['.json']!, JSON.stringify({ error: message }));

    try {
      if (!hosts.has(req.headers.host ?? '')) return fail(403, 'Forbidden host');
      const origin = req.headers.origin;
      if (origin !== undefined && !origins.has(origin)) return fail(403, 'Forbidden origin');
      if (req.method !== 'GET' && req.method !== 'HEAD') return send(405, TYPES['.json']!, JSON.stringify({ error: 'Method not allowed' }), { allow: 'GET, HEAD' });
      if ((req.url ?? '').length > 4096) return fail(414, 'URL too long');
      const url = new URL(req.url ?? '/', `http://127.0.0.1:${port}`);
      if (url.pathname.startsWith('/api/')) {
        const out = await api(url);
        return send(200, out.type, out.body);
      }
      const file = staticFile(url.pathname);
      return send(200, file.type, file.body);
    } catch (err) {
      if (err instanceof HttpError) return fail(err.status, err.message);
      if (!opts.quiet) console.error(`loci serve: request failed: ${err instanceof Error ? err.message : String(err)}`);
      return fail(500, 'Internal error');
    }
  };
}

export async function startServer(cfg: Config, opts: ServerOptions = {}): Promise<RunningServer> {
  let engine: Engine;
  let model = cfg.embedModel;
  if (opts.demo) {
    // Demo data lives in a throwaway temp directory and is rebuilt on every start.
    const dir = path.join(os.tmpdir(), `loci-demo-${process.pid}`);
    const demoCfg = await buildDemoStore(dir, 6000, 7, { prose: true });
    engine = Engine.open(demoCfg, { embedder: hashEmbedder(), save: false });
    model = demoCfg.embedModel;
  } else {
    engine = Engine.open(cfg, { embedder: ollamaEmbedder(cfg.ollamaUrl, cfg.embedModel, 20_000, 0), save: true });
  }
  const data = buildViewerData(engine, { demo: opts.demo === true, model });
  // Repeats: computed on the spot for the demo (fast, no model); read from the private store otherwise.
  let repeatsReport: RepeatsReport | null = null;
  if (opts.demo) repeatsReport = await runRepeats(engine, { embedder: hashEmbedder() });
  else {
    const file = path.join(cfg.home, 'repeats.json');
    try {
      if (fs.existsSync(file)) repeatsReport = JSON.parse(fs.readFileSync(file, 'utf8')) as RepeatsReport;
    } catch {
      repeatsReport = null; // unreadable: treated as not computed
    }
  }
  const repeats = repeatsView(repeatsReport, engine, data);
  // Profile: computed on the spot for the demo (in UTC, so it is the same everywhere); read from the private store otherwise.
  let profileReport: ProfileReport | null = null;
  if (opts.demo) profileReport = runProfile(engine, { timeZone: 'UTC', repeats: repeatsReport, notTyped: templatedPrompts(engine) });
  else {
    const file = path.join(cfg.home, 'profile.json');
    try {
      if (fs.existsSync(file)) profileReport = JSON.parse(fs.readFileSync(file, 'utf8')) as ProfileReport;
    } catch {
      profileReport = null; // unreadable: treated as not computed
    }
  }
  const profile = profileView(profileReport, engine);
  const devOrigins = opts.devOrigins ?? ['http://127.0.0.1:4320', 'http://localhost:4320'];
  const handler = createHandler(engine, data, cfg.port, { ...opts, devOrigins, repeats, profile });
  const server = http.createServer((req, res) => void handler(req, res));
  server.requestTimeout = 30_000;
  server.headersTimeout = 10_000;
  await new Promise<void>((resolve, reject) => {
    server.once('error', (err: NodeJS.ErrnoException) => reject(new Error(err.code === 'EADDRINUSE' ? `Port ${cfg.port} is already in use. Set LOCI_PORT to another port.` : 'Could not start the server')));
    server.listen(cfg.port, '127.0.0.1', resolve);
  });
  const url = `http://127.0.0.1:${cfg.port}`;
  if (!opts.quiet) {
    console.log(`loci: viewing ${engine.n} chunks${opts.demo ? ' (demo data)' : ''} at ${url}`);
    if (!fs.existsSync(path.join(DIST, 'index.html'))) console.log('loci: the viewer is not built yet. Run `npm run build` first.');
  }
  return {
    port: cfg.port,
    url,
    close: () =>
      new Promise<void>((resolve) => {
        server.close(() => {
          engine.close();
          resolve();
        });
        server.closeAllConnections();
      }),
  };
}
