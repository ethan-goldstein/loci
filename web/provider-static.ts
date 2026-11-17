import { BM25 } from '../src/index/bm25.ts';
import { Hnsw } from '../src/index/hnsw.ts';
import { hashEmbedder } from '../src/prove/hash-embed.ts';
import { searchCore } from '../src/search/core.ts';
import type { CorpusView } from '../src/search/core.ts';
import { clip, shapeChunk, shapeSearch } from '../src/serve/shape.ts';
import type { Describe } from '../src/serve/shape.ts';
import type { ChunkResponse, Corpus, Manifest, Mode, RepeatsView, SearchResponse } from './api.ts';

/**
 * The public demo's data source: the whole engine, running in the page.
 *
 * The local viewer asks a server on this machine for data and for search. The public demo has no
 * server, so it loads a prebuilt index of invented conversations and runs the same modules the
 * server runs: the same HNSW graph, the same BM25, the same rank fusion, the same code that
 * shapes a result. Only the embedder differs from a real install (a small hashing stand-in rather
 * than a language model), and it is the one the demo index was built with.
 *
 * Nothing here talks to anything but this site's own files.
 */

const BASE = `${import.meta.env.BASE_URL}demo/`;

/** Everything about the corpus that is not a typed array. Written by scripts/build-pages.ts. */
interface CorpusFile {
  texts: string[];
  /** Per node: milliseconds since the epoch, project id, role (0 typed by the person, 1 the assistant), session id. */
  ts: number[];
  project: number[];
  role: number[];
  session: number[];
  sessionTitles: Record<string, string>;
  projectNames: Record<string, string>;
  /** Project id for each entry of `manifest.projects`, in the same order. */
  projectIds: number[];
  rankOf: number[];
  nodeAt: number[];
  repeats: RepeatsView;
}

interface Loaded {
  view: CorpusView;
  file: CorpusFile;
  describe: Describe;
}

async function get(name: string): Promise<Response> {
  const res = await fetch(`${BASE}${name}`, { credentials: 'omit' });
  if (!res.ok) throw new Error(`Could not load the demo data (${name}: ${res.status})`);
  return res;
}
const getJson = async <T>(name: string): Promise<T> => (await (await get(name)).json()) as T;
const getBuffer = async (name: string): Promise<ArrayBuffer> => (await get(name)).arrayBuffer();

const isArrayOf = (v: unknown, n: number): boolean => Array.isArray(v) && v.length === n;

let manifestOnce: Promise<Manifest> | null = null;
let engineOnce: Promise<Loaded> | null = null;

function manifest(): Promise<Manifest> {
  manifestOnce ??= getJson<Manifest>('manifest.json');
  return manifestOnce;
}

/** Loads the text and both indexes. Started as soon as the picture is on screen, awaited by the first search. */
function engine(): Promise<Loaded> {
  engineOnce ??= (async () => {
    const [m, file, vectorBytes, graphBytes] = await Promise.all([manifest(), getJson<CorpusFile>('corpus.json'), getBuffer('vectors.f32'), getBuffer('hnsw.bin')]);
    const n = m.n;
    const vectors = new Float32Array(vectorBytes);
    // These arrays index each other, so check their shape before trusting them.
    const ok =
      vectors.length === n * m.dim &&
      isArrayOf(file.texts, n) &&
      isArrayOf(file.ts, n) &&
      isArrayOf(file.project, n) &&
      isArrayOf(file.role, n) &&
      isArrayOf(file.session, n) &&
      isArrayOf(file.rankOf, n) &&
      isArrayOf(file.nodeAt, n) &&
      isArrayOf(file.projectIds, m.projects.length);
    if (!ok) throw new Error('The demo data does not match its manifest');

    const hnsw = Hnsw.fromBytes(new Uint8Array(graphBytes), vectors, m.dim);
    if (hnsw.n !== n) throw new Error('The demo index does not match its manifest');
    const ts = Float64Array.from(file.ts);
    const view: CorpusView = {
      n,
      dim: m.dim,
      vectors,
      hnsw,
      bm25: BM25.build(file.texts),
      ts,
      project: Uint16Array.from(file.project),
      role: Uint8Array.from(file.role),
      sessionPk: Int32Array.from(file.session),
      newestFirst: Int32Array.from({ length: n }, (_, i) => i).sort((a, b) => ts[b]! - ts[a]! || b - a),
    };
    const describe: Describe = (node, chars) => ({
      particle: file.rankOf[node]!,
      chunkId: node + 1,
      project: file.projectNames[String(file.project[node]!)] ?? 'unknown',
      role: file.role[node] === 0 ? 'human' : 'assistant',
      date: new Date(file.ts[node]!).toISOString(),
      session: file.sessionTitles[String(file.session[node]!)] ?? null,
      text: clip(file.texts[node]!, chars),
    });
    return { view, file, describe };
  })();
  return engineOnce;
}

export async function loadCorpus(): Promise<Corpus> {
  const [m, pos, home, knn, meta, sentences, cluster] = await Promise.all([
    manifest(),
    getBuffer('pos.f32'),
    getBuffer('home.f32'),
    getBuffer('knn.u32'),
    getBuffer('meta.u32'),
    getBuffer('sentences.u8'),
    getBuffer('cluster.u8'),
  ]);
  // Fetch the rest in the background; a failure surfaces when something first needs it.
  void engine().catch(() => undefined);
  return {
    manifest: m,
    pos: new Float32Array(pos),
    home: new Float32Array(home),
    knn: new Uint32Array(knn),
    meta: new Uint32Array(meta),
    sentences: new Uint8Array(sentences),
    cluster: new Uint8Array(cluster),
  };
}

const embedder = hashEmbedder();

export async function search(q: string, mode: Mode, project: number | null): Promise<SearchResponse> {
  const { view, file, describe } = await engine();
  const query = q.trim().slice(0, 512);
  if (query.length === 0) throw new Error('Type something to search for');
  const projectId = project === null ? undefined : file.projectIds[project];
  const t0 = performance.now();
  const qvec = mode === 'bm25' ? null : ((await embedder.embed([query], 'query'))[0] ?? null);
  const res = searchCore(view, query, qvec, { k: 12, mode, projectId, trace: true });
  return shapeSearch(res, describe, file.rankOf, { ms: performance.now() - t0, mode, degraded: false });
}

export async function chunk(particle: number): Promise<ChunkResponse> {
  const { view, file, describe } = await engine();
  if (!Number.isInteger(particle) || particle < 0 || particle >= view.n) throw new Error('No such point');
  return shapeChunk(file.nodeAt[particle]!, view.hnsw, describe);
}

export async function repeats(): Promise<RepeatsView> {
  return (await engine()).file.repeats;
}
