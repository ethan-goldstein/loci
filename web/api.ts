/**
 * Where the viewer gets its data. Normally that is the LOCI server on this machine (same origin,
 * loopback only). In the public demo build there is no server: the same calls are answered by an
 * engine running in the page (provider-static.ts). The rest of the viewer cannot tell which.
 */

/** True only in the public demo build (`vite build --mode pages`); the code below it is dropped otherwise. */
export const STATIC = import.meta.env.MODE === 'pages';
const provider = STATIC ? import('./provider-static.ts') : null;

export interface Manifest {
  n: number;
  dim: number;
  demo: boolean;
  model: string;
  textureSize: number;
  knn: number;
  projects: { index: number; name: string; count: number }[];
  clusters: { index: number; size: number; words: string[] }[];
  sessions: number;
  from: number;
  to: number;
  hnsw: { M: number; M0: number; efConstruction: number; maxLevel: number; meanDegree0: number };
}

export interface Described {
  particle: number;
  chunkId: number;
  project: string;
  role: 'human' | 'assistant';
  date: string;
  session: string | null;
  text: string;
}

export interface Hit extends Described {
  score: number;
  similarity: number | null;
  denseRank: number;
  bm25Rank: number;
}

/** [layer, particle, parent particle or -1, similarity] */
export type TraceStep = [number, number, number, number];

export interface SearchResponse {
  ms: number;
  mode: string;
  degraded: boolean;
  dense: 'hnsw' | 'exact' | 'none';
  /** Near-copies left out of the list. */
  folded: number;
  hits: Hit[];
  trace: TraceStep[];
}

export interface ChunkResponse extends Described {
  level: number;
  neighbours: (Described & { similarity: number })[];
}

export type Mode = 'hybrid' | 'dense' | 'bm25' | 'recent';

async function getJson<T>(url: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(url, { signal, credentials: 'omit', cache: 'no-store' });
  if (!res.ok) {
    let message = `Request failed (${res.status})`;
    try {
      const body = (await res.json()) as { error?: unknown };
      if (typeof body.error === 'string') message = body.error;
    } catch {
      /* keep the generic message */
    }
    throw new Error(message);
  }
  return (await res.json()) as T;
}

async function getBuffer(name: string): Promise<ArrayBuffer> {
  const res = await fetch(`/api/buffers/${name}`, { credentials: 'omit', cache: 'no-store' });
  if (!res.ok) throw new Error(`Could not load ${name} (${res.status})`);
  return res.arrayBuffer();
}

export interface Corpus {
  manifest: Manifest;
  /** Starting positions (PCA projection). */
  pos: Float32Array;
  /** Settled positions (layout of the index's neighbour graph). */
  home: Float32Array;
  knn: Uint32Array;
  meta: Uint32Array;
  /** Rough sentence count per particle (1 to 10). */
  sentences: Uint8Array;
  /** Index into `manifest.clusters` per particle. */
  cluster: Uint8Array;
  /** Highest layer of the index graph each particle is on (0 for most). */
  level: Uint8Array;
}

export async function loadCorpus(): Promise<Corpus> {
  return checked(provider ? await (await provider).loadCorpus() : await loadFromServer());
}

async function loadFromServer(): Promise<Corpus> {
  const [manifest, pos, home, knn, meta, sentences, cluster, level] = await Promise.all([
    getJson<Manifest>('/api/manifest'),
    getBuffer('pos'),
    getBuffer('home'),
    getBuffer('knn'),
    getBuffer('meta'),
    getBuffer('sentences'),
    getBuffer('cluster'),
    getBuffer('level'),
  ]);
  return {
    manifest,
    pos: new Float32Array(pos),
    home: new Float32Array(home),
    knn: new Uint32Array(knn),
    meta: new Uint32Array(meta),
    sentences: new Uint8Array(sentences),
    cluster: new Uint8Array(cluster),
    level: new Uint8Array(level),
  };
}

/** The buffers drive GPU indexing, so check their shape before trusting them. */
function checked(corpus: Corpus): Corpus {
  const { manifest } = corpus;
  const n = manifest.n;
  const ok =
    Number.isInteger(n) &&
    n >= 0 &&
    corpus.pos.length === n * 3 &&
    corpus.home.length === n * 3 &&
    corpus.knn.length === n * manifest.knn &&
    corpus.meta.length === n * 3 &&
    corpus.sentences.length === n &&
    corpus.cluster.length === n &&
    corpus.level.length === n &&
    Array.isArray(manifest.clusters) &&
    Array.isArray(manifest.projects);
  if (!ok) throw new Error('The data does not match its manifest');
  return corpus;
}

export async function search(q: string, mode: Mode, project: number | null, signal?: AbortSignal): Promise<SearchResponse> {
  if (provider) return (await provider).search(q, mode, project);
  const params = new URLSearchParams({ q, mode, k: '12' });
  if (project !== null) params.set('project', String(project));
  return getJson<SearchResponse>(`/api/search?${params.toString()}`, signal);
}

export async function chunk(particle: number, signal?: AbortSignal): Promise<ChunkResponse> {
  if (provider) return (await provider).chunk(particle);
  return getJson<ChunkResponse>(`/api/chunk?particle=${particle}`, signal);
}

export interface RepeatItem {
  text: string;
  kind: 'instruction' | 'task' | 'question' | 'statement';
  occurrences: number;
  sessions: number;
  projects: string[];
  first: number;
  last: number;
  particles: number[];
  variants: string[];
}

export interface RepeatsView {
  available: boolean;
  reason?: string;
  minSessions?: number;
  groups?: number;
  share?: number;
  items?: RepeatItem[];
}

export interface ProfileView {
  available: boolean;
  reason?: string;
  profile?: import('../src/insight/profile.ts').ProfileReport;
}

export async function profile(signal?: AbortSignal): Promise<ProfileView> {
  if (provider) return (await provider).profile();
  return getJson<ProfileView>('/api/profile', signal);
}

export async function repeats(signal?: AbortSignal): Promise<RepeatsView> {
  if (provider) return (await provider).repeats();
  return getJson<RepeatsView>('/api/repeats', signal);
}
