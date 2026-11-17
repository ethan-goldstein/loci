import fs from 'node:fs';
import { paths } from './config.ts';
import type { Config } from './config.ts';
import type { Embedder } from './embed/ollama.ts';
import { Store } from './ingest/store.ts';
import type { ProjectRow, SessionRow } from './ingest/store.ts';
import { BM25 } from './index/bm25.ts';
import { DEFAULT_PARAMS, Hnsw } from './index/hnsw.ts';
import type { HnswParams } from './index/hnsw.ts';
import { loadVectors } from './index/vecfile.ts';
import { allowFor, searchCore } from './search/core.ts';
import type { EngineHit, Filter, SearchRequest, SearchResult } from './search/core.ts';

/**
 * The search engine: one place that owns the store, the vectors, both indexes and the fusion, so
 * the CLI, the HTTP server, the MCP server and the eval all exercise exactly the same code path.
 *
 * "Node" means a row of the vector file; node i is chunk id i + 1.
 */

export type { EngineHit, Filter, Mode, SearchRequest, SearchResult } from './search/core.ts';

export interface EngineOptions {
  embedder?: Embedder;
  params?: HnswParams;
  /** Persist a rebuilt or extended index. Off for read-only consumers that should never write. */
  save?: boolean;
  onProgress?: (stage: string, done: number, total: number) => void;
}

export class Engine {
  readonly store: Store;
  readonly n: number;
  readonly dim: number;
  readonly vectors: Float32Array;
  readonly hnsw: Hnsw;
  readonly bm25: BM25;
  readonly ts: Float64Array;
  readonly project: Uint16Array;
  readonly role: Uint8Array;
  readonly sessionPk: Int32Array;
  readonly texts: string[];
  readonly projects: ProjectRow[];
  readonly sessions: Map<number, SessionRow>;
  readonly indexBuildMs: number;
  /** Nodes from newest to oldest, for the recency list. */
  readonly newestFirst: Int32Array;
  private readonly embedder: Embedder | undefined;

  private constructor(init: {
    store: Store;
    n: number;
    dim: number;
    vectors: Float32Array;
    hnsw: Hnsw;
    bm25: BM25;
    ts: Float64Array;
    project: Uint16Array;
    role: Uint8Array;
    sessionPk: Int32Array;
    texts: string[];
    embedder: Embedder | undefined;
    indexBuildMs: number;
  }) {
    this.store = init.store;
    this.n = init.n;
    this.dim = init.dim;
    this.vectors = init.vectors;
    this.hnsw = init.hnsw;
    this.bm25 = init.bm25;
    this.ts = init.ts;
    this.project = init.project;
    this.role = init.role;
    this.sessionPk = init.sessionPk;
    this.texts = init.texts;
    this.embedder = init.embedder;
    this.indexBuildMs = init.indexBuildMs;
    this.newestFirst = Int32Array.from({ length: this.n }, (_, i) => i).sort((a, b) => this.ts[b]! - this.ts[a]! || b - a);
    this.projects = this.store.projects();
    this.sessions = new Map(this.store.sessions().map((s) => [s.pk, s]));
  }

  static open(cfg: Config, opts: EngineOptions = {}): Engine {
    const p = paths(cfg);
    if (!fs.existsSync(p.db)) throw new Error('No store yet. Run `loci ingest` first.');
    const store = new Store(p.db, { readOnly: true });
    const vec = loadVectors(p.vectors, cfg.embedModel);
    const chunks = store.allChunks();
    // Chunks ingested after the last `loci embed` have no vector yet; they join at the next embed.
    const n = Math.min(vec.rows, chunks.length);
    const ts = new Float64Array(n);
    const project = new Uint16Array(n);
    const role = new Uint8Array(n);
    const sessionPk = new Int32Array(n);
    const texts: string[] = new Array<string>(n);
    for (let i = 0; i < n; i++) {
      const c = chunks[i]!;
      if (c.id !== i + 1) throw new Error('Chunk ids are not contiguous. Run `loci reset`.');
      ts[i] = c.ts;
      project[i] = c.projectId;
      role[i] = c.role === 'human' ? 0 : 1;
      sessionPk[i] = c.sessionPk;
      texts[i] = c.text;
    }

    const params = opts.params ?? DEFAULT_PARAMS;
    let hnsw: Hnsw | null = null;
    let indexBuildMs = 0;
    if (fs.existsSync(p.hnsw)) {
      try {
        const loaded = Hnsw.fromBuffer(fs.readFileSync(p.hnsw), vec.data, vec.dim);
        const same = loaded.params.M === params.M && loaded.params.M0 === params.M0 && loaded.params.efConstruction === params.efConstruction;
        if (same && loaded.n <= n) hnsw = loaded;
      } catch {
        hnsw = null; // unreadable or stale: rebuild below
      }
    }
    const before = hnsw?.n ?? 0;
    if (!hnsw) hnsw = new Hnsw(vec.data, vec.dim, params);
    if (hnsw.n < n) {
      const t0 = performance.now();
      hnsw.extend(n, (done, total) => opts.onProgress?.('index', done, total));
      indexBuildMs = performance.now() - t0;
    }
    if (opts.save && hnsw.n !== before) {
      // Write-then-rename, so a reader never sees a half-written index.
      const tmp = `${p.hnsw}.${process.pid}.tmp`;
      fs.writeFileSync(tmp, hnsw.toBuffer(), { mode: 0o600 });
      fs.renameSync(tmp, p.hnsw);
    }

    const bm25 = BM25.build(texts);
    return new Engine({ store, n, dim: vec.dim, vectors: vec.data, hnsw, bm25, ts, project, role, sessionPk, texts, embedder: opts.embedder, indexBuildMs });
  }

  close(): void {
    this.store.close();
  }

  /** Builds a node predicate from a filter, or undefined when nothing is filtered. */
  allow(f: Filter): ((node: number) => boolean) | undefined {
    return allowFor(this, f);
  }

  async embedQuery(query: string): Promise<Float32Array> {
    if (!this.embedder) throw new Error('This engine was opened without an embedder');
    const [v] = await this.embedder.embed([query], 'query');
    if (!v || v.length !== this.dim) throw new Error('Query embedding has the wrong dimension. Was the model changed? Run `loci embed`.');
    return v;
  }

  async search(query: string, req: SearchRequest = {}): Promise<SearchResult> {
    const mode = req.mode ?? 'hybrid';
    const qvec = mode === 'bm25' ? null : await this.embedQuery(query);
    return this.searchWith(query, qvec, req);
  }

  /**
   * Synchronous search for callers that already hold the query vector (eval, benchmarks). The
   * ranking itself lives in search/core.ts, shared with the browser demo.
   */
  searchWith(query: string, qvec: Float32Array | null, req: SearchRequest = {}): SearchResult {
    return searchCore(this, query, qvec, req);
  }

  projectName(id: number): string {
    return this.projects.find((p) => p.id === id)?.name ?? 'unknown';
  }

  /** A hit with the fields a caller shows to a person or returns to an agent. */
  describe(hit: EngineHit, snippetChars = 600) {
    const node = hit.node;
    const session = this.sessions.get(this.sessionPk[node]!);
    const text = this.texts[node]!;
    return {
      chunkId: hit.chunkId,
      score: Number(hit.score.toFixed(5)),
      similarity: hit.sim === null ? null : Number(hit.sim.toFixed(4)),
      project: this.projectName(this.project[node]!),
      role: this.role[node] === 0 ? ('human' as const) : ('assistant' as const),
      date: new Date(this.ts[node]!).toISOString(),
      session: session?.title ?? null,
      sessionPk: this.sessionPk[node]!,
      text: text.length > snippetChars ? `${text.slice(0, snippetChars)}…` : text,
    };
  }
}
