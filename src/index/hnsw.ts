import { Heap } from './heap.ts';

/**
 * Hierarchical Navigable Small World index (Malkov and Yashunin, 2016), written from scratch on
 * typed arrays with no dependencies.
 *
 * Vectors must be L2-normalised: similarity is the dot product and distance is `1 - dot`.
 * Node ids are row numbers in the vector array, so node i is chunk id i + 1.
 *
 * What keeps it honest: tests/unit/hnsw.test.ts and `loci bench` compare every answer against an
 * exact brute-force scan (`exact()` below) and report recall, so the approximation is measured
 * rather than assumed.
 */

export interface HnswParams {
  /** Links chosen per new node, and the link cap on upper layers. */
  M: number;
  /** Link cap on layer 0. */
  M0: number;
  /** Candidate list size while building. */
  efConstruction: number;
  /** Seed for the level draw, so the same input builds the same graph. */
  seed: number;
}

export const DEFAULT_PARAMS: HnswParams = { M: 16, M0: 32, efConstruction: 200, seed: 0x10c1 };

export interface Hit {
  id: number;
  sim: number;
}

/** One step of a search: at `layer`, `node` was evaluated while expanding `parent` (-1 for the entry point). */
export type TraceStep = [layer: number, node: number, parent: number, sim: number];

export interface SearchOptions {
  /** Only nodes passing this test may be returned. Traversal still walks through the rest. */
  allow?: (id: number) => boolean;
  /** When given, filled with the steps the search took (capped at TRACE_CAP). */
  trace?: TraceStep[];
}

export const TRACE_CAP = 512;
const EMPTY = 0xffffffff;
const MAX_LEVEL = 24;
const MAGIC = 'LOCIHNSW';
const VERSION = 1;
const HEADER_BYTES = 48;

/** Dot product of two rows, unrolled by four. */
export function dot(a: Float32Array, ao: number, b: Float32Array, bo: number, dim: number): number {
  let s0 = 0;
  let s1 = 0;
  let s2 = 0;
  let s3 = 0;
  let i = 0;
  const n4 = dim - (dim & 3);
  for (; i < n4; i += 4) {
    s0 += a[ao + i]! * b[bo + i]!;
    s1 += a[ao + i + 1]! * b[bo + i + 1]!;
    s2 += a[ao + i + 2]! * b[bo + i + 2]!;
    s3 += a[ao + i + 3]! * b[bo + i + 3]!;
  }
  for (; i < dim; i++) s0 += a[ao + i]! * b[bo + i]!;
  return s0 + s1 + s2 + s3;
}

const pad4 = (n: number) => (n + 3) & ~3;

export class Hnsw {
  readonly dim: number;
  readonly params: HnswParams;
  /** Number of nodes inserted. */
  n = 0;
  entry = -1;
  maxLevel = -1;

  private vectors: Float32Array;
  private capacity = 0;
  private levels = new Uint8Array(0);
  private count0 = new Uint8Array(0);
  private links0 = new Uint32Array(0);
  /** Upper-layer links per node: for layer l >= 1, a block at (l-1)*(M+1) holding [count, ...ids]. */
  private upper: (Uint32Array | undefined)[] = [];
  private visited = new Uint32Array(0);
  private epoch = 0;
  private rng: number;
  private readonly mL: number;
  private readonly cand = new Heap(false, 256);
  private readonly res = new Heap(true, 256);

  constructor(vectors: Float32Array, dim: number, params: HnswParams = DEFAULT_PARAMS) {
    if (!Number.isInteger(dim) || dim < 1) throw new Error('dim must be a positive integer');
    if (params.M < 2 || params.M0 < params.M || params.M0 > 255) throw new Error('need 2 <= M <= M0 <= 255');
    this.vectors = vectors;
    this.dim = dim;
    this.params = { ...params };
    this.mL = 1 / Math.log(params.M);
    this.rng = params.seed >>> 0;
    this.reserve(Math.floor(vectors.length / dim));
  }

  /** Builds an index over the first `count` rows of `vectors`. */
  static build(
    vectors: Float32Array,
    dim: number,
    params: HnswParams = DEFAULT_PARAMS,
    count = Math.floor(vectors.length / dim),
    onProgress?: (done: number, total: number) => void,
  ): Hnsw {
    const index = new Hnsw(vectors, dim, params);
    index.extend(count, onProgress);
    return index;
  }

  /** Points the index at a (longer) vector array, e.g. after new chunks were embedded. */
  setVectors(vectors: Float32Array): void {
    if (Math.floor(vectors.length / this.dim) < this.n) throw new Error('vector array is shorter than the index');
    this.vectors = vectors;
    this.reserve(Math.floor(vectors.length / this.dim));
  }

  /** Inserts rows until the index holds `count` nodes. */
  extend(count: number, onProgress?: (done: number, total: number) => void): void {
    if (count * this.dim > this.vectors.length) throw new Error('not enough vectors for that many nodes');
    this.reserve(count);
    while (this.n < count) {
      this.add();
      if (onProgress && (this.n % 500 === 0 || this.n === count)) onProgress(this.n, count);
    }
  }

  private reserve(capacity: number): void {
    if (capacity <= this.capacity) return;
    const { M0 } = this.params;
    const levels = new Uint8Array(capacity);
    levels.set(this.levels);
    this.levels = levels;
    const count0 = new Uint8Array(capacity);
    count0.set(this.count0);
    this.count0 = count0;
    const links0 = new Uint32Array(capacity * M0).fill(EMPTY);
    links0.set(this.links0);
    this.links0 = links0;
    const visited = new Uint32Array(capacity);
    visited.set(this.visited);
    this.visited = visited;
    this.upper.length = capacity;
    this.capacity = capacity;
  }

  /** mulberry32: small, fast, and reproducible across runs and machines. */
  private random(): number {
    this.rng = (this.rng + 0x6d2b79f5) >>> 0;
    let t = this.rng;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  private nextEpoch(): number {
    if (this.epoch === 0xffffffff) {
      this.visited.fill(0);
      this.epoch = 0;
    }
    return ++this.epoch;
  }

  private sim(a: number, b: number): number {
    return dot(this.vectors, a * this.dim, this.vectors, b * this.dim, this.dim);
  }

  /**
   * Best-first search of one layer. Returns up to `ef` nodes ordered nearest first.
   * Not re-entrant: it uses the instance's two heaps.
   */
  private searchLayer(
    q: Float32Array,
    qo: number,
    entryIds: ArrayLike<number>,
    entryDists: ArrayLike<number>,
    ef: number,
    layer: number,
    allow?: (id: number) => boolean,
    trace?: TraceStep[],
  ): { ids: Int32Array; keys: Float64Array } {
    const { M, M0 } = this.params;
    const dim = this.dim;
    const vectors = this.vectors;
    const visited = this.visited;
    const epoch = this.nextEpoch();
    const cand = this.cand;
    const res = this.res;
    cand.clear();
    res.clear();

    for (let i = 0; i < entryIds.length; i++) {
      const id = entryIds[i]!;
      if (visited[id] === epoch) continue;
      visited[id] = epoch;
      const d = entryDists[i]!;
      cand.push(d, id);
      if (!allow || allow(id)) {
        res.push(d, id);
        if (res.size > ef) res.pop();
      }
    }

    while (cand.size > 0) {
      const cd = cand.peekKey();
      if (res.size >= ef && cd > res.peekKey()) break;
      const c = cand.pop();
      let arr: Uint32Array;
      let off: number;
      let cnt: number;
      if (layer === 0) {
        arr = this.links0;
        off = c * M0;
        cnt = this.count0[c]!;
      } else {
        arr = this.upper[c]!;
        const base = (layer - 1) * (M + 1);
        cnt = arr[base]!;
        off = base + 1;
      }
      for (let i = 0; i < cnt; i++) {
        const e = arr[off + i]!;
        if (visited[e] === epoch) continue;
        visited[e] = epoch;
        const d = 1 - dot(q, qo, vectors, e * dim, dim);
        if (res.size < ef || d < res.peekKey()) {
          cand.push(d, e);
          if (!allow || allow(e)) {
            res.push(d, e);
            if (res.size > ef) res.pop();
          }
          if (trace && trace.length < TRACE_CAP) trace.push([layer, e, c, 1 - d]);
        }
      }
    }
    return res.drainAscending();
  }

  /**
   * Neighbour selection heuristic (the paper's Algorithm 4): walk candidates nearest first and
   * keep one only if it is closer to the new node than to every neighbour already kept. This
   * spreads links across directions instead of piling them into one dense cluster.
   */
  private select(ids: ArrayLike<number>, dists: ArrayLike<number>, m: number): number[] {
    const out: number[] = [];
    for (let i = 0; i < ids.length && out.length < m; i++) {
      const e = ids[i]!;
      const de = dists[i]!;
      let keep = true;
      for (let j = 0; j < out.length; j++) {
        if (1 - this.sim(e, out[j]!) < de) {
          keep = false;
          break;
        }
      }
      if (keep) out.push(e);
    }
    return out;
  }

  private setLinks(node: number, layer: number, ids: readonly number[]): void {
    const { M, M0 } = this.params;
    if (layer === 0) {
      const off = node * M0;
      for (let i = 0; i < M0; i++) this.links0[off + i] = i < ids.length ? ids[i]! : EMPTY;
      this.count0[node] = ids.length;
    } else {
      const arr = this.upper[node]!;
      const base = (layer - 1) * (M + 1);
      arr[base] = ids.length;
      for (let i = 0; i < M; i++) arr[base + 1 + i] = i < ids.length ? ids[i]! : EMPTY;
    }
  }

  /** Adds the link `from -> to`, re-selecting `from`'s neighbours if it is already full. */
  private link(from: number, to: number, layer: number): void {
    const { M, M0 } = this.params;
    const cap = layer === 0 ? M0 : M;
    let arr: Uint32Array;
    let off: number;
    let cnt: number;
    if (layer === 0) {
      arr = this.links0;
      off = from * M0;
      cnt = this.count0[from]!;
    } else {
      arr = this.upper[from]!;
      const base = (layer - 1) * (M + 1);
      cnt = arr[base]!;
      off = base + 1;
    }
    if (cnt < cap) {
      arr[off + cnt] = to;
      if (layer === 0) this.count0[from] = cnt + 1;
      else arr[off - 1] = cnt + 1;
      return;
    }
    const pool: { id: number; d: number }[] = [{ id: to, d: 1 - this.sim(from, to) }];
    for (let i = 0; i < cnt; i++) {
      const e = arr[off + i]!;
      pool.push({ id: e, d: 1 - this.sim(from, e) });
    }
    pool.sort((a, b) => a.d - b.d || a.id - b.id);
    this.setLinks(
      from,
      layer,
      this.select(
        pool.map((p) => p.id),
        pool.map((p) => p.d),
        cap,
      ),
    );
  }

  /** One greedy step-down through a layer: move to whichever neighbour is nearer until none is. */
  private greedy(q: Float32Array, qo: number, start: number, startDist: number, layer: number, trace?: TraceStep[]): [number, number] {
    const { M } = this.params;
    let ep = start;
    let epDist = startDist;
    let moved = true;
    while (moved) {
      moved = false;
      const arr = this.upper[ep]!;
      const base = (layer - 1) * (M + 1);
      const cnt = arr[base]!;
      const from = ep;
      for (let i = 0; i < cnt; i++) {
        const e = arr[base + 1 + i]!;
        const d = 1 - dot(q, qo, this.vectors, e * this.dim, this.dim);
        if (d < epDist) {
          epDist = d;
          ep = e;
          moved = true;
        }
      }
      if (moved && trace && trace.length < TRACE_CAP) trace.push([layer, ep, from, 1 - epDist]);
    }
    return [ep, epDist];
  }

  /** Inserts the next row of the vector array and returns its node id. */
  private add(): number {
    const { M, efConstruction } = this.params;
    const q = this.n;
    const qo = q * this.dim;
    const level = Math.min(Math.floor(-Math.log(1 - this.random()) * this.mL), MAX_LEVEL);
    this.levels[q] = level;
    if (level > 0) this.upper[q] = new Uint32Array(level * (M + 1));
    this.n++;

    if (this.entry < 0) {
      this.entry = q;
      this.maxLevel = level;
      return q;
    }

    let ep = this.entry;
    let epDist = 1 - dot(this.vectors, qo, this.vectors, ep * this.dim, this.dim);
    for (let l = this.maxLevel; l > level; l--) [ep, epDist] = this.greedy(this.vectors, qo, ep, epDist, l);

    let eps: ArrayLike<number> = [ep];
    let epDists: ArrayLike<number> = [epDist];
    for (let l = Math.min(level, this.maxLevel); l >= 0; l--) {
      const found = this.searchLayer(this.vectors, qo, eps, epDists, efConstruction, l);
      const chosen = this.select(found.ids, found.keys, M);
      this.setLinks(q, l, chosen);
      for (const s of chosen) this.link(s, q, l);
      eps = found.ids;
      epDists = found.keys;
    }

    if (level > this.maxLevel) {
      this.maxLevel = level;
      this.entry = q;
    }
    return q;
  }

  /** Approximate `k` nearest neighbours of `q`. Larger `ef` trades speed for recall. */
  search(q: Float32Array, k: number, ef: number, opts: SearchOptions = {}): Hit[] {
    if (q.length !== this.dim) throw new Error(`query has dimension ${q.length}, index has ${this.dim}`);
    if (this.n === 0 || k <= 0) return [];
    const { trace, allow } = opts;
    let ep = this.entry;
    let epDist = 1 - dot(q, 0, this.vectors, ep * this.dim, this.dim);
    if (trace && trace.length < TRACE_CAP) trace.push([this.maxLevel, ep, -1, 1 - epDist]);
    for (let l = this.maxLevel; l >= 1; l--) [ep, epDist] = this.greedy(q, 0, ep, epDist, l, trace);
    const found = this.searchLayer(q, 0, [ep], [epDist], Math.max(ef, k), 0, allow, trace);
    const out: Hit[] = [];
    for (let i = 0; i < found.ids.length && out.length < k; i++) out.push({ id: found.ids[i]!, sim: 1 - found.keys[i]! });
    return out;
  }

  /** Exact `k` nearest neighbours by scanning every node. The oracle the graph is measured against. */
  exact(q: Float32Array, k: number, allow?: (id: number) => boolean): Hit[] {
    if (q.length !== this.dim) throw new Error(`query has dimension ${q.length}, index has ${this.dim}`);
    return exactTopK(this.vectors, this.dim, this.n, q, k, allow);
  }

  level(node: number): number {
    return this.levels[node]!;
  }

  /** Layer-0 neighbours of a node, nearest first. */
  neighbours(node: number): Hit[] {
    const { M0 } = this.params;
    const out: Hit[] = [];
    const cnt = this.count0[node]!;
    for (let i = 0; i < cnt; i++) {
      const e = this.links0[node * M0 + i]!;
      out.push({ id: e, sim: this.sim(node, e) });
    }
    return out.sort((a, b) => b.sim - a.sim || a.id - b.id);
  }

  /** Counts used by tests and stats. */
  stats(): { n: number; maxLevel: number; meanDegree0: number; upperNodes: number } {
    let deg = 0;
    let upperNodes = 0;
    for (let i = 0; i < this.n; i++) {
      deg += this.count0[i]!;
      if (this.levels[i]! > 0) upperNodes++;
    }
    return { n: this.n, maxLevel: this.maxLevel, meanDegree0: this.n ? deg / this.n : 0, upperNodes };
  }

  /**
   * Serialises the graph to plain bytes. Uses only typed arrays and DataView, so the same code
   * writes the file in Node and reads it back in a browser.
   */
  toBytes(): Uint8Array {
    const { M, M0, efConstruction, seed } = this.params;
    const n = this.n;
    let upperWords = 0;
    for (let i = 0; i < n; i++) upperWords += this.levels[i]! * (M + 1);
    const size = HEADER_BYTES + pad4(n) + pad4(n) + n * M0 * 4 + upperWords * 4;
    const bytes = new Uint8Array(size);
    const view = new DataView(bytes.buffer);
    for (let i = 0; i < MAGIC.length; i++) bytes[i] = MAGIC.charCodeAt(i);
    const head = [VERSION, n, this.dim, M, M0, efConstruction, this.entry, this.maxLevel, seed >>> 0, this.rng >>> 0];
    head.forEach((v, i) => view.setInt32(8 + i * 4, v | 0, true));
    let off = HEADER_BYTES;
    bytes.set(this.levels.subarray(0, n), off);
    off += pad4(n);
    bytes.set(this.count0.subarray(0, n), off);
    off += pad4(n);
    for (let i = 0; i < n * M0; i++, off += 4) view.setUint32(off, this.links0[i]!, true);
    for (let i = 0; i < n; i++) {
      const u = this.upper[i];
      if (!u) continue;
      for (let j = 0; j < u.length; j++, off += 4) view.setUint32(off, u[j]!, true);
    }
    return bytes;
  }

  /** Node convenience: the same bytes as a Buffer. */
  toBuffer(): Buffer {
    const bytes = this.toBytes();
    return Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  }

  /** Node convenience: a Buffer is a Uint8Array, so this is `fromBytes`. */
  static fromBuffer(buf: Uint8Array, vectors: Float32Array, dim: number): Hnsw {
    return Hnsw.fromBytes(buf, vectors, dim);
  }

  /** Loads a saved graph. Every link is bounds-checked, so a corrupt file fails here, not mid-search. */
  static fromBytes(bytes: Uint8Array, vectors: Float32Array, dim: number): Hnsw {
    let magic = '';
    for (let i = 0; i < Math.min(MAGIC.length, bytes.length); i++) magic += String.fromCharCode(bytes[i]!);
    if (bytes.length < HEADER_BYTES || magic !== MAGIC) throw new Error('Not a LOCI index file');
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const h = (i: number) => view.getInt32(8 + i * 4, true);
    if (h(0) !== VERSION) throw new Error('Unsupported index file version');
    const n = h(1);
    if (h(2) !== dim) throw new Error('Index file dimension does not match the vectors');
    const params: HnswParams = { M: h(3), M0: h(4), efConstruction: h(5), seed: h(8) >>> 0 };
    if (n < 0 || n * dim > vectors.length) throw new Error('Index file has more nodes than there are vectors');
    const index = new Hnsw(vectors, dim, params);
    index.reserve(n);
    index.n = n;
    index.entry = h(6);
    index.maxLevel = h(7);
    index.rng = h(9) >>> 0;
    const { M, M0 } = params;
    let off = HEADER_BYTES;
    const need = (count: number) => {
      if (off + count > bytes.length) throw new Error('Index file is truncated');
    };
    need(pad4(n) * 2 + n * M0 * 4);
    index.levels.set(bytes.subarray(off, off + n));
    off += pad4(n);
    index.count0.set(bytes.subarray(off, off + n));
    off += pad4(n);
    for (let i = 0; i < n * M0; i++, off += 4) index.links0[i] = view.getUint32(off, true);
    for (let i = 0; i < n; i++) {
      const level = index.levels[i]!;
      if (level > MAX_LEVEL) throw new Error('Index file is corrupt (level)');
      if (index.count0[i]! > M0) throw new Error('Index file is corrupt (degree)');
      for (let j = 0; j < index.count0[i]!; j++) {
        if (index.links0[i * M0 + j]! >= n) throw new Error('Index file is corrupt (link)');
      }
      if (level === 0) continue;
      const words = level * (M + 1);
      need(words * 4);
      const u = new Uint32Array(words);
      for (let j = 0; j < words; j++, off += 4) u[j] = view.getUint32(off, true);
      for (let l = 0; l < level; l++) {
        const cnt = u[l * (M + 1)]!;
        if (cnt > M) throw new Error('Index file is corrupt (upper degree)');
        for (let j = 0; j < cnt; j++) {
          const e = u[l * (M + 1) + 1 + j]!;
          if (e >= n || index.levels[e]! < l + 1) throw new Error('Index file is corrupt (upper link)');
        }
      }
      index.upper[i] = u;
    }
    if (n > 0 && (index.entry < 0 || index.entry >= n || index.levels[index.entry] !== index.maxLevel)) {
      throw new Error('Index file is corrupt (entry point)');
    }
    return index;
  }
}

/** Exact top-`k` by dot product over the first `n` rows. Ties break towards the lower id. */
export function exactTopK(
  vectors: Float32Array,
  dim: number,
  n: number,
  q: Float32Array,
  k: number,
  allow?: (id: number) => boolean,
): Hit[] {
  if (k <= 0 || n === 0) return [];
  const heap = new Heap(false, k + 1); // min-heap on similarity keeps the k largest
  for (let i = 0; i < n; i++) {
    if (allow && !allow(i)) continue;
    const s = dot(q, 0, vectors, i * dim, dim);
    if (heap.size < k) heap.push(s, i);
    else if (s > heap.peekKey()) {
      heap.pop();
      heap.push(s, i);
    }
  }
  const { ids, keys } = heap.drainAscending();
  const out: Hit[] = [];
  for (let i = ids.length - 1; i >= 0; i--) out.push({ id: ids[i]!, sim: keys[i]! });
  return out.sort((a, b) => b.sim - a.sim || a.id - b.id);
}
