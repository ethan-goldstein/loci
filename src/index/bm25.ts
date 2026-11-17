import { Heap } from './heap.ts';
import { tokenize, uniqueTokens } from './tokenize.ts';

/**
 * From-scratch BM25 over an in-memory inverted index.
 *
 * The scoring formula and constants are the ones SQLite FTS5 uses (k1 = 1.2, b = 0.75, IDF floored
 * at 1e-6), so the two can be compared directly: tests/unit/bm25.test.ts checks top-10 agreement
 * against FTS5's own `bm25()` on the same documents.
 */

export interface Scored {
  /** Document ordinal: the position of the document in the build order. */
  doc: number;
  score: number;
}

interface Posting {
  docs: Uint32Array;
  tfs: Uint16Array;
}

export const K1 = 1.2;
export const B = 0.75;

export class BM25 {
  readonly n: number;
  readonly avgLen: number;
  private readonly lens: Uint32Array;
  private readonly postings: Map<string, Posting>;
  private readonly scratch: Float64Array;

  private constructor(lens: Uint32Array, avgLen: number, postings: Map<string, Posting>) {
    this.n = lens.length;
    this.lens = lens;
    this.avgLen = avgLen;
    this.postings = postings;
    this.scratch = new Float64Array(this.n);
  }

  static build(docs: readonly string[]): BM25 {
    const lens = new Uint32Array(docs.length);
    const building = new Map<string, { docs: number[]; tfs: number[] }>();
    let total = 0;
    for (let d = 0; d < docs.length; d++) {
      const toks = tokenize(docs[d]!);
      lens[d] = toks.length;
      total += toks.length;
      const tf = new Map<string, number>();
      for (const t of toks) tf.set(t, (tf.get(t) ?? 0) + 1);
      for (const [t, f] of tf) {
        let p = building.get(t);
        if (!p) {
          p = { docs: [], tfs: [] };
          building.set(t, p);
        }
        p.docs.push(d);
        p.tfs.push(Math.min(f, 65535));
      }
    }
    const postings = new Map<string, Posting>();
    for (const [t, p] of building) postings.set(t, { docs: Uint32Array.from(p.docs), tfs: Uint16Array.from(p.tfs) });
    return new BM25(lens, docs.length ? total / docs.length : 0, postings);
  }

  get vocabulary(): number {
    return this.postings.size;
  }

  /** Number of documents containing `term`. */
  df(term: string): number {
    return this.postings.get(term)?.docs.length ?? 0;
  }

  idf(term: string): number {
    const df = this.df(term);
    const idf = Math.log((this.n - df + 0.5) / (df + 0.5));
    return idf <= 0 ? 1e-6 : idf;
  }

  /**
   * Top-`k` documents for a free-text query. Each distinct query term counts once. `allow`
   * restricts the candidate set (used for the temporal guard and project filters).
   */
  search(query: string, k: number, allow?: (doc: number) => boolean): Scored[] {
    const terms = uniqueTokens(query);
    if (terms.length === 0 || this.n === 0 || k <= 0) return [];
    const scores = this.scratch;
    const touched: number[] = [];
    for (const term of terms) {
      const p = this.postings.get(term);
      if (!p) continue;
      const idf = this.idf(term);
      const { docs, tfs } = p;
      for (let i = 0; i < docs.length; i++) {
        const d = docs[i]!;
        const f = tfs[i]!;
        const norm = f + K1 * (1 - B + (B * this.lens[d]!) / this.avgLen);
        const s = (idf * f * (K1 + 1)) / norm;
        if (scores[d] === 0) touched.push(d);
        scores[d] = scores[d]! + s;
      }
    }
    // Keep the k best with a min-heap of size k.
    const heap = new Heap(false, k + 1);
    for (const d of touched) {
      const s = scores[d]!;
      scores[d] = 0; // reset scratch for the next query
      if (allow && !allow(d)) continue;
      if (heap.size < k) heap.push(s, d);
      else if (s > heap.peekKey()) {
        heap.pop();
        heap.push(s, d);
      }
    }
    const { ids, keys } = heap.drainAscending();
    const out: Scored[] = [];
    for (let i = ids.length - 1; i >= 0; i--) out.push({ doc: ids[i]!, score: keys[i]! });
    // Stable order for ties: higher score first, then lower ordinal.
    out.sort((a, b) => b.score - a.score || a.doc - b.doc);
    return out;
  }
}

/** FTS5 MATCH expression equivalent to `BM25.search`: each distinct term, quoted, joined by OR. */
export function ftsQuery(query: string): string | null {
  const terms = uniqueTokens(query);
  if (terms.length === 0) return null;
  return terms.map((t) => `"${t.replaceAll('"', '""')}"`).join(' OR ');
}
