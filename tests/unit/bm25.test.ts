import { DatabaseSync } from 'node:sqlite';
import { describe, expect, it } from 'vitest';
import { BM25, ftsQuery } from '../../src/index/bm25.ts';
import { rrf } from '../../src/index/fusion.ts';
import { tokenize } from '../../src/index/tokenize.ts';
import { rng, syntheticDocs, SYNTH_TOPICS } from '../../src/prove/synth.ts';

const docs = syntheticDocs(3000, 21);
const texts = docs.map((d) => d.text);
const bm25 = BM25.build(texts);

describe('tokenize', () => {
  it('lower-cases, strips diacritics and splits on non-alphanumerics like FTS5 unicode61', () => {
    expect(tokenize("Café déjà-vu: it's 3.14, snake_case & CamelCase!")).toEqual([
      'cafe', 'deja', 'vu', 'it', 's', '3', '14', 'snake', 'case', 'camelcase',
    ]);
    expect(tokenize('  \n\t ')).toEqual([]);
  });
});

describe('BM25 against the SQLite FTS5 oracle', () => {
  const db = new DatabaseSync(':memory:');
  db.exec("CREATE VIRTUAL TABLE t USING fts5(text, tokenize='unicode61')");
  const ins = db.prepare('INSERT INTO t(rowid, text) VALUES (?, ?)');
  db.exec('BEGIN');
  texts.forEach((t, i) => ins.run(i + 1, t));
  db.exec('COMMIT');
  const top = db.prepare('SELECT rowid AS id, bm25(t) AS score FROM t WHERE t MATCH ? ORDER BY score, rowid LIMIT ?');

  const rand = rng(5);
  const queries: string[] = [];
  for (let i = 0; i < 150; i++) {
    const words = tokenize(texts[Math.floor(rand() * texts.length)]!);
    const n = 1 + Math.floor(rand() * 4);
    queries.push(Array.from({ length: n }, () => words[Math.floor(rand() * words.length)]!).join(' '));
  }

  it('produces the same scores as FTS5 for the same documents', () => {
    let compared = 0;
    for (const q of queries.slice(0, 40)) {
      const mine = new Map(bm25.search(q, 3000).map((s) => [s.doc + 1, s.score]));
      const rows = top.all(ftsQuery(q)!, 50) as { id: number; score: number }[];
      for (const r of rows) {
        expect(mine.get(Number(r.id))!).toBeCloseTo(-Number(r.score), 4);
        compared++;
      }
    }
    expect(compared).toBeGreaterThan(500);
  });

  it('agrees with FTS5 on the top 10 (ties aside)', () => {
    let overlap = 0;
    let total = 0;
    for (const q of queries) {
      const mine = bm25.search(q, 10);
      const rows = top.all(ftsQuery(q)!, 10) as { id: number; score: number }[];
      // Documents tied with the 10th score may legitimately swap places, so compare by score.
      const cutoff = rows.length ? -Number(rows[rows.length - 1]!.score) : 0;
      const theirs = new Set(rows.map((r) => Number(r.id)));
      for (const m of mine) {
        total++;
        if (theirs.has(m.doc + 1) || Math.abs(m.score - cutoff) < 1e-6) overlap++;
      }
    }
    expect(total).toBeGreaterThan(1000);
    expect(overlap / total).toBeGreaterThanOrEqual(0.999);
  });

  it('ranks on-topic documents first', () => {
    const hits = bm25.search('sourdough starter hydration', 20);
    expect(hits).toHaveLength(20);
    expect(hits.filter((h) => docs[h.doc]!.topic === 'kitchen').length).toBeGreaterThanOrEqual(19);
    expect(SYNTH_TOPICS).toContain('kitchen');
  });

  it('honours the allow filter and handles unknown terms', () => {
    const even = bm25.search('satellite orbit', 10, (d) => d % 2 === 0);
    expect(even.every((h) => h.doc % 2 === 0)).toBe(true);
    expect(bm25.search('zzzzunknownzzzz', 10)).toEqual([]);
    expect(bm25.search('', 10)).toEqual([]);
    // The scratch buffer must be clean after every query, filtered or not.
    expect(bm25.search('satellite orbit', 10)).toEqual(bm25.search('satellite orbit', 10));
  });
});

describe('reciprocal-rank fusion', () => {
  it('rewards agreement between lists', () => {
    const fused = rrf([[1, 2, 3, 4], [3, 1, 9, 8]], 10);
    expect(fused.map((f) => f.id).slice(0, 2).sort()).toEqual([1, 3]);
    expect(fused.find((f) => f.id === 1)!.ranks).toEqual([1, 2]);
    expect(fused.find((f) => f.id === 9)!.ranks).toEqual([0, 3]);
    expect(fused[0]!.score).toBeCloseTo(1 / 61 + 1 / 62, 10);
  });

  it('is stable on ties and respects the limit', () => {
    expect(rrf([[5], [7]], 10).map((f) => f.id)).toEqual([5, 7]);
    expect(rrf([[1, 2, 3], [4, 5, 6]], 2)).toHaveLength(2);
    expect(rrf([], 5)).toEqual([]);
  });
});
