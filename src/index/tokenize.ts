/**
 * Tokeniser matching SQLite FTS5's default `unicode61`: tokens are runs of letters and digits,
 * lower-cased, with diacritics removed. Matching it exactly is what lets FTS5 act as an oracle for
 * the from-scratch BM25 in bm25.ts.
 */

const TOKEN = /[\p{L}\p{N}]+/gu;
const MARKS = /\p{M}+/gu;

export function tokenize(text: string): string[] {
  const folded = text.normalize('NFD').replace(MARKS, '').toLowerCase();
  return folded.match(TOKEN) ?? [];
}

/** Distinct tokens in first-seen order. Queries are scored once per distinct term. */
export function uniqueTokens(text: string): string[] {
  return [...new Set(tokenize(text))];
}
