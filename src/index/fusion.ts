/**
 * Reciprocal-rank fusion (Cormack, Clarke and Buettcher, 2009). Combines ranked lists using only
 * ranks, so a BM25 score and a cosine similarity never have to be put on the same scale.
 *
 *   score(d) = sum over lists of 1 / (k + rank of d in that list), rank starting at 1
 */
export const RRF_K = 60;

export interface Fused {
  id: number;
  score: number;
  /** 1-based rank in each input list, or 0 when absent from it. */
  ranks: number[];
}

export function rrf(lists: readonly (readonly number[])[], limit: number, k = RRF_K): Fused[] {
  const acc = new Map<number, Fused>();
  lists.forEach((list, li) => {
    list.forEach((id, i) => {
      let f = acc.get(id);
      if (!f) {
        f = { id, score: 0, ranks: new Array<number>(lists.length).fill(0) };
        acc.set(id, f);
      }
      if (f.ranks[li] === 0) {
        f.ranks[li] = i + 1;
        f.score += 1 / (k + i + 1);
      }
    });
  });
  return [...acc.values()].sort((a, b) => b.score - a.score || a.id - b.id).slice(0, limit);
}
