/**
 * Keeps a result list from being the same thing twelve times.
 *
 * A history is full of near-copies: the same explanation given in three sessions, the same error
 * pasted twice. A ranker scores each on its own merits, so they arrive together and push
 * everything else off the page. Two remedies, both greedy and both starting from the ranker's
 * own order:
 *
 *   fold     skip a candidate that is a near-copy of something already picked. Nothing else is
 *            reordered, and the top result never changes. One parameter, set from the data.
 *   spread   also trade a little rank for variety (maximal marginal relevance). Stronger, and
 *            it reorders, so it can cost relevance.
 *
 * Pure TypeScript, no Node dependencies: the public demo runs it in the browser.
 */

export interface DiverseOptions {
  /** Two results at or above this similarity are near-copies. */
  dupSim: number;
  /**
   * 1 (or absent): fold only. Below 1: each pick maximises
   * lambda * rank-relevance - (1 - lambda) * similarity to what is already picked.
   */
  lambda?: number;
  /** Similarity of a typical unrelated pair, so that only similarity above it counts against a candidate. */
  baseline?: number;
}

export interface Diverse {
  /** At most k candidates, in the order to show them. */
  picked: number[];
  /** How many of the plain top k were left out as near-copies of something shown. */
  folded: number;
}

/**
 * Picks up to `k` of `ranked` (best first). If folding leaves fewer than `k`, the skipped
 * near-copies fill the remaining places in their original order, so a caller that asked for
 * twelve results still gets twelve when twelve exist.
 */
export function pickDiverse(ranked: readonly number[], k: number, sim: (a: number, b: number) => number, opts: DiverseOptions): Diverse {
  const lambda = opts.lambda ?? 1;
  const baseline = opts.baseline ?? 0;
  const window = ranked.length;
  if (window === 0 || k <= 0) return { picked: [], folded: 0 };

  const picked: number[] = [ranked[0]!];
  const used = new Uint8Array(window);
  used[0] = 1;
  /** Highest similarity of each candidate to anything picked so far. */
  const closest = new Float64Array(window).fill(-Infinity);
  const skipped: number[] = [];

  let last = ranked[0]!;
  while (picked.length < k) {
    let best = -1;
    let bestValue = -Infinity;
    for (let i = 0; i < window; i++) {
      if (used[i]) continue;
      const s = sim(ranked[i]!, last);
      if (s > closest[i]!) closest[i] = s;
      if (closest[i]! >= opts.dupSim) {
        used[i] = 1; // a near-copy of something already shown
        skipped.push(i);
        continue;
      }
      if (lambda >= 1) {
        // Fold only: the first survivor in rank order is the next pick.
        if (best < 0) best = i;
        continue;
      }
      const relevance = 1 - i / window;
      const excess = Math.max(0, (closest[i]! - baseline) / Math.max(1e-9, 1 - baseline));
      const value = lambda * relevance - (1 - lambda) * excess;
      if (value > bestValue) {
        bestValue = value;
        best = i;
      }
    }
    if (best < 0) break;
    used[best] = 1;
    last = ranked[best]!;
    picked.push(last);
  }

  // Not enough distinct results: let the best of the near-copies back in rather than return a short list.
  skipped.sort((a, b) => a - b);
  for (const i of skipped) {
    if (picked.length >= k) break;
    picked.push(ranked[i]!);
  }
  const shown = new Set(picked);
  let folded = 0;
  for (let i = 0; i < Math.min(k, window); i++) if (!shown.has(ranked[i]!)) folded++;
  return { picked, folded };
}

/** How repetitive a result list is: mean similarity over all pairs, and how many pairs are near-copies. */
export function redundancy(nodes: readonly number[], sim: (a: number, b: number) => number, dupSim: number): { meanSim: number; dupPairs: number } {
  let total = 0;
  let pairs = 0;
  let dupPairs = 0;
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const s = sim(nodes[i]!, nodes[j]!);
      total += s;
      pairs++;
      if (s >= dupSim) dupPairs++;
    }
  }
  return { meanSim: pairs ? total / pairs : 0, dupPairs };
}
