import { rng } from './synth.ts';

export interface Interval {
  value: number;
  lo: number;
  hi: number;
}

/**
 * Stratified cluster bootstrap. `strata[g]` holds the clusters of group g; each resample draws, for
 * every group, as many of that group's clusters as it has, with replacement.
 *
 * Two reasons for this shape:
 *   - Queries from one session are not independent (same topic, same afternoon), so whole sessions
 *     are resampled, not individual queries.
 *   - The statistic is a mean over projects. Resampling sessions across projects would let a small
 *     project vanish from a resample and silently change what is being averaged, which skews the
 *     interval. Resampling inside each project keeps every project present.
 *
 * A project with a single session contributes no spread, so intervals are, if anything, too narrow
 * for the smallest projects. Returns the statistic on the full data with a percentile interval.
 */
export function clusterBootstrap<T>(strata: readonly (readonly (readonly T[])[])[], stat: (sample: readonly T[]) => number, resamples = 2000, seed = 20261116, level = 0.95): Interval {
  const value = stat(strata.flat(2));
  const rand = rng(seed);
  const draws: number[] = [];
  for (let b = 0; b < resamples; b++) {
    const sample: T[] = [];
    for (const clusters of strata) {
      for (let i = 0; i < clusters.length; i++) {
        const c = clusters[Math.floor(rand() * clusters.length)]!;
        for (const x of c) sample.push(x);
      }
    }
    const s = stat(sample);
    if (Number.isFinite(s)) draws.push(s);
  }
  if (draws.length === 0) return { value, lo: value, hi: value };
  draws.sort((a, b) => a - b);
  const q = (p: number) => draws[Math.min(draws.length - 1, Math.max(0, Math.floor(p * draws.length)))] ?? value;
  return { value, lo: q((1 - level) / 2), hi: q(1 - (1 - level) / 2) };
}

/** Mean of per-group means, so one huge group cannot dominate. */
export function macroMean<T>(items: readonly T[], group: (x: T) => number, value: (x: T) => number): number {
  const sums = new Map<number, { s: number; n: number }>();
  for (const x of items) {
    const g = group(x);
    const acc = sums.get(g) ?? { s: 0, n: 0 };
    acc.s += value(x);
    acc.n++;
    sums.set(g, acc);
  }
  if (sums.size === 0) return Number.NaN;
  let total = 0;
  for (const { s, n } of sums.values()) total += s / n;
  return total / sums.size;
}

export function percentile(sorted: readonly number[], p: number): number {
  if (sorted.length === 0) return Number.NaN;
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.ceil(p * sorted.length) - 1));
  return sorted[idx]!;
}
