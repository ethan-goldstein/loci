import { dot } from './hnsw.ts';

/**
 * "How similar is similar?" answered from the data rather than by a hand-picked number.
 *
 * Used twice: by the repeat-finder, to decide when two sentences say the same thing, and by search,
 * to decide when two results are near-copies of each other. Pure TypeScript, no Node dependencies.
 */

export interface Threshold {
  /** Similarity of a typical pair of unrelated rows. */
  median: number;
  /** Robust spread of that similarity (scaled median absolute deviation). */
  sigma: number;
  /** How many spreads above the median the threshold sits, after clamping. */
  z: number;
  /** The threshold itself. */
  tau: number;
  sampledPairs: number;
}

export function mulberry(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Sets the "same thing" threshold from the data. Samples pairs of rows from different groups
 * (sessions), takes the median similarity and a robust spread (scaled median absolute deviation,
 * so the genuine repeats in the tail do not inflate it), and returns median + z * spread.
 */
export function calibrate(vectors: Float32Array, dim: number, n: number, groupOf: ArrayLike<number>, z: number, seed = 11, samples = 200_000): Threshold {
  const rand = mulberry(seed);
  const sims: number[] = [];
  const want = Math.min(samples, Math.floor((n * (n - 1)) / 2));
  for (let tries = 0; sims.length < want && tries < want * 20; tries++) {
    const i = Math.floor(rand() * n);
    const j = Math.floor(rand() * n);
    if (i === j || groupOf[i] === groupOf[j]) continue;
    sims.push(dot(vectors, i * dim, vectors, j * dim, dim));
  }
  if (sims.length === 0) return { median: 0, sigma: 0, z, tau: 1, sampledPairs: 0 };
  sims.sort((a, b) => a - b);
  const median = sims[Math.floor(sims.length / 2)]!;
  const dev = sims.map((s) => Math.abs(s - median)).sort((a, b) => a - b);
  const sigma = 1.4826 * dev[Math.floor(dev.length / 2)]!;
  // Never so low that loosely related rows match, never so high that only exact copies do.
  const tau = Math.min(0.97, Math.max(0.7, median + z * sigma));
  return { median, sigma, z: sigma > 0 ? (tau - median) / sigma : z, tau, sampledPairs: sims.length };
}
