/**
 * k-means with k-means++ seeding, written from scratch. Deterministic for a given seed.
 *
 * Used to split the viewer's layout into regions that can carry a label. It runs on the 3-D layout
 * coordinates, not on the embeddings: the point is to name what you can see on screen.
 */

export interface KMeansResult {
  /** Cluster index of each point. */
  assign: Uint16Array;
  /** Row-major `k * dims` cluster centres. */
  centroids: Float32Array;
  sizes: Uint32Array;
  iterations: number;
}

export function kmeans(points: Float32Array, n: number, dims: number, k: number, opts: { seed?: number; maxIterations?: number } = {}): KMeansResult {
  const maxIterations = opts.maxIterations ?? 40;
  k = Math.max(1, Math.min(k, n));
  let state = (opts.seed ?? 17) >>> 0;
  const rand = () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const dist2 = (i: number, c: Float32Array, co: number) => {
    let s = 0;
    for (let d = 0; d < dims; d++) s += (points[i * dims + d]! - c[co + d]!) ** 2;
    return s;
  };

  const centroids = new Float32Array(k * dims);
  const assign = new Uint16Array(n);
  const sizes = new Uint32Array(k);
  if (n === 0) return { assign, centroids, sizes, iterations: 0 };

  // k-means++: each new centre is drawn with probability proportional to its squared distance
  // from the nearest centre chosen so far, which spreads the starting centres out.
  const nearest = new Float64Array(n).fill(Infinity);
  let pick = Math.floor(rand() * n);
  for (let c = 0; c < k; c++) {
    for (let d = 0; d < dims; d++) centroids[c * dims + d] = points[pick * dims + d]!;
    let total = 0;
    for (let i = 0; i < n; i++) {
      const d2 = dist2(i, centroids, c * dims);
      if (d2 < nearest[i]!) nearest[i] = d2;
      total += nearest[i]!;
    }
    if (total === 0) pick = Math.floor(rand() * n); // every point already sits on a centre
    else {
      let target = rand() * total;
      pick = n - 1;
      for (let i = 0; i < n; i++) {
        target -= nearest[i]!;
        if (target <= 0) {
          pick = i;
          break;
        }
      }
    }
  }

  let iterations = 0;
  const sums = new Float64Array(k * dims);
  for (; iterations < maxIterations; iterations++) {
    let moved = 0;
    sums.fill(0);
    sizes.fill(0);
    for (let i = 0; i < n; i++) {
      let best = 0;
      let bestD = Infinity;
      for (let c = 0; c < k; c++) {
        const d2 = dist2(i, centroids, c * dims);
        if (d2 < bestD) {
          bestD = d2;
          best = c;
        }
      }
      if (assign[i] !== best || iterations === 0) moved++;
      assign[i] = best;
      sizes[best] = sizes[best]! + 1;
      for (let d = 0; d < dims; d++) sums[best * dims + d] = sums[best * dims + d]! + points[i * dims + d]!;
    }
    for (let c = 0; c < k; c++) {
      if (sizes[c] === 0) continue; // an empty cluster keeps its centre
      for (let d = 0; d < dims; d++) centroids[c * dims + d] = sums[c * dims + d]! / sizes[c]!;
    }
    if (moved === 0) break;
  }
  return { assign, centroids, sizes, iterations };
}
