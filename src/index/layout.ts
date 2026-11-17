/**
 * Low-dimensional layout of a nearest-neighbour graph, written from scratch.
 *
 * This is the optimisation step of UMAP (McInnes, Healy and Melville, 2018) applied directly to
 * the HNSW index's layer-0 graph: neighbours attract, a few randomly sampled non-neighbours repel
 * (negative sampling), and the step size anneals to zero. It is not a full UMAP: every graph edge
 * has the same weight, where UMAP would weight edges by a fuzzy membership strength.
 *
 * Deterministic for a given seed. The viewer starts particles at their PCA positions and eases
 * them into this layout, so what settles on screen is the index's own neighbour structure.
 */

export interface LayoutOptions {
  epochs?: number;
  negatives?: number;
  seed?: number;
  /** Curve parameters of UMAP's low-dimensional similarity; these are its values for min_dist = 0.1. */
  a?: number;
  b?: number;
}

const CLIP = 4;
const clip = (v: number) => (v > CLIP ? CLIP : v < -CLIP ? -CLIP : v);

/**
 * @param init   n * dims starting coordinates (e.g. PCA), any scale
 * @param knn    n * k neighbour indices; an entry equal to its own row is padding and is skipped
 * @returns      n * dims coordinates, centred, with RMS radius 1
 */
export function graphLayout(init: Float32Array, knn: Uint32Array, n: number, k: number, dims = 3, opts: LayoutOptions = {}): Float32Array {
  const epochs = opts.epochs ?? 300;
  const negatives = opts.negatives ?? 5;
  const a = opts.a ?? 1.577;
  const b = opts.b ?? 0.895;
  const y = new Float64Array(n * dims);
  if (n === 0) return new Float32Array(0);

  // UMAP's convention: rescale the initialisation into a box of side 10.
  for (let d = 0; d < dims; d++) {
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = 0; i < n; i++) {
      const v = init[i * dims + d]!;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    const span = hi - lo || 1;
    for (let i = 0; i < n; i++) y[i * dims + d] = (10 * (init[i * dims + d]! - lo)) / span;
  }

  let state = (opts.seed ?? 42) >>> 0;
  const rand = () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };

  for (let epoch = 0; epoch < epochs; epoch++) {
    const alpha = 1 - epoch / epochs;
    for (let i = 0; i < n; i++) {
      const io = i * dims;
      for (let e = 0; e < k; e++) {
        const j = knn[i * k + e]!;
        if (j === i || j >= n) continue;
        const jo = j * dims;
        let d2 = 0;
        for (let d = 0; d < dims; d++) d2 += (y[io + d]! - y[jo + d]!) ** 2;
        if (d2 > 0) {
          // Attraction along the edge.
          const g = (-2 * a * b * Math.pow(d2, b - 1)) / (1 + a * Math.pow(d2, b));
          for (let d = 0; d < dims; d++) {
            const step = clip(g * (y[io + d]! - y[jo + d]!)) * alpha;
            y[io + d] = y[io + d]! + step;
            y[jo + d] = y[jo + d]! - step;
          }
        }
        // Repulsion from a few random nodes, standing in for "everything that is not a neighbour".
        for (let s = 0; s < negatives; s++) {
          const m = Math.floor(rand() * n);
          if (m === i) continue;
          const mo = m * dims;
          let r2 = 0;
          for (let d = 0; d < dims; d++) r2 += (y[io + d]! - y[mo + d]!) ** 2;
          const g = r2 > 0 ? (2 * b) / ((0.001 + r2) * (1 + a * Math.pow(r2, b))) : 0;
          for (let d = 0; d < dims; d++) {
            const step = g > 0 ? clip(g * (y[io + d]! - y[mo + d]!)) : CLIP;
            y[io + d] = y[io + d]! + step * alpha;
          }
        }
      }
    }
  }

  // Centre and scale to RMS radius 1.
  const out = new Float32Array(n * dims);
  const mean = new Float64Array(dims);
  for (let i = 0; i < n; i++) for (let d = 0; d < dims; d++) mean[d] = mean[d]! + y[i * dims + d]!;
  let sq = 0;
  for (let i = 0; i < n; i++) {
    for (let d = 0; d < dims; d++) {
      const v = y[i * dims + d]! - mean[d]! / n;
      out[i * dims + d] = v;
      sq += v * v;
    }
  }
  const scale = 1 / (Math.sqrt(sq / n) || 1);
  for (let i = 0; i < out.length; i++) out[i] = out[i]! * scale;
  return out;
}
