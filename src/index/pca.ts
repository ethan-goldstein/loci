/**
 * Principal component analysis by power iteration with deflation. Only used to give the particle
 * layout a sensible starting position; the GPU force simulation does the real arranging.
 *
 * Works on the data matrix directly (X^T (X v)), so the d x d covariance matrix is never formed.
 */

export interface Projection {
  /** Row-major `n * components` coordinates, scaled so the first axis has unit standard deviation. */
  coords: Float32Array;
  /** Fraction of total variance captured by each component. */
  explained: number[];
}

export function pca(data: Float32Array, dim: number, n: number, components = 3, iterations = 40, seed = 7): Projection {
  const mean = new Float64Array(dim);
  for (let i = 0; i < n; i++) for (let j = 0; j < dim; j++) mean[j] = mean[j]! + data[i * dim + j]!;
  for (let j = 0; j < dim; j++) mean[j] = mean[j]! / Math.max(n, 1);

  let total = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < dim; j++) {
      const c = data[i * dim + j]! - mean[j]!;
      total += c * c;
    }
  }

  const basis: Float64Array[] = [];
  const explained: number[] = [];
  const proj = new Float64Array(n);
  let state = seed >>> 0;
  const rand = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296 - 0.5;
  };

  for (let c = 0; c < components; c++) {
    let v = new Float64Array(dim);
    for (let j = 0; j < dim; j++) v[j] = rand();
    let eig = 0;
    for (let it = 0; it < iterations; it++) {
      // Remove what earlier components already explain.
      for (const b of basis) {
        let p = 0;
        for (let j = 0; j < dim; j++) p += v[j]! * b[j]!;
        for (let j = 0; j < dim; j++) v[j] = v[j]! - p * b[j]!;
      }
      // proj = X v, then w = X^T proj, on centred data.
      const w = new Float64Array(dim);
      for (let i = 0; i < n; i++) {
        let p = 0;
        const off = i * dim;
        for (let j = 0; j < dim; j++) p += (data[off + j]! - mean[j]!) * v[j]!;
        proj[i] = p;
        for (let j = 0; j < dim; j++) w[j] = w[j]! + (data[off + j]! - mean[j]!) * p;
      }
      let norm = 0;
      for (let j = 0; j < dim; j++) norm += w[j]! * w[j]!;
      norm = Math.sqrt(norm);
      if (norm === 0) break;
      eig = norm;
      for (let j = 0; j < dim; j++) w[j] = w[j]! / norm;
      v = w;
    }
    basis.push(v);
    explained.push(total > 0 ? eig / total : 0);
  }

  const coords = new Float32Array(n * components);
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < components; c++) {
      let p = 0;
      const b = basis[c]!;
      for (let j = 0; j < dim; j++) p += (data[i * dim + j]! - mean[j]!) * b[j]!;
      coords[i * components + c] = p;
    }
  }
  // Scale so the first axis has unit standard deviation; keeps the relative spread of the others.
  let s = 0;
  for (let i = 0; i < n; i++) s += coords[i * components]! ** 2;
  const sd = Math.sqrt(s / Math.max(n, 1)) || 1;
  for (let i = 0; i < coords.length; i++) coords[i] = coords[i]! / sd;
  return { coords, explained };
}
