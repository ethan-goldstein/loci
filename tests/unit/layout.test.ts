import { describe, expect, it } from 'vitest';
import { exactTopK } from '../../src/index/hnsw.ts';
import { graphLayout } from '../../src/index/layout.ts';
import { pca } from '../../src/index/pca.ts';
import { clusteredVectors } from '../../src/prove/synth.ts';

const N = 1200;
const DIM = 32;
const K = 8;
const { data, labels } = clusteredVectors(N, DIM, 6, 0.25, 3);

/** Exact k nearest neighbours of every row (excluding itself). */
function knnGraph(): Uint32Array {
  const knn = new Uint32Array(N * K);
  for (let i = 0; i < N; i++) {
    const hits = exactTopK(data, DIM, N, data.subarray(i * DIM, (i + 1) * DIM), K + 1).filter((h) => h.id !== i);
    for (let j = 0; j < K; j++) knn[i * K + j] = hits[j]?.id ?? i;
  }
  return knn;
}

const knn = knnGraph();
const init = pca(data, DIM, N, 3).coords;
const out = graphLayout(init, knn, N, K, 3, { epochs: 150 });
const dist = (y: Float32Array, a: number, b: number) => Math.hypot(y[a * 3]! - y[b * 3]!, y[a * 3 + 1]! - y[b * 3 + 1]!, y[a * 3 + 2]! - y[b * 3 + 2]!);

describe('pca', () => {
  it('orders components by explained variance and keeps clusters apart', () => {
    const { explained } = pca(data, DIM, N, 3);
    expect(explained[0]!).toBeGreaterThanOrEqual(explained[1]!);
    expect(explained[1]!).toBeGreaterThanOrEqual(explained[2]!);
    expect(explained[0]!).toBeGreaterThan(0.05);
  });
});

describe('graphLayout', () => {
  it('is centred with unit RMS radius and contains no NaN', () => {
    let sq = 0;
    const mean = [0, 0, 0];
    for (let i = 0; i < N; i++) {
      for (let d = 0; d < 3; d++) {
        const v = out[i * 3 + d]!;
        expect(Number.isFinite(v)).toBe(true);
        mean[d] = mean[d]! + v / N;
        sq += v * v;
      }
    }
    expect(Math.sqrt(sq / N)).toBeCloseTo(1, 3);
    for (const m of mean) expect(Math.abs(m)).toBeLessThan(1e-3);
  });

  it('puts graph neighbours much closer together than random pairs', () => {
    let edge = 0;
    let edges = 0;
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < K; j++) {
        edge += dist(out, i, knn[i * K + j]!);
        edges++;
      }
    }
    let random = 0;
    for (let i = 0; i < N; i++) random += dist(out, i, (i * 7919 + 13) % N);
    expect(edge / edges).toBeLessThan(0.25 * (random / N));
  });

  it('separates the clusters the data was drawn from', () => {
    // Every point should sit nearer to its own cluster's centroid than to any other's.
    const centroids = Array.from({ length: 6 }, () => [0, 0, 0, 0]);
    for (let i = 0; i < N; i++) {
      const c = centroids[labels[i]!]!;
      for (let d = 0; d < 3; d++) c[d] = c[d]! + out[i * 3 + d]!;
      c[3] = c[3]! + 1;
    }
    let correct = 0;
    for (let i = 0; i < N; i++) {
      let best = -1;
      let bestD = Infinity;
      centroids.forEach((c, ci) => {
        const d = Math.hypot(out[i * 3]! - c[0]! / c[3]!, out[i * 3 + 1]! - c[1]! / c[3]!, out[i * 3 + 2]! - c[2]! / c[3]!);
        if (d < bestD) {
          bestD = d;
          best = ci;
        }
      });
      if (best === labels[i]) correct++;
    }
    expect(correct / N).toBeGreaterThan(0.97);
  });

  it('is deterministic for a seed and handles padding and empty input', () => {
    expect(graphLayout(init, knn, N, K, 3, { epochs: 20, seed: 9 })).toEqual(graphLayout(init, knn, N, K, 3, { epochs: 20, seed: 9 }));
    const padded = new Uint32Array(4 * 2).map((_, i) => Math.floor(i / 2)); // every entry points at its own row
    const y = graphLayout(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]), padded, 4, 2, 3, { epochs: 10 });
    expect([...y].every(Number.isFinite)).toBe(true);
    expect(graphLayout(new Float32Array(0), new Uint32Array(0), 0, 8)).toHaveLength(0);
  });
});
