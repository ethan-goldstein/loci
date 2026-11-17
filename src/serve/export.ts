import type { Engine } from '../engine.ts';
import { kmeans } from '../index/kmeans.ts';
import { graphLayout } from '../index/layout.ts';
import { pca } from '../index/pca.ts';
import { labelClusters, sentenceCount } from './labels.ts';

/**
 * Everything the viewer needs to draw the corpus, as flat typed arrays.
 *
 * Particles are ordered chronologically: particle 0 is the oldest chunk. That makes the time
 * scrubber a single number on the GPU (show particles below a threshold) and it means a particle
 * index reveals nothing but order.
 */

export const KNN = 8;

export interface Manifest {
  n: number;
  dim: number;
  demo: boolean;
  model: string;
  /** Side of the square texture that holds one particle per texel. */
  textureSize: number;
  knn: number;
  projects: { index: number; name: string; count: number }[];
  /** Regions of the layout, each named by its most distinctive words (may be empty). */
  clusters: { index: number; size: number; words: string[] }[];
  sessions: number;
  from: number;
  to: number;
  hnsw: { M: number; M0: number; efConstruction: number; maxLevel: number; meanDegree0: number };
}

export interface ViewerData {
  manifest: Manifest;
  /** n * 3 starting positions (PCA of the embeddings), RMS radius about 1. */
  pos: Float32Array;
  /** n * 3 settled positions: a UMAP-style layout of the index's neighbour graph, RMS radius 1. */
  home: Float32Array;
  /** n * KNN nearest neighbours from layer 0 of the HNSW graph, as particle indices. */
  knn: Uint32Array;
  /** n * 3: [unix seconds, project index, role (0 human, 1 assistant)]. */
  meta: Uint32Array;
  /** Per particle: rough number of sentences in the chunk (1 to 10), for how densely to draw it. */
  sentences: Uint8Array;
  /** Per particle: index into `manifest.clusters`. */
  cluster: Uint8Array;
  /** Engine node -> particle index. */
  rankOf: Int32Array;
  /** Particle index -> engine node. */
  nodeAt: Int32Array;
}

export function buildViewerData(engine: Engine, opts: { demo: boolean; model: string }): ViewerData {
  const n = engine.n;
  const nodeAt = Int32Array.from({ length: n }, (_, i) => i).sort((a, b) => engine.ts[a]! - engine.ts[b]! || a - b);
  const rankOf = new Int32Array(n);
  for (let r = 0; r < n; r++) rankOf[nodeAt[r]!] = r;

  // Projects by size, largest first, so colour assignment is stable and the legend reads well.
  const counts = new Map<number, number>();
  for (let i = 0; i < n; i++) counts.set(engine.project[i]!, (counts.get(engine.project[i]!) ?? 0) + 1);
  const ordered = [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0] - b[0]);
  const projectIndex = new Map(ordered.map(([id], index) => [id, index]));

  const { coords } = pca(engine.vectors, engine.dim, n, 3);
  let sq = 0;
  for (let i = 0; i < coords.length; i++) sq += coords[i]! * coords[i]!;
  const scale = 1 / (Math.sqrt(sq / Math.max(n, 1)) || 1);

  const pos = new Float32Array(n * 3);
  const knn = new Uint32Array(n * KNN);
  const meta = new Uint32Array(n * 3);
  for (let r = 0; r < n; r++) {
    const node = nodeAt[r]!;
    for (let c = 0; c < 3; c++) pos[r * 3 + c] = coords[node * 3 + c]! * scale;
    const nb = engine.hnsw.neighbours(node);
    for (let j = 0; j < KNN; j++) knn[r * KNN + j] = j < nb.length ? rankOf[nb[j]!.id]! : r; // pad with self: zero force
    meta[r * 3] = Math.floor(engine.ts[node]! / 1000);
    meta[r * 3 + 1] = projectIndex.get(engine.project[node]!)!;
    meta[r * 3 + 2] = engine.role[node]!;
  }

  const home = graphLayout(pos, knn, n, KNN, 3);

  // Regions of the settled layout, named from their own text. About one label per 320 chunks.
  const k = Math.max(1, Math.min(48, Math.round(n / 320)));
  const regions = kmeans(home, n, 3, k);
  const textAt = (r: number) => engine.texts[nodeAt[r]!]!;
  const texts = Array.from({ length: n }, (_, r) => textAt(r));
  const clusters = labelClusters(texts, regions.assign, k);
  const sentences = Uint8Array.from(texts, (t) => sentenceCount(t));
  const cluster = Uint8Array.from(regions.assign);

  const stats = engine.hnsw.stats();
  return {
    manifest: {
      n,
      dim: engine.dim,
      demo: opts.demo,
      model: opts.model,
      textureSize: Math.max(2, Math.ceil(Math.sqrt(n))),
      knn: KNN,
      projects: ordered.map(([id, count], index) => ({ index, name: engine.projectName(id), count })),
      clusters,
      sessions: new Set(engine.sessionPk).size,
      from: n ? engine.ts[nodeAt[0]!]! : 0,
      to: n ? engine.ts[nodeAt[n - 1]!]! : 0,
      hnsw: { M: engine.hnsw.params.M, M0: engine.hnsw.params.M0, efConstruction: engine.hnsw.params.efConstruction, maxLevel: stats.maxLevel, meanDegree0: stats.meanDegree0 },
    },
    pos,
    home,
    knn,
    meta,
    sentences,
    cluster,
    rankOf,
    nodeAt,
  };
}
