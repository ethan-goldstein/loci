/**
 * Seeded synthetic data. Used by unit tests, CI, scale benchmarks and the public demo mode, so no
 * real conversation ever has to leave the private store. Everything here is clearly artificial.
 */

/** mulberry32 generator returning floats in [0, 1). */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Standard normal via Box-Muller. */
export function gaussian(rand: () => number): number {
  const u = Math.max(rand(), 1e-12);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}

/**
 * Unit vectors drawn from `clusters` Gaussian blobs on the sphere. Clustered data is the realistic
 * case for embeddings and the hard case for graph indexes, so benchmarks use this rather than
 * uniform noise.
 */
export function clusteredVectors(n: number, dim: number, clusters: number, spread: number, seed: number): { data: Float32Array; labels: Uint16Array } {
  const rand = rng(seed);
  const centres = new Float32Array(clusters * dim);
  for (let i = 0; i < centres.length; i++) centres[i] = gaussian(rand);
  const data = new Float32Array(n * dim);
  const labels = new Uint16Array(n);
  for (let i = 0; i < n; i++) {
    const c = Math.floor(rand() * clusters);
    labels[i] = c;
    let norm = 0;
    for (let j = 0; j < dim; j++) {
      const v = centres[c * dim + j]! + spread * gaussian(rand);
      data[i * dim + j] = v;
      norm += v * v;
    }
    const inv = 1 / Math.sqrt(norm);
    for (let j = 0; j < dim; j++) data[i * dim + j] = data[i * dim + j]! * inv;
  }
  return { data, labels };
}

const TOPICS: Record<string, string[]> = {
  orbit: ['satellite', 'ephemeris', 'kepler', 'perigee', 'propagate', 'orbit', 'tle', 'apogee', 'inclination', 'sgp4'],
  kitchen: ['sourdough', 'starter', 'hydration', 'crumb', 'proof', 'oven', 'flour', 'levain', 'bake', 'crust'],
  parser: ['tokenizer', 'grammar', 'ast', 'lexer', 'recursive', 'descent', 'precedence', 'parse', 'syntax', 'node'],
  garden: ['tomato', 'compost', 'mulch', 'seedling', 'prune', 'trellis', 'soil', 'harvest', 'water', 'bed'],
  network: ['socket', 'latency', 'packet', 'handshake', 'retry', 'timeout', 'backoff', 'stream', 'buffer', 'tcp'],
  music: ['chord', 'tempo', 'reverb', 'melody', 'synth', 'octave', 'rhythm', 'mix', 'bass', 'scale'],
};
const FILLER = ['the', 'a', 'we', 'then', 'should', 'this', 'with', 'and', 'for', 'because', 'when', 'it', 'now', 'next', 'check', 'fix', 'add', 'so'];

export interface SynthDoc {
  topic: string;
  text: string;
  /** Milliseconds since the epoch; documents are generated in chronological order. */
  ts: number;
}

/** Invented documents: each mixes words from one topic with common filler. */
export function syntheticDocs(n: number, seed: number): SynthDoc[] {
  const rand = rng(seed);
  const names = Object.keys(TOPICS);
  const pick = <T>(xs: readonly T[]): T => xs[Math.floor(rand() * xs.length)]!;
  const out: SynthDoc[] = [];
  const start = Date.UTC(2026, 0, 1);
  for (let i = 0; i < n; i++) {
    const topic = pick(names);
    const words: string[] = [];
    const len = 12 + Math.floor(rand() * 40);
    // Words are grouped into short sentences, so the text has the shape of prose.
    let untilStop = 5 + Math.floor(rand() * 6);
    for (let w = 0; w < len; w++) {
      const word = rand() < 0.45 ? pick(TOPICS[topic]!) : pick(FILLER);
      untilStop--;
      if (untilStop === 0 || w === len - 1) {
        words.push(`${word}.`);
        untilStop = 5 + Math.floor(rand() * 6);
      } else words.push(word);
    }
    out.push({ topic, text: words.join(' '), ts: start + i * 3_600_000 });
  }
  return out;
}

export const SYNTH_TOPICS = Object.keys(TOPICS);
