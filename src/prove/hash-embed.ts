import type { Embedder } from '../embed/ollama.ts';
import { tokenize } from '../index/tokenize.ts';
import { gaussian, rng } from './synth.ts';

/**
 * A deterministic embedder that needs no model, for the demo and for tests. Pure TypeScript with
 * no Node dependencies, so the public demo runs it in the browser to embed what a visitor types.
 */

export const DEMO_MODEL = 'demo-hash';
export const DEMO_DIM = 64;
/** Common words that say little; they count for much less than content words. */
export const FILLER = new Set(['the', 'a', 'we', 'then', 'should', 'this', 'with', 'and', 'for', 'because', 'when', 'it', 'now', 'next', 'check', 'fix', 'add', 'so']);

function fnv1a(s: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

/**
 * Bag-of-words embedder: every token gets a fixed random direction, a text is the normalised sum
 * of its tokens' directions. Crude, but texts sharing vocabulary land near each other, which is
 * all the demo needs.
 */
export function hashEmbedder(dim = DEMO_DIM): Embedder {
  const cache = new Map<string, Float32Array>();
  const direction = (token: string): Float32Array => {
    let v = cache.get(token);
    if (!v) {
      const rand = rng(fnv1a(token));
      v = new Float32Array(dim);
      for (let i = 0; i < dim; i++) v[i] = gaussian(rand);
      cache.set(token, v);
    }
    return v;
  };
  return {
    model: DEMO_MODEL,
    async embed(texts) {
      return texts.map((text) => {
        const out = new Float32Array(dim);
        for (const tok of tokenize(text)) {
          const w = FILLER.has(tok) ? 0.12 : 1;
          const d = direction(tok);
          for (let i = 0; i < dim; i++) out[i] = out[i]! + w * d[i]!;
        }
        let norm = 0;
        for (let i = 0; i < dim; i++) norm += out[i]! * out[i]!;
        norm = Math.sqrt(norm);
        if (norm === 0) out[0] = 1;
        else for (let i = 0; i < dim; i++) out[i] = out[i]! / norm;
        return out;
      });
    },
  };
}

