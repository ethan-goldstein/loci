import { normaliseRow } from '../index/vecfile.ts';

/**
 * Embeddings from a local Ollama server. The URL is checked to be loopback in config.ts, so chunk
 * text never leaves this machine.
 */

/** `cluster` asks for embeddings meant for comparing texts with each other, not for retrieval. */
export type EmbedKind = 'document' | 'query' | 'cluster';

export interface Embedder {
  readonly model: string;
  embed(texts: string[], kind: EmbedKind): Promise<Float32Array[]>;
}

/**
 * Task prefixes a model was trained with. nomic-embed-text expects one of these in front of every
 * input; models not listed get no prefix.
 */
const PREFIXES: Record<string, Record<EmbedKind, string>> = {
  'nomic-embed-text': { document: 'search_document: ', query: 'search_query: ', cluster: 'clustering: ' },
};

export function prefixFor(model: string, kind: EmbedKind): string {
  const base = model.split(':')[0] ?? model;
  return PREFIXES[base]?.[kind] ?? '';
}

interface EmbedResponse {
  embeddings?: number[][];
  error?: string;
}

export function ollamaEmbedder(baseUrl: string, model: string, timeoutMs = 120_000): Embedder {
  return {
    model,
    async embed(texts, kind) {
      if (texts.length === 0) return [];
      const prefix = prefixFor(model, kind);
      let res: Response;
      try {
        res = await fetch(`${baseUrl}/api/embed`, {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          // truncate:false makes an over-long input fail loudly instead of being cut silently.
          body: JSON.stringify({ model, input: texts.map((t) => prefix + t), truncate: false }),
          signal: AbortSignal.timeout(timeoutMs),
        });
      } catch {
        throw new Error('Could not reach the local Ollama server. Is `ollama serve` running?');
      }
      const body = (await res.json().catch(() => ({}))) as EmbedResponse;
      if (!res.ok || !body.embeddings) {
        // Ollama's error text can echo input, so report only the status and a short reason.
        const reason = typeof body.error === 'string' ? body.error.slice(0, 120) : 'no embeddings returned';
        throw new Error(`Ollama embed failed (${res.status}): ${reason}`);
      }
      if (body.embeddings.length !== texts.length) throw new Error('Ollama returned a different number of embeddings than inputs');
      return body.embeddings.map((e) => {
        const v = Float32Array.from(e);
        normaliseRow(v);
        return v;
      });
    },
  };
}
