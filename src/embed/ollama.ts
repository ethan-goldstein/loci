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

/** The server answered and refused. Not retried. */
class EmbedRefused extends Error {}

export function ollamaEmbedder(baseUrl: string, model: string, timeoutMs = 120_000, retries = 2, retryDelayMs = 1500): Embedder {
  return {
    model,
    async embed(texts, kind) {
      if (texts.length === 0) return [];
      const prefix = prefixFor(model, kind);
      const payload = JSON.stringify({ model, input: texts.map((t) => prefix + t), truncate: false });
      // A long run makes thousands of calls, and a busy or sleepy machine drops the odd one (a
      // reset connection, a reply cut short). Those are worth a second and third try. A reply
      // that carries an error message is not: the server is saying no, and asking again will
      // not help.
      const post = async (): Promise<number[][]> => {
        for (let attempt = 1; ; attempt++) {
          let failure: string;
          try {
            const res = await fetch(`${baseUrl}/api/embed`, {
              method: 'POST',
              headers: { 'content-type': 'application/json' },
              // truncate:false makes an over-long input fail loudly instead of being cut silently.
              body: payload,
              signal: AbortSignal.timeout(timeoutMs),
            });
            const body = (await res.json().catch(() => ({}))) as EmbedResponse;
            if (res.ok && body.embeddings) return body.embeddings;
            // Ollama's error text can echo input, so report only the status and a short reason.
            if (typeof body.error === 'string') throw new EmbedRefused(`Ollama embed failed (${res.status}): ${body.error.slice(0, 120)}`);
            failure = `Ollama embed failed (${res.status}): no embeddings returned`;
          } catch (err) {
            if (err instanceof EmbedRefused) throw err;
            failure = 'Could not reach the local Ollama server. Is `ollama serve` running?';
          }
          if (attempt > retries) throw new Error(failure);
          await new Promise((resolve) => setTimeout(resolve, retryDelayMs * attempt));
        }
      };
      const embeddings = await post();
      if (embeddings.length !== texts.length) throw new Error('Ollama returned a different number of embeddings than inputs');
      return embeddings.map((e) => {
        const v = Float32Array.from(e);
        normaliseRow(v);
        return v;
      });
    },
  };
}
