import type { Store } from '../ingest/store.ts';
import { appendRows, prepareVecFile } from '../index/vecfile.ts';
import type { Embedder } from './ollama.ts';

export interface EmbedReport {
  model: string;
  dim: number;
  alreadyEmbedded: number;
  embedded: number;
  total: number;
  seconds: number;
  chunksPerSecond: number;
}

/**
 * Embeds every chunk that does not yet have a vector, appending rows in chunk-id order.
 * Resumable: kill it at any point and the next run continues from the last complete row.
 */
export async function embedAll(
  store: Store,
  vecFile: string,
  embedder: Embedder,
  opts: { batch?: number; onProgress?: (done: number, total: number) => void } = {},
): Promise<EmbedReport> {
  const batch = opts.batch ?? 32;
  const total = store.chunkCount();
  const t0 = performance.now();

  // One probe call tells us the dimension before the file is created.
  const probe = await embedder.embed(['dimension probe'], 'document');
  const dim = probe[0]!.length;
  const already = prepareVecFile(vecFile, dim, embedder.model);
  if (already > total) throw new Error('Vector file has more rows than the store has chunks. Run `loci reset`.');

  let done = already;
  while (done < total) {
    const chunks = store.chunksAfter(done, batch);
    if (chunks.length === 0) break;
    // Ids are contiguous, so this batch must start exactly where the file ends.
    if (chunks[0]!.id !== done + 1) throw new Error('Chunk ids are not contiguous. Run `loci reset`.');
    const vecs = await embedder.embed(
      chunks.map((c) => c.text),
      'document',
    );
    const rows = new Float32Array(vecs.length * dim);
    vecs.forEach((v, i) => {
      if (v.length !== dim) throw new Error('Embedding dimension changed mid-run');
      rows.set(v, i * dim);
    });
    appendRows(vecFile, rows);
    done += chunks.length;
    opts.onProgress?.(done, total);
  }

  const seconds = (performance.now() - t0) / 1000;
  const embedded = done - already;
  return {
    model: embedder.model,
    dim,
    alreadyEmbedded: already,
    embedded,
    total,
    seconds: Number(seconds.toFixed(1)),
    chunksPerSecond: Number((embedded / Math.max(seconds, 1e-9)).toFixed(1)),
  };
}
