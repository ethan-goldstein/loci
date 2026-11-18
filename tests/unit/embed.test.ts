import { afterEach, describe, expect, it, vi } from 'vitest';
import { ollamaEmbedder } from '../../src/embed/ollama.ts';

/** The embedding client against a stubbed server: no model and no network are involved. */

const ok = (n: number) => new Response(JSON.stringify({ embeddings: Array.from({ length: n }, () => [3, 4]) }), { status: 200 });

afterEach(() => vi.unstubAllGlobals());

describe('ollama embedding client', () => {
  it('returns one normalised vector per input', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => ok(2)));
    const out = await ollamaEmbedder('http://127.0.0.1:11434', 'some-model', 1000, 0, 1).embed(['a', 'b'], 'document');
    expect(out).toHaveLength(2);
    expect(out[0]![0]).toBeCloseTo(0.6, 6);
    expect(out[0]![1]).toBeCloseTo(0.8, 6);
  });

  it('tries again after a dropped connection and after a reply cut short', async () => {
    const fetchMock = vi
      .fn()
      .mockRejectedValueOnce(new TypeError('fetch failed'))
      .mockResolvedValueOnce(new Response('{"embeddings": [[1, 0', { status: 200 })) // truncated JSON
      .mockResolvedValueOnce(ok(1));
    vi.stubGlobal('fetch', fetchMock);
    const out = await ollamaEmbedder('http://127.0.0.1:11434', 'some-model', 1000, 2, 1).embed(['a'], 'query');
    expect(out).toHaveLength(1);
    expect(fetchMock).toHaveBeenCalledTimes(3);
  });

  it('gives up after the allowed tries, with a message that does not echo the input', async () => {
    const fetchMock = vi.fn().mockRejectedValue(new TypeError('fetch failed'));
    vi.stubGlobal('fetch', fetchMock);
    await expect(ollamaEmbedder('http://127.0.0.1:11434', 'some-model', 1000, 2, 1).embed(['a private sentence'], 'query')).rejects.toThrow('Could not reach the local Ollama server');
    expect(fetchMock).toHaveBeenCalledTimes(3);
    await expect(ollamaEmbedder('http://127.0.0.1:11434', 'some-model', 1000, 0, 1).embed(['a private sentence'], 'query')).rejects.not.toThrow('private');
  });

  it('does not ask again when the server answers with an error', async () => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({ error: 'model not found' }), { status: 404 }));
    vi.stubGlobal('fetch', fetchMock);
    await expect(ollamaEmbedder('http://127.0.0.1:11434', 'some-model', 1000, 2, 1).embed(['a'], 'query')).rejects.toThrow('Ollama embed failed (404): model not found');
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('refuses a reply with the wrong number of vectors', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => ok(1)));
    await expect(ollamaEmbedder('http://127.0.0.1:11434', 'some-model', 1000, 0, 1).embed(['a', 'b'], 'document')).rejects.toThrow('different number of embeddings');
  });
});
