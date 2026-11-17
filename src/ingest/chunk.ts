export interface ChunkOptions {
  /** Preferred chunk size in characters. */
  target: number;
  /** Hard upper bound in characters. */
  max: number;
  /** Characters repeated from the end of one chunk at the start of the next. */
  overlap: number;
  /** Stop after this many chunks (caps huge pastes). */
  maxChunks: number;
}

export const HUMAN_CHUNKS: ChunkOptions = { target: 800, max: 1200, overlap: 100, maxChunks: 6 };
export const ASSISTANT_CHUNKS: ChunkOptions = { target: 800, max: 1200, overlap: 100, maxChunks: 40 };

/** Collapses runs of blank lines and trailing spaces so equal content hashes equally. */
export function normalise(text: string): string {
  return text
    .replace(/\r\n?/g, '\n')
    .replace(/[ \t]+\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

/**
 * Finds the best place to cut `text` at or before `limit`, preferring a paragraph break, then a
 * line break, then a sentence end, then a space. Never returns less than half of `floor`.
 */
function cutPoint(text: string, floor: number, limit: number): number {
  if (text.length <= limit) return text.length;
  const window = text.slice(0, limit);
  const candidates = [window.lastIndexOf('\n\n'), window.lastIndexOf('\n'), Math.max(window.lastIndexOf('. '), window.lastIndexOf('? '), window.lastIndexOf('! ')) + 1, window.lastIndexOf(' ')];
  for (const c of candidates) {
    if (c >= floor) return c;
  }
  return limit;
}

/**
 * Splits text into overlapping chunks near `target` characters and never above `max`.
 * Deterministic: the same input always yields the same chunks, which keeps `(uuid, part)` stable
 * across re-ingests.
 */
export function chunkText(raw: string, opts: ChunkOptions): string[] {
  const text = normalise(raw);
  if (text.length === 0) return [];
  if (text.length <= opts.max) return [text];

  const out: string[] = [];
  let start = 0;
  while (start < text.length && out.length < opts.maxChunks) {
    const rest = text.slice(start);
    if (rest.length <= opts.max) {
      out.push(rest.trim());
      break;
    }
    const cut = cutPoint(rest, Math.floor(opts.target / 2), opts.target);
    const piece = rest.slice(0, cut).trim();
    if (piece.length > 0) out.push(piece);
    const advance = Math.max(cut - opts.overlap, Math.floor(opts.target / 4));
    start += advance;
  }
  return out.filter((c) => c.length > 0);
}
