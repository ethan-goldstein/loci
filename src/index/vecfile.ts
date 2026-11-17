import { createHash } from 'node:crypto';
import fs from 'node:fs';

/**
 * Flat vector file: a 32-byte header followed by append-only float32 rows.
 *
 *   bytes 0..7    magic "LOCIVEC1"
 *   bytes 8..11   dimension (uint32 LE)
 *   bytes 12..15  reserved (0)
 *   bytes 16..31  first 16 bytes of sha256(model name)
 *
 * Row r holds the embedding of chunk id r + 1. Because rows are only ever appended, an interrupted
 * embedding run resumes at `rows` with no bookkeeping beyond the file size.
 */

export const HEADER_BYTES = 32;
const MAGIC = 'LOCIVEC1';

function modelHash(model: string): Buffer {
  return createHash('sha256').update(model).digest().subarray(0, 16);
}

function header(dim: number, model: string): Buffer {
  const h = Buffer.alloc(HEADER_BYTES);
  h.write(MAGIC, 0, 'ascii');
  h.writeUInt32LE(dim, 8);
  modelHash(model).copy(h, 16);
  return h;
}

export interface VecInfo {
  dim: number;
  rows: number;
}

/** Reads the header. Returns null when the file does not exist. Throws when it is not ours. */
export function vecInfo(file: string, model?: string): VecInfo | null {
  if (!fs.existsSync(file)) return null;
  const fd = fs.openSync(file, 'r');
  try {
    const h = Buffer.alloc(HEADER_BYTES);
    const n = fs.readSync(fd, h, 0, HEADER_BYTES, 0);
    if (n < HEADER_BYTES || h.toString('ascii', 0, 8) !== MAGIC) throw new Error('Not a LOCI vector file');
    const dim = h.readUInt32LE(8);
    if (dim < 1 || dim > 8192) throw new Error('Vector file has an invalid dimension');
    if (model !== undefined && !h.subarray(16, 32).equals(modelHash(model))) {
      throw new Error('Vector file was written by a different embedding model');
    }
    const rows = Math.floor((fs.fstatSync(fd).size - HEADER_BYTES) / (4 * dim));
    return { dim, rows };
  } finally {
    fs.closeSync(fd);
  }
}

/**
 * Creates the file if needed and drops any partial trailing row left by an interrupted write.
 * Returns the number of complete rows present.
 */
export function prepareVecFile(file: string, dim: number, model: string): number {
  const info = vecInfo(file, model);
  if (!info) {
    fs.writeFileSync(file, header(dim, model), { mode: 0o600 });
    return 0;
  }
  if (info.dim !== dim) throw new Error(`Vector file has dimension ${info.dim}, model returns ${dim}`);
  const whole = HEADER_BYTES + info.rows * 4 * dim;
  if (fs.statSync(file).size !== whole) fs.truncateSync(file, whole);
  return info.rows;
}

export function appendRows(file: string, rows: Float32Array): void {
  fs.appendFileSync(file, Buffer.from(rows.buffer, rows.byteOffset, rows.byteLength));
}

export interface Vectors {
  dim: number;
  rows: number;
  /** Row-major, `rows * dim` floats. */
  data: Float32Array;
}

export function loadVectors(file: string, model?: string): Vectors {
  const info = vecInfo(file, model);
  if (!info) throw new Error('No vectors yet. Run `loci embed` first.');
  const buf = fs.readFileSync(file);
  const bytes = info.rows * info.dim * 4;
  // Copy into a fresh, aligned buffer: Buffer pools are not guaranteed to be 4-byte aligned.
  const data = new Float32Array(info.rows * info.dim);
  new Uint8Array(data.buffer).set(buf.subarray(HEADER_BYTES, HEADER_BYTES + bytes));
  return { dim: info.dim, rows: info.rows, data };
}

/** In-place L2 normalisation of one row. Returns the original norm. */
export function normaliseRow(v: Float32Array, offset = 0, dim = v.length): number {
  let s = 0;
  for (let i = 0; i < dim; i++) s += v[offset + i]! * v[offset + i]!;
  const norm = Math.sqrt(s);
  if (norm > 0) {
    const inv = 1 / norm;
    for (let i = 0; i < dim; i++) v[offset + i] = v[offset + i]! * inv;
  }
  return norm;
}
