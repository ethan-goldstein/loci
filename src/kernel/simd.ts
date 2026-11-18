import type { QueryKernel } from '../index/hnsw.ts';
import { kernelModule } from './wasm.ts';

/**
 * Loads the WebAssembly SIMD kernel (wasm.ts) over a set of vectors, or returns null when the
 * runtime cannot run it, in which case the caller keeps using the TypeScript loop.
 *
 * The vectors are copied into the module's memory once. A search then writes its query there and
 * every comparison is one call with two addresses.
 *
 * Used by the local engine only. The public demo never loads this file: its vectors are 64
 * numbers long, where there is little to gain, and it ships no WebAssembly at all.
 */

const PAGE = 65536;
const align = (n: number) => (n + 15) & ~15;

interface Exports {
  dot: (a: number, b: number, bytes: number) => number;
  scan: (q: number, base: number, bytes: number, n: number, out: number) => void;
}

/** True when this runtime can compile the module (it needs WebAssembly with 128-bit SIMD). */
export function simdAvailable(): boolean {
  try {
    return typeof WebAssembly === 'object' && WebAssembly.validate(kernelModule());
  } catch {
    return false;
  }
}

export function loadSimdKernel(vectors: Float32Array, dim: number, n: number): QueryKernel | null {
  if (dim <= 0 || n <= 0 || vectors.length < n * dim) return null;
  if (!simdAvailable()) return null;
  try {
    const rowBytes = dim * 4;
    // Layout: the query, then room for one similarity per row, then the rows themselves.
    const queryAt = 0;
    const outAt = align(rowBytes);
    const rowsAt = align(outAt + n * 4);
    const total = rowsAt + n * rowBytes;
    if (total > 0x7fff0000) return null; // would not fit a 32-bit memory
    const memory = new WebAssembly.Memory({ initial: Math.ceil(total / PAGE) });
    const instance = new WebAssembly.Instance(new WebAssembly.Module(kernelModule()), { env: { memory } });
    const kernel = instance.exports as unknown as Exports;
    const floats = new Float32Array(memory.buffer);
    floats.set(vectors.subarray(0, n * dim), rowsAt / 4);
    const out = floats.subarray(outAt / 4, outAt / 4 + n);

    return {
      name: 'wasm-simd',
      n,
      setQuery(q: Float32Array): void {
        floats.set(q, queryAt / 4);
      },
      toRow(node: number): number {
        return kernel.dot(queryAt, rowsAt + node * rowBytes, rowBytes);
      },
      scan(count: number, into: Float32Array): void {
        kernel.scan(queryAt, rowsAt, rowBytes, count, outAt);
        into.set(count === n ? out : out.subarray(0, count));
      },
    };
  } catch {
    return null; // any failure to compile or allocate: fall back to TypeScript
  }
}
