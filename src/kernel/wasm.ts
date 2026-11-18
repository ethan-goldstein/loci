/**
 * The index's inner loop as a WebAssembly module, assembled here by hand.
 *
 * Nearly all of a vector search is one operation done thousands of times: the dot product of the
 * query with a stored row. In TypeScript that is a scalar loop. WebAssembly has 128-bit SIMD,
 * which multiplies and adds four floats per instruction, and the engine can keep four such sums
 * running at once so the additions do not wait on each other.
 *
 * The module is small enough (about 300 bytes) that writing its instructions out directly is
 * clearer than adding a compiler to the project: there is no build step, no binary file in the
 * repository, and every byte below has a name. The TypeScript loop in hnsw.ts stays as the
 * fallback and as the oracle this is tested against.
 *
 * Exports:
 *   dot(a, b, bytes): f32               dot product of two rows of `bytes / 4` floats
 *   scan(q, base, bytes, n, out)        out[r] = dot(q, base + r * bytes) for r in 0..n-1
 *
 * Pure TypeScript with no Node dependencies. Nothing here runs the module; see simd.ts.
 */

// ---- encoding helpers ---------------------------------------------------------------------

/** Unsigned LEB128. */
function u(n: number): number[] {
  const out: number[] = [];
  do {
    let b = n & 0x7f;
    n >>>= 7;
    if (n !== 0) b |= 0x80;
    out.push(b);
  } while (n !== 0);
  return out;
}

/** Signed LEB128, for integer constants. */
function s(n: number): number[] {
  const out: number[] = [];
  for (;;) {
    const b = n & 0x7f;
    n >>= 7;
    if ((n === 0 && (b & 0x40) === 0) || (n === -1 && (b & 0x40) !== 0)) {
      out.push(b);
      return out;
    }
    out.push(b | 0x80);
  }
}

const name = (text: string): number[] => [...u(text.length), ...[...text].map((c) => c.charCodeAt(0))];
const section = (id: number, body: number[]): number[] => [id, ...u(body.length), ...body];
const vec = (items: number[][]): number[] => [...u(items.length), ...items.flat()];

// ---- types and instructions ---------------------------------------------------------------

const I32 = 0x7f;
const F32 = 0x7d;
const V128 = 0x7b;
const EMPTY = 0x40; // a block that leaves nothing on the stack

const block = [0x02, EMPTY];
const loop = [0x03, EMPTY];
const end = [0x0b];
const br = (depth: number) => [0x0c, ...u(depth)];
const brIf = (depth: number) => [0x0d, ...u(depth)];
const call = (fn: number) => [0x10, ...u(fn)];
const get = (local: number) => [0x20, ...u(local)];
const set = (local: number) => [0x21, ...u(local)];
const i32 = (n: number) => [0x41, ...s(n)];
const i32Add = [0x6a];
const i32Mul = [0x6c];
const i32Shl = [0x74];
const i32GtU = [0x4b];
const i32GeU = [0x4f];
/** Memory access immediates: alignment as a power of two (a hint), then a byte offset. */
const at = (offset: number) => [0x02, ...u(offset)];
const f32Load = (offset = 0) => [0x2a, ...at(offset)];
const f32Store = (offset = 0) => [0x38, ...at(offset)];
const f32Add = [0x92];
const f32Mul = [0x94];
// SIMD instructions carry the 0xfd prefix.
const v128Load = (offset = 0) => [0xfd, 0x00, ...at(offset)];
const f32x4Lane = (lane: number) => [0xfd, 0x1f, lane];
const f32x4Add = [0xfd, ...u(0xe4)];
const f32x4Mul = [0xfd, ...u(0xe6)];

// ---- dot(a, b, bytes) ---------------------------------------------------------------------

// Parameters and locals of `dot`, by index.
const A = 0;
const B = 1;
const BYTES = 2;
const I = 3; // byte offset into both rows
const ACC = [4, 5, 6, 7]; // four running sums of four floats each
const SUM = 8; // the scalar result

/** acc += a[i + offset .. +4] * b[i + offset .. +4], four floats at once. */
const step = (acc: number, offset: number) => [
  ...get(acc),
  ...get(A), ...get(I), ...i32Add, ...v128Load(offset),
  ...get(B), ...get(I), ...i32Add, ...v128Load(offset),
  ...f32x4Mul,
  ...f32x4Add,
  ...set(acc),
];

/** while (i + width <= bytes) { body; i += width } */
const whileRoom = (width: number, body: number[]) => [
  ...block,
  ...loop,
  ...get(I), ...i32(width), ...i32Add, ...get(BYTES), ...i32GtU, ...brIf(1),
  ...body,
  ...get(I), ...i32(width), ...i32Add, ...set(I),
  ...br(0),
  ...end,
  ...end,
];

const dotBody = [
  // Sixteen floats per turn, into four independent sums.
  ...whileRoom(64, [...step(ACC[0]!, 0), ...step(ACC[1]!, 16), ...step(ACC[2]!, 32), ...step(ACC[3]!, 48)]),
  // Fold the four sums into the first.
  ...get(ACC[0]!), ...get(ACC[1]!), ...f32x4Add, ...get(ACC[2]!), ...get(ACC[3]!), ...f32x4Add, ...f32x4Add, ...set(ACC[0]!),
  // Whatever is left in blocks of four floats.
  ...whileRoom(16, step(ACC[0]!, 0)),
  // Add the four lanes together.
  ...get(ACC[0]!), ...f32x4Lane(0),
  ...get(ACC[0]!), ...f32x4Lane(1), ...f32Add,
  ...get(ACC[0]!), ...f32x4Lane(2), ...f32Add,
  ...get(ACC[0]!), ...f32x4Lane(3), ...f32Add,
  ...set(SUM),
  // The last one to three floats, one at a time.
  ...whileRoom(4, [...get(SUM), ...get(A), ...get(I), ...i32Add, ...f32Load(), ...get(B), ...get(I), ...i32Add, ...f32Load(), ...f32Mul, ...f32Add, ...set(SUM)]),
  ...get(SUM),
  ...end,
];
// Locals after the three parameters: one i32, four v128, one f32.
const dotLocals = vec([[1, I32], [4, V128], [1, F32]]);

// ---- scan(q, base, bytes, n, out) ---------------------------------------------------------

const Q = 0;
const BASE = 1;
const ROW_BYTES = 2;
const N = 3;
const OUT = 4;
const R = 5; // row counter
const DOT_FN = 0;

const scanBody = [
  ...block,
  ...loop,
  ...get(R), ...get(N), ...i32GeU, ...brIf(1),
  // out[r] = dot(q, base + r * bytes, bytes)
  ...get(OUT), ...get(R), ...i32(2), ...i32Shl, ...i32Add,
  ...get(Q),
  ...get(BASE), ...get(R), ...get(ROW_BYTES), ...i32Mul, ...i32Add,
  ...get(ROW_BYTES),
  ...call(DOT_FN),
  ...f32Store(),
  ...get(R), ...i32(1), ...i32Add, ...set(R),
  ...br(0),
  ...end,
  ...end,
  ...end,
];
const scanLocals = vec([[1, I32]]);

// ---- the module ---------------------------------------------------------------------------

const func = (locals: number[], body: number[]): number[] => {
  const code = [...locals, ...body];
  return [...u(code.length), ...code];
};

/** The bytes of the kernel module. Memory is imported as `env.memory`, so the caller owns it. */
export function kernelModule(): Uint8Array<ArrayBuffer> {
  const bytes = [
    0x00, 0x61, 0x73, 0x6d, // "\0asm"
    0x01, 0x00, 0x00, 0x00, // version 1
    // Types: dot is (i32, i32, i32) -> f32; scan is (i32 x5) -> nothing.
    ...section(1, vec([[0x60, ...vec([[I32], [I32], [I32]]), ...vec([[F32]])], [0x60, ...vec([[I32], [I32], [I32], [I32], [I32]]), ...vec([])]])),
    // Import: memory "env"."memory", at least one page.
    ...section(2, vec([[...name('env'), ...name('memory'), 0x02, 0x00, ...u(1)]])),
    // Functions: dot has type 0, scan has type 1.
    ...section(3, vec([[0], [1]])),
    // Exports.
    ...section(7, vec([[...name('dot'), 0x00, 0], [...name('scan'), 0x00, 1]])),
    // Code.
    ...section(10, vec([func(dotLocals, dotBody), func(scanLocals, scanBody)])),
  ];
  return new Uint8Array(bytes);
}
