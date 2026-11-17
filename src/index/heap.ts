/**
 * Binary heap of (key, id) pairs on parallel typed arrays, so the hot loop of graph search
 * allocates nothing. `max = false` pops the smallest key first; `max = true` the largest.
 */
export class Heap {
  private keys: Float64Array;
  private ids: Int32Array;
  private n = 0;
  private readonly max: boolean;

  constructor(max: boolean, capacity = 64) {
    this.max = max;
    this.keys = new Float64Array(capacity);
    this.ids = new Int32Array(capacity);
  }

  get size(): number {
    return this.n;
  }

  clear(): void {
    this.n = 0;
  }

  /** Key at the top. Undefined behaviour when empty: check `size` first. */
  peekKey(): number {
    return this.keys[0]!;
  }

  peekId(): number {
    return this.ids[0]!;
  }

  private before(a: number, b: number): boolean {
    return this.max ? a > b : a < b;
  }

  push(key: number, id: number): void {
    if (this.n === this.keys.length) {
      const k = new Float64Array(this.n * 2);
      k.set(this.keys);
      this.keys = k;
      const i = new Int32Array(this.n * 2);
      i.set(this.ids);
      this.ids = i;
    }
    let pos = this.n++;
    const keys = this.keys;
    const ids = this.ids;
    while (pos > 0) {
      const parent = (pos - 1) >> 1;
      if (!this.before(key, keys[parent]!)) break;
      keys[pos] = keys[parent]!;
      ids[pos] = ids[parent]!;
      pos = parent;
    }
    keys[pos] = key;
    ids[pos] = id;
  }

  /** Removes the top and returns its id. The key is available from `peekKey()` beforehand. */
  pop(): number {
    const keys = this.keys;
    const ids = this.ids;
    const top = ids[0]!;
    const n = --this.n;
    if (n > 0) {
      const key = keys[n]!;
      const id = ids[n]!;
      let pos = 0;
      for (;;) {
        let child = 2 * pos + 1;
        if (child >= n) break;
        if (child + 1 < n && this.before(keys[child + 1]!, keys[child]!)) child++;
        if (!this.before(keys[child]!, key)) break;
        keys[pos] = keys[child]!;
        ids[pos] = ids[child]!;
        pos = child;
      }
      keys[pos] = key;
      ids[pos] = id;
    }
    return top;
  }

  /** Drains the heap into arrays ordered by ascending key, whatever the heap direction. */
  drainAscending(): { ids: Int32Array; keys: Float64Array } {
    const n = this.n;
    const outIds = new Int32Array(n);
    const outKeys = new Float64Array(n);
    if (this.max) {
      for (let i = n - 1; i >= 0; i--) {
        outKeys[i] = this.peekKey();
        outIds[i] = this.pop();
      }
    } else {
      for (let i = 0; i < n; i++) {
        outKeys[i] = this.peekKey();
        outIds[i] = this.pop();
      }
    }
    return { ids: outIds, keys: outKeys };
  }
}
