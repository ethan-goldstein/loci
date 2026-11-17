import { OTHER, SERIES, SLOTS, SPEAKER, timeColor } from './palette.ts';
import type { ColorMode } from './palette.ts';

/**
 * Activity over time: a stacked bar per week, sharing the time slider's axis.
 *
 * Form: the question is "how did the split change over time", so stacked bars (not a stacked
 * area) with the bar height carrying the count. Colours follow the entity and the current colour
 * mode, so the chart doubles as the key for the field: by project (three fixed slots plus a
 * neutral "other"), by speaker, or by time (one bar colour per week from the same ramp).
 *
 * Marks: 2px gaps (the panel surface shows through) between bars and between stacked segments, rounded
 * data-ends, a hairline baseline, month ticks in muted ink. Hover snaps to the nearest week and
 * lists every series at that week. A visually hidden table carries the same numbers.
 */

const INK_MUTED = '#898781';
const HAIRLINE = '#2c2c2a';
const WEEK = 7 * 24 * 3600;

interface Series {
  name: string;
  color: string;
}

export interface ActivityData {
  /** Unix seconds per particle, ascending. */
  ts: Uint32Array | number[];
  /** Project index per particle (0 is the largest project). */
  project: Uint32Array | number[];
  /** 0 for a prompt you typed, 1 for the assistant. */
  role: Uint32Array | number[];
  projectNames: string[];
}

export class Activity {
  private readonly canvas: HTMLCanvasElement;
  private readonly tip: HTMLElement;
  private readonly table: HTMLElement;
  private readonly ctx: CanvasRenderingContext2D;
  private readonly data: ActivityData;
  private readonly n: number;
  private readonly t0: number;
  private readonly t1: number;
  private readonly binSeconds: number;
  private readonly bins: number;
  private series: Series[] = [];
  /** counts[bin][series] */
  private counts: number[][] = [];
  private max = 1;
  private mode: ColorMode = 'project';
  private reveal = 1;
  private hoverBin = -1;
  private dragging = false;

  constructor(canvas: HTMLCanvasElement, tip: HTMLElement, table: HTMLElement, data: ActivityData, onSeek: (fraction: number) => void) {
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Could not draw the activity chart');
    this.canvas = canvas;
    this.ctx = ctx;
    this.tip = tip;
    this.table = table;
    this.data = data;
    this.n = data.ts.length;
    this.t0 = this.n ? data.ts[0]! : 0;
    this.t1 = this.n ? data.ts[this.n - 1]! : 1;
    // Weekly bars, widened if the history is long enough that weekly bars would get too thin.
    const weeks = Math.max(1, Math.ceil((this.t1 - this.t0 + 1) / WEEK));
    const per = Math.max(1, Math.ceil(weeks / 60));
    this.binSeconds = WEEK * per;
    this.bins = Math.max(1, Math.ceil((this.t1 - this.t0 + 1) / this.binSeconds));
    this.setMode('project');

    const seek = (e: PointerEvent) => onSeek(this.fractionAt(this.timeAtX(e)));
    canvas.addEventListener('pointerdown', (e) => {
      this.dragging = true;
      canvas.setPointerCapture(e.pointerId);
      seek(e);
    });
    canvas.addEventListener('pointermove', (e) => {
      if (this.dragging) seek(e);
      const bin = this.binAtX(e);
      if (bin !== this.hoverBin) {
        this.hoverBin = bin;
        this.draw();
      }
      this.showTip(e);
    });
    const end = (e: PointerEvent) => {
      this.dragging = false;
      if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
    };
    canvas.addEventListener('pointerup', end);
    canvas.addEventListener('pointercancel', end);
    canvas.addEventListener('pointerleave', () => {
      this.hoverBin = -1;
      this.tip.hidden = true;
      this.draw();
    });
    new ResizeObserver(() => this.draw()).observe(canvas);
  }

  /** Re-bins by the current colouring. Colour follows the entity, never its rank in a week. */
  setMode(mode: ColorMode): void {
    this.mode = mode;
    const { ts, project, role, projectNames } = this.data;
    let pick: (i: number) => number;
    if (mode === 'speaker') {
      this.series = [
        { name: 'You', color: SPEAKER.human },
        { name: 'Assistant', color: SPEAKER.assistant },
      ];
      pick = (i) => (role[i] === 0 ? 0 : 1);
    } else if (mode === 'time') {
      this.series = [{ name: 'Chunks', color: SERIES[0] }];
      pick = () => 0;
    } else {
      const named = projectNames.slice(0, SLOTS).map((name, i) => ({ name, color: SERIES[i]! }));
      const rest = projectNames.length - named.length;
      this.series = rest > 0 ? [...named, { name: `Other (${rest} projects)`, color: OTHER }] : named;
      pick = (i) => Math.min(project[i]!, SLOTS);
    }
    this.counts = Array.from({ length: this.bins }, () => new Array<number>(this.series.length).fill(0));
    for (let i = 0; i < this.n; i++) {
      const b = Math.min(this.bins - 1, Math.floor((ts[i]! - this.t0) / this.binSeconds));
      const row = this.counts[b]!;
      const s = pick(i);
      row[s] = row[s]! + 1;
    }
    this.max = Math.max(1, ...this.counts.map((row) => row.reduce((a, c) => a + c, 0)));
    this.fillTable();
    this.draw();
  }

  /** Where the time slider is, as a fraction of particles shown. */
  setReveal(fraction: number): void {
    if (Math.abs(fraction - this.reveal) < 1e-4) return;
    this.reveal = fraction;
    this.draw();
  }

  private revealTime(): number {
    if (this.n === 0) return this.t1;
    return this.data.ts[Math.min(this.n - 1, Math.max(0, Math.round(this.reveal * (this.n - 1))))]!;
  }

  private xOfTime(t: number, width: number): number {
    return ((t - this.t0) / Math.max(1, this.bins * this.binSeconds)) * width;
  }

  private timeAtX(e: PointerEvent): number {
    const rect = this.canvas.getBoundingClientRect();
    const f = Math.min(1, Math.max(0, (e.clientX - rect.left) / Math.max(1, rect.width)));
    return this.t0 + f * this.bins * this.binSeconds;
  }

  private binAtX(e: PointerEvent): number {
    return Math.min(this.bins - 1, Math.max(0, Math.floor((this.timeAtX(e) - this.t0) / this.binSeconds)));
  }

  /** Fraction of particles at or before time `t` (binary search; particles are in time order). */
  private fractionAt(t: number): number {
    let lo = 0;
    let hi = this.n;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (this.data.ts[mid]! <= t) lo = mid + 1;
      else hi = mid;
    }
    return this.n > 1 ? Math.min(1, Math.max(0, (lo - 1) / (this.n - 1))) : 1;
  }

  private binLabel(b: number): string {
    const day = (s: number) => new Date(s * 1000).toISOString().slice(0, 10);
    return this.binSeconds === WEEK ? `week of ${day(this.t0 + b * this.binSeconds)}` : `${day(this.t0 + b * this.binSeconds)} to ${day(this.t0 + (b + 1) * this.binSeconds - 1)}`;
  }

  private showTip(e: PointerEvent): void {
    const b = this.hoverBin;
    if (b < 0) return;
    const row = this.counts[b]!;
    const total = row.reduce((a, c) => a + c, 0);
    this.tip.replaceChildren();
    const head = document.createElement('div');
    head.className = 'tip-head';
    head.textContent = `${this.binLabel(b)} · ${total.toLocaleString('en-US')} chunks`;
    this.tip.append(head);
    // Every series at this week, largest first, so the pointer never has to land on a segment.
    row
      .map((count, s) => ({ count, s }))
      .filter((x) => x.count > 0 && this.series.length > 1)
      .sort((a, c) => c.count - a.count)
      .forEach(({ count, s }) => {
        const line = document.createElement('div');
        line.className = 'tip-row';
        const dot = document.createElement('span');
        dot.className = 'dot';
        dot.style.background = this.series[s]!.color;
        const name = document.createElement('span');
        name.textContent = this.series[s]!.name;
        const value = document.createElement('span');
        value.className = 'tip-value';
        value.textContent = count.toLocaleString('en-US');
        line.append(dot, name, value);
        this.tip.append(line);
      });
    this.tip.hidden = false;
    const rect = this.canvas.getBoundingClientRect();
    const w = this.tip.offsetWidth;
    this.tip.style.left = `${Math.min(window.innerWidth - w - 8, Math.max(8, e.clientX - w / 2))}px`;
    this.tip.style.top = `${rect.top - this.tip.offsetHeight - 8}px`;
  }

  /** The same numbers as a table, for people who are not reading the picture. */
  private fillTable(): void {
    const table = document.createElement('table');
    const caption = document.createElement('caption');
    caption.textContent = 'Chunks per period, by series';
    const headRow = document.createElement('tr');
    for (const h of ['Period', ...this.series.map((s) => s.name), 'Total']) {
      const th = document.createElement('th');
      th.scope = 'col';
      th.textContent = h;
      headRow.append(th);
    }
    table.append(caption, headRow);
    this.counts.forEach((row, b) => {
      const tr = document.createElement('tr');
      const th = document.createElement('th');
      th.scope = 'row';
      th.textContent = this.binLabel(b);
      tr.append(th);
      for (const v of [...row, row.reduce((a, c) => a + c, 0)]) {
        const td = document.createElement('td');
        td.textContent = String(v);
        tr.append(td);
      }
      table.append(tr);
    });
    this.table.replaceChildren(table);
  }

  draw(): void {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    if (w === 0 || h === 0) return;
    if (this.canvas.width !== Math.round(w * dpr) || this.canvas.height !== Math.round(h * dpr)) {
      this.canvas.width = Math.round(w * dpr);
      this.canvas.height = Math.round(h * dpr);
    }
    const ctx = this.ctx;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const axis = 13; // room for month ticks under the baseline
    const top = 13; // room for the scale note above the bars
    const plot = h - axis - top;
    const baseline = top + plot;
    const slot = w / this.bins;
    const gap = slot > 6 ? 2 : 1;
    const barW = Math.max(1, slot - gap);
    const nowX = this.xOfTime(this.revealTime(), w);

    for (let b = 0; b < this.bins; b++) {
      const row = this.counts[b]!;
      const x = b * slot + gap / 2;
      // Weeks the slider has not reached yet stay visible but recede.
      ctx.globalAlpha = x > nowX + 0.5 ? 0.28 : b === this.hoverBin ? 1 : 0.9;
      let y = baseline;
      const total = row.reduce((a, c) => a + c, 0);
      const last = row.reduce((acc, c, s) => (c > 0 ? s : acc), -1);
      row.forEach((count, s) => {
        if (count === 0) return;
        const full = (count / this.max) * plot;
        // The 2px surface gap comes out of the segment, but a segment never vanishes entirely.
        const segH = Math.max(1, full - (s === last ? 0 : gap));
        ctx.fillStyle = this.mode === 'time' ? timeColor(this.bins > 1 ? b / (this.bins - 1) : 1) : this.series[s]!.color;
        const yTop = y - full + (s === last ? 0 : gap);
        if (s === last) {
          // Rounded data-end on the top of the stack only; the base stays square on the baseline.
          const r = Math.min(4, barW / 2, segH);
          ctx.beginPath();
          ctx.roundRect(x, yTop, barW, segH, [r, r, 0, 0]);
          ctx.fill();
        } else ctx.fillRect(x, yTop, barW, segH);
        y -= full;
      });
      if (b === this.hoverBin && total > 0) {
        ctx.globalAlpha = 1;
        ctx.fillStyle = 'rgba(255,255,255,0.10)';
        ctx.fillRect(x, top, barW, plot);
      }
    }
    ctx.globalAlpha = 1;

    // Baseline hairline and month ticks: recessive, in muted ink.
    ctx.fillStyle = HAIRLINE;
    ctx.fillRect(0, baseline, w, 1);
    ctx.fillStyle = INK_MUTED;
    ctx.font = "10px 'JetBrains Mono Variable', ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.textBaseline = 'top';
    const start = new Date(this.t0 * 1000);
    const cursor = new Date(Date.UTC(start.getUTCFullYear(), start.getUTCMonth() + 1, 1));
    let lastLabelX = -100;
    while (cursor.getTime() / 1000 <= this.t0 + this.bins * this.binSeconds) {
      const x = this.xOfTime(cursor.getTime() / 1000, w);
      if (x - lastLabelX > 38 && x < w - 24) {
        ctx.fillRect(x, baseline, 1, 3);
        ctx.fillText(cursor.toLocaleString('en-US', { month: 'short', timeZone: 'UTC' }), x + 3, baseline + 2);
        lastLabelX = x;
      }
      cursor.setUTCMonth(cursor.getUTCMonth() + 1);
    }
    // Scale: what the tallest bar stands for, above the plot.
    ctx.textBaseline = 'top';
    ctx.fillText(`chunks per ${this.binSeconds === WEEK ? 'week' : 'period'} · tallest bar ${this.max.toLocaleString('en-US')}`, 0, 0);

    // The time slider's position.
    ctx.fillStyle = '#ffd9a0';
    ctx.fillRect(Math.min(w - 1, Math.max(0, nowX)), top, 1, plot);
  }
}
