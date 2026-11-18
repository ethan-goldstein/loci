import type { ProfileReport } from '../src/insight/profile.ts';
import { SERIES, timeColor } from './palette.ts';

/**
 * The "where the time goes" panel: a few figures and three small charts, drawn from counts only.
 *
 * Each chart is built with DOM calls (no markup strings), carries a table for screen readers,
 * and shows the value under the pointer in a tooltip anchored to the mark itself.
 *   - rhythm: prompts by weekday and hour, one hue from dark (few) to light (many)
 *   - sessions by size: how many prompts a session had
 *   - said before: per week, the share of typed sentences that repeat an earlier session
 */

const SVG = 'http://www.w3.org/2000/svg';
const DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
const INK = '#dfe7f2';
const MUTED = '#8493a8';
const EMPTY = 'rgba(160, 185, 220, 0.07)';

function h<K extends keyof HTMLElementTagNameMap>(tag: K, className?: string, text?: string): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text !== undefined) el.textContent = text;
  return el;
}

function s(tag: string, attrs: Record<string, string | number>): SVGElement {
  const el = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v));
  return el;
}

/** A table for screen readers that says what the chart shows. */
function table(caption: string, head: string[], rows: (string | number)[][]): HTMLTableElement {
  const t = h('table', 'sr-only');
  t.append(h('caption', undefined, caption));
  const tr = h('tr');
  for (const c of head) tr.append(h('th', undefined, c));
  t.append(tr);
  for (const r of rows) {
    const row = h('tr');
    for (const c of r) row.append(h('td', undefined, String(c)));
    t.append(row);
  }
  return t;
}

const n = (x: number) => x.toLocaleString('en-US');
const day = (ms: number) => new Date(ms).toISOString().slice(0, 10);

export function renderProfile(root: HTMLElement, tip: HTMLElement, p: ProfileReport): void {
  root.replaceChildren();

  /** Shows `text` beside a mark while the pointer or keyboard focus is on it. */
  const hint = (mark: Element, text: string) => {
    const show = () => {
      const r = mark.getBoundingClientRect();
      tip.textContent = text;
      tip.hidden = false;
      const w = tip.offsetWidth;
      tip.style.left = `${Math.max(8, Math.min(window.innerWidth - w - 8, r.left + r.width / 2 - w / 2))}px`;
      tip.style.top = `${Math.max(8, r.top - tip.offsetHeight - 6)}px`;
    };
    const hide = () => {
      tip.hidden = true;
    };
    mark.addEventListener('pointerenter', show);
    mark.addEventListener('pointerleave', hide);
    mark.addEventListener('focus', show);
    mark.addEventListener('blur', hide);
  };

  // ---- figures -----------------------------------------------------------------------------
  const tiles = h('div', 'tiles');
  const tile = (value: string, label: string, note: string) => {
    const t = h('div', 'tile');
    t.append(h('div', 'tile-value', value), h('div', 'tile-label', label), h('div', 'tile-note', note));
    tiles.append(t);
  };
  tile(n(p.totals.sessions), 'sessions', `on ${n(p.totals.activeDays)} days`);
  tile(n(Math.round(p.totals.activeHours)), 'active hours', `gaps over ${p.idleMinutes} min left out`);
  tile(`${p.perSession.prompts.median} · ${p.perSession.activeMinutes.median} min`, 'typical session', 'prompts · minutes');
  const sentences = p.saidBefore?.reduce((a, w) => a + w.sentences, 0) ?? 0;
  const again = p.saidBefore?.reduce((a, w) => a + w.saidBefore, 0) ?? 0;
  tile(p.saidBefore && sentences > 0 ? `${((again / sentences) * 100).toFixed(1)}%` : 'n/a', 'said before', 'of typed sentences');
  root.append(tiles);

  // ---- rhythm ------------------------------------------------------------------------------
  root.append(h('p', 'label', `when prompts are typed (${p.timeZone})`));
  const peak = Math.max(1, ...p.rhythm.flat());
  const cell = 11;
  const gap = 2;
  const left = 30;
  const top = 12;
  const width = left + 24 * (cell + gap);
  const height = top + 7 * (cell + gap);
  const map = s('svg', { viewBox: `0 0 ${width} ${height}`, class: 'chart rhythm', role: 'img', 'aria-label': 'Prompts typed by weekday and hour. The same numbers follow as a table.' });
  for (const hour of [0, 6, 12, 18]) {
    const t = s('text', { x: left + hour * (cell + gap), y: 8, fill: MUTED, 'font-size': 8 });
    t.textContent = `${String(hour).padStart(2, '0')}:00`;
    map.append(t);
  }
  p.rhythm.forEach((row, d) => {
    const label = s('text', { x: 0, y: top + d * (cell + gap) + 9, fill: MUTED, 'font-size': 8 });
    label.textContent = DAYS[d]!;
    map.append(label);
    row.forEach((v, hour) => {
      const r = s('rect', {
        x: left + hour * (cell + gap),
        y: top + d * (cell + gap),
        width: cell,
        height: cell,
        rx: 2,
        fill: v > 0 ? timeColor(v / peak) : EMPTY,
        tabindex: v > 0 ? 0 : -1,
        'data-v': v,
      });
      hint(r, `${DAYS[d]} ${String(hour).padStart(2, '0')}:00 · ${n(v)} prompt${v === 1 ? '' : 's'}`);
      map.append(r);
    });
  });
  const scale = h('div', 'scale');
  scale.append(h('span', undefined, 'fewer'));
  for (const f of [0.1, 0.35, 0.6, 0.85, 1]) {
    const sw = h('span', 'swatch');
    sw.style.background = timeColor(f);
    scale.append(sw);
  }
  scale.append(h('span', undefined, `more (most in one hour: ${n(peak)})`));
  root.append(map, scale, table('Prompts typed by weekday and hour', ['day', ...Array.from({ length: 24 }, (_, i) => String(i))], p.rhythm.map((row, d) => [DAYS[d]!, ...row])));

  // ---- sessions by size --------------------------------------------------------------------
  root.append(h('p', 'label', 'sessions by number of prompts'));
  const most = Math.max(1, ...p.promptsHistogram.map((b) => b.sessions));
  const bars = h('div', 'bars');
  for (const b of p.promptsHistogram) {
    const row = h('div', 'bar-row');
    const track = h('div', 'bar-track');
    const bar = h('div', 'bar');
    bar.style.width = `${(b.sessions / most) * 100}%`;
    bar.style.background = SERIES[0];
    bar.tabIndex = 0;
    hint(bar, `${n(b.sessions)} session${b.sessions === 1 ? '' : 's'} with ${b.label} prompt${b.label === '1' ? '' : 's'}`);
    track.append(bar);
    row.append(h('span', 'bar-label', b.label), track, h('span', 'bar-value', n(b.sessions)));
    bars.append(row);
  }
  root.append(bars, table('Sessions by number of prompts', ['prompts', 'sessions'], p.promptsHistogram.map((b) => [b.label, b.sessions])));

  // ---- said before -------------------------------------------------------------------------
  root.append(h('p', 'label', 'share of typed sentences said before, by week'));
  const weeks = (p.saidBefore ?? []).filter((w) => w.sentences > 0);
  if (weeks.length < 2) {
    root.append(h('p', 'muted note', p.saidBefore ? 'Not enough weeks to draw.' : 'Run `loci repeats`, then `loci profile`, to add this.'));
  } else {
    const W = 330;
    const H = 86;
    const padL = 26;
    const padB = 14;
    const padT = 8;
    const rates = weeks.map((w) => w.saidBefore / w.sentences);
    const ceil = Math.max(0.05, Math.ceil(Math.max(...rates) * 20) / 20);
    const x = (i: number) => padL + (i / (weeks.length - 1)) * (W - padL - 34);
    const y = (r: number) => padT + (1 - r / ceil) * (H - padT - padB);
    const line = s('svg', { viewBox: `0 0 ${W} ${H}`, class: 'chart trend', role: 'img', 'aria-label': 'Share of typed sentences that repeat an earlier session, per week. The same numbers follow as a table.' });
    for (const r of [0, ceil]) {
      line.append(s('line', { x1: padL, x2: W - 34, y1: y(r), y2: y(r), stroke: 'rgba(160,185,220,0.16)', 'stroke-width': 1 }));
      const t = s('text', { x: padL - 4, y: y(r) + 3, fill: MUTED, 'font-size': 8, 'text-anchor': 'end' });
      t.textContent = `${Math.round(r * 100)}%`;
      line.append(t);
    }
    line.append(s('polyline', { points: rates.map((r, i) => `${x(i).toFixed(1)},${y(r).toFixed(1)}`).join(' '), fill: 'none', stroke: SERIES[0], 'stroke-width': 2, 'stroke-linejoin': 'round' }));
    rates.forEach((r, i) => {
      // A wide invisible target over each week, with a visible dot only at the last one.
      const hit = s('circle', { cx: x(i), cy: y(r), r: 6, fill: 'transparent', tabindex: 0 });
      hint(hit, `week of ${day(weeks[i]!.t)} · ${(r * 100).toFixed(1)}% (${n(weeks[i]!.saidBefore)} of ${n(weeks[i]!.sentences)})`);
      line.append(hit);
    });
    const lastX = x(rates.length - 1);
    const lastY = y(rates[rates.length - 1]!);
    line.append(s('circle', { cx: lastX, cy: lastY, r: 3, fill: SERIES[0], stroke: '#0a0e15', 'stroke-width': 2 }));
    const end = s('text', { x: lastX + 6, y: lastY + 3, fill: INK, 'font-size': 9 });
    end.textContent = `${(rates[rates.length - 1]! * 100).toFixed(0)}%`;
    line.append(end);
    for (const i of [0, weeks.length - 1]) {
      const t = s('text', { x: x(i), y: H - 2, fill: MUTED, 'font-size': 8, 'text-anchor': i === 0 ? 'start' : 'end' });
      t.textContent = day(weeks[i]!.t).slice(0, 7);
      line.append(t);
    }
    root.append(line, table('Share of typed sentences said before, by week', ['week of', 'sentences', 'said before'], weeks.map((w) => [day(w.t), w.sentences, w.saidBefore])));
    root.append(h('p', 'muted note', 'A sentence counts when the same thing was said in an earlier, different session within the eight weeks before.'));
  }

  root.append(h('p', 'muted note', `Counted from timestamps only: no text is read. The busiest project holds ${(p.concentration.topShare * 100).toFixed(0)}% of everything, the busiest three ${(p.concentration.top3Share * 100).toFixed(0)}%.`));
}
