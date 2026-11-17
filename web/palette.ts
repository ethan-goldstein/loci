/**
 * Colours for the viewer and its chart.
 *
 * Categorical: three hues in a fixed order, stepped for a dark surface. The field is a scatter:
 * any two projects can end up side by side, so every PAIR of colours has to be distinguishable,
 * not just neighbours in a list. Checked with a palette validator against this app's surface
 * (#0a0e15) across all pairs, these three pass under normal vision and under simulated
 * protanopia and deuteranopia. Adding a fourth hue from the same family fails that check (orange
 * and green collapse under red-green colour blindness; red and orange are close for everyone),
 * so the three largest projects take the three slots and every further project shares one
 * neutral "other" colour. Any project can still be isolated by clicking it in the legend.
 *
 * Sequential: one hue (blue), dark to light, used when colouring by time.
 *
 * The shaders write straight to the canvas with no colour-space conversion, so everything here
 * stays in display (sRGB) values and matches the legend swatches exactly.
 */

export const SERIES = ['#3987e5', '#d95926', '#199e70'] as const;
export const OTHER = '#898781';
export const SLOTS = SERIES.length;

/** Oldest to newest. */
const TIME_RAMP = ['#184f95', '#256abf', '#3987e5', '#6da7ec', '#b7d3f6'] as const;

export const SPEAKER = { human: SERIES[1], assistant: SERIES[0] } as const;

export type ColorMode = 'project' | 'time' | 'speaker';
export type Rgb = [number, number, number];

export function hexRgb(hex: string): Rgb {
  const ch = (i: number) => parseInt(hex.slice(1 + i * 2, 3 + i * 2), 16) / 255;
  return [ch(0), ch(1), ch(2)];
}

/** Colour of a project by its size rank (0 is the largest). */
export function projectColor(index: number): string {
  return index >= 0 && index < SLOTS ? SERIES[index]! : OTHER;
}

/** Position along the time ramp, 0 (oldest) to 1 (newest). */
export function timeRgb(t: number): Rgb {
  const x = Math.min(1, Math.max(0, t)) * (TIME_RAMP.length - 1);
  const i = Math.min(TIME_RAMP.length - 2, Math.floor(x));
  const a = hexRgb(TIME_RAMP[i]!);
  const b = hexRgb(TIME_RAMP[i + 1]!);
  const f = x - i;
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

export function timeColor(t: number): string {
  const [r, g, b] = timeRgb(t);
  const h = (v: number) => Math.round(v * 255).toString(16).padStart(2, '0');
  return `#${h(r)}${h(g)}${h(b)}`;
}

/** CSS gradient of the time ramp, for the legend. */
export const TIME_GRADIENT = `linear-gradient(to right, ${TIME_RAMP.join(', ')})`;
