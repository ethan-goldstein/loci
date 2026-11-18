import type { RepeatsReport } from './repeats.ts';

/**
 * Session profile: where does the time go?
 *
 * A code profiler does not read your program's data, it counts where the program spends its time.
 * This does the same for work with an assistant. It reads only timestamps, who was speaking, and
 * which session and project a chunk belongs to. No text is read and none appears in the output,
 * and projects are never named: the report holds counts, durations and shares.
 *
 * Pure TypeScript with no Node dependencies, so the public demo can show the same panel from
 * invented data.
 */

/** What the profile reads. The engine satisfies this; so does the demo's static data. */
export interface ProfileInput {
  n: number;
  /** Milliseconds since the epoch, per chunk. */
  ts: ArrayLike<number>;
  /** 0 for something the person typed, 1 for the assistant. */
  role: ArrayLike<number>;
  project: ArrayLike<number>;
  sessionPk: ArrayLike<number>;
}

export interface ProfileOptions {
  /** IANA time zone for the weekday-and-hour rhythm, e.g. "America/New_York". Use "UTC" for reproducible output. */
  timeZone: string;
  /** A gap longer than this inside a session is time away, not time working. */
  idleMinutes?: number;
  /** The repeat-finder's report, for the weekly "said before" series. */
  repeats?: Pick<RepeatsReport, 'weekly'> | null;
  /**
   * Chunks that look typed but were written by a program (a repeated template). They are not
   * counted as prompts, the same rule the repeat-finder applies.
   */
  notTyped?: ReadonlySet<number>;
}

export interface Quantiles {
  median: number;
  p90: number;
  max: number;
}

export interface ProfileWeek {
  /** Start of the week (Monday 00:00 UTC), milliseconds. */
  t: number;
  sessions: number;
  prompts: number;
  activeMinutes: number;
  /** Projects touched that week. */
  projects: number;
  /** Share of that week's chunks in its busiest project. */
  topProjectShare: number;
}

export interface ProfileReport {
  generated: string;
  /** Chunks in the index when this was computed. */
  corpus: number;
  timeZone: string;
  idleMinutes: number;
  /** `prompts` are typed prompts: one stored as several chunks counts once, and program-written ones are left out. `replies` are reply chunks. */
  totals: { sessions: number; prompts: number; replies: number; activeDays: number; activeHours: number; weeks: number; projects: number };
  perSession: { prompts: Quantiles; activeMinutes: Quantiles; chunks: Quantiles };
  /** How many sessions had this many prompts. */
  promptsHistogram: { label: string; sessions: number }[];
  /** Prompts typed, by weekday (Monday first) and hour of day, in `timeZone`. Seven rows of 24. */
  rhythm: number[][];
  weekly: ProfileWeek[];
  /** Share of all chunks in the busiest project, and in the busiest three. Projects are not named. */
  concentration: { topShare: number; top3Share: number };
  /** Per week: typed sentences, and how many repeat something said in an earlier session. Null without a repeats report. */
  saidBefore: { t: number; sentences: number; saidBefore: number }[] | null;
  /** Total active hours under other idle cut-offs, to show how much the choice matters. */
  sensitivity: { idleMinutes: number; activeHours: number }[];
}

const MINUTE = 60_000;
const DAY = 86_400_000;
const WEEK = 7 * DAY;
/** 1970-01-05 was a Monday, so weeks counted from it start on Mondays. */
const FIRST_MONDAY = 4 * DAY;

export const weekStart = (ts: number): number => Math.floor((ts - FIRST_MONDAY) / WEEK) * WEEK + FIRST_MONDAY;

function quantiles(values: number[]): Quantiles {
  if (values.length === 0) return { median: 0, p90: 0, max: 0 };
  const sorted = [...values].sort((a, b) => a - b);
  const at = (p: number) => sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))]!;
  return { median: at(0.5), p90: at(0.9), max: sorted[sorted.length - 1]! };
}

/** Minutes of work in a list of sorted timestamps: the gaps between them, leaving out the long ones. */
function activeMinutes(sortedTs: readonly number[], idleMs: number): number {
  let ms = 0;
  for (let i = 1; i < sortedTs.length; i++) {
    const gap = sortedTs[i]! - sortedTs[i - 1]!;
    if (gap <= idleMs) ms += gap;
  }
  return ms / MINUTE;
}

const HISTOGRAM: [label: string, upTo: number][] = [
  ['1', 1],
  ['2 to 3', 3],
  ['4 to 7', 7],
  ['8 to 15', 15],
  ['16 to 31', 31],
  ['32 or more', Infinity],
];

const WEEKDAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];

/** Weekday (0 = Monday) and hour (0 to 23) of a moment in a time zone. */
function clock(timeZone: string): (ts: number) => [weekday: number, hour: number] {
  const fmt = new Intl.DateTimeFormat('en-US', { timeZone, weekday: 'short', hour: 'numeric', hourCycle: 'h23' });
  return (ts) => {
    let weekday = 0;
    let hour = 0;
    for (const part of fmt.formatToParts(new Date(ts))) {
      if (part.type === 'weekday') weekday = Math.max(0, WEEKDAYS.indexOf(part.value));
      else if (part.type === 'hour') hour = Number(part.value) % 24;
    }
    return [weekday, hour];
  };
}

export function runProfile(input: ProfileInput, opts: ProfileOptions): ProfileReport {
  const idleMinutes = opts.idleMinutes ?? 30;
  const idleMs = idleMinutes * MINUTE;
  const n = input.n;

  // Session spans come from the chunks themselves, not from any stored first and last time.
  const bySession = new Map<number, { ts: number[]; prompts: Set<number>; project: number }>();
  const projectChunks = new Map<number, number>();
  let replies = 0;
  for (let i = 0; i < n; i++) {
    const pk = input.sessionPk[i]!;
    let s = bySession.get(pk);
    if (!s) {
      s = { ts: [], prompts: new Set(), project: input.project[i]! };
      bySession.set(pk, s);
    }
    const t = input.ts[i]!;
    s.ts.push(t);
    // A long prompt is stored as several chunks with one timestamp: count it once.
    if (input.role[i] !== 0) replies++;
    else if (!opts.notTyped?.has(i)) s.prompts.add(t);
    projectChunks.set(input.project[i]!, (projectChunks.get(input.project[i]!) ?? 0) + 1);
  }

  const sessionPrompts: number[] = [];
  const sessionMinutes: number[] = [];
  const sessionChunks: number[] = [];
  const histogram = HISTOGRAM.map(([label]) => ({ label, sessions: 0 }));
  const rhythm = Array.from({ length: 7 }, () => new Array<number>(24).fill(0));
  const at = clock(opts.timeZone);
  const days = new Set<number>();
  const weeks = new Map<number, { sessions: number; prompts: number; activeMinutes: number; chunks: Map<number, number> }>();
  const week = (t: number) => {
    const key = weekStart(t);
    let w = weeks.get(key);
    if (!w) {
      w = { sessions: 0, prompts: 0, activeMinutes: 0, chunks: new Map() };
      weeks.set(key, w);
    }
    return w;
  };
  const sensitivityCuts = [15, idleMinutes, 60].filter((v, i, a) => a.indexOf(v) === i).sort((a, b) => a - b);
  const sensitivity = new Map(sensitivityCuts.map((c) => [c, 0]));

  for (const s of bySession.values()) {
    s.ts.sort((a, b) => a - b);
    const minutes = activeMinutes(s.ts, idleMs);
    const prompts = s.prompts.size;
    sessionPrompts.push(prompts);
    sessionMinutes.push(minutes);
    sessionChunks.push(s.ts.length);
    if (prompts > 0) histogram[HISTOGRAM.findIndex(([, upTo]) => prompts <= upTo)]!.sessions++;
    for (const c of sensitivityCuts) sensitivity.set(c, sensitivity.get(c)! + activeMinutes(s.ts, c * MINUTE));

    // A session belongs to the week it started in; its minutes are counted there.
    const w = week(s.ts[0]!);
    w.sessions++;
    w.activeMinutes += minutes;
    for (const t of s.prompts) {
      const [weekday, hour] = at(t);
      rhythm[weekday]![hour]!++;
      week(t).prompts++;
    }
    for (const t of s.ts) {
      days.add(Math.floor(t / DAY));
      const wk = week(t);
      wk.chunks.set(s.project, (wk.chunks.get(s.project) ?? 0) + 1);
    }
  }

  const weekly: ProfileWeek[] = [...weeks.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([t, w]) => {
      const counts = [...w.chunks.values()];
      const total = counts.reduce((a, b) => a + b, 0);
      return { t, sessions: w.sessions, prompts: w.prompts, activeMinutes: Math.round(w.activeMinutes), projects: counts.length, topProjectShare: total ? Math.max(...counts) / total : 0 };
    });

  const shares = [...projectChunks.values()].sort((a, b) => b - a).map((c) => c / Math.max(1, n));
  const totalPrompts = sessionPrompts.reduce((a, b) => a + b, 0);
  const round1 = (x: number) => Math.round(x * 10) / 10;
  return {
    generated: new Date().toISOString(),
    corpus: n,
    timeZone: opts.timeZone,
    idleMinutes,
    totals: {
      sessions: bySession.size,
      prompts: totalPrompts,
      replies,
      activeDays: days.size,
      activeHours: round1(sessionMinutes.reduce((a, b) => a + b, 0) / 60),
      weeks: weekly.length,
      projects: projectChunks.size,
    },
    perSession: {
      prompts: quantiles(sessionPrompts),
      activeMinutes: (({ median, p90, max }) => ({ median: Math.round(median), p90: Math.round(p90), max: Math.round(max) }))(quantiles(sessionMinutes)),
      chunks: quantiles(sessionChunks),
    },
    promptsHistogram: histogram,
    rhythm,
    weekly,
    concentration: { topShare: shares[0] ?? 0, top3Share: shares.slice(0, 3).reduce((a, b) => a + b, 0) },
    saidBefore: opts.repeats?.weekly ? opts.repeats.weekly.map((w) => ({ t: w.t, sentences: w.sentences, saidBefore: w.saidBefore })) : null,
    sensitivity: sensitivityCuts.map((c) => ({ idleMinutes: c, activeHours: round1(sensitivity.get(c)! / 60) })),
  };
}

/** A few plain lines for the terminal. Counts only. */
export function profileSummary(p: ProfileReport): string[] {
  const t = p.totals;
  const lines = [
    `${t.sessions} sessions on ${t.activeDays} days over ${t.weeks} weeks, ${t.projects} projects.`,
    `${t.prompts.toLocaleString('en-US')} prompts typed, ${t.replies.toLocaleString('en-US')} reply chunks.`,
    `About ${t.activeHours} hours of active time (gaps over ${p.idleMinutes} minutes left out).`,
    `A typical session: ${p.perSession.prompts.median} prompts and ${p.perSession.activeMinutes.median} minutes. One in ten runs past ${p.perSession.prompts.p90} prompts or ${p.perSession.activeMinutes.p90} minutes.`,
    `The busiest project holds ${(p.concentration.topShare * 100).toFixed(0)}% of everything; the busiest three hold ${(p.concentration.top3Share * 100).toFixed(0)}%.`,
  ];
  let peak = 0;
  let peakDay = 0;
  let peakHour = 0;
  p.rhythm.forEach((row, d) =>
    row.forEach((v, h) => {
      if (v > peak) {
        peak = v;
        peakDay = d;
        peakHour = h;
      }
    }),
  );
  if (peak > 0) lines.push(`Busiest hour of the week: ${WEEKDAYS[peakDay]} ${String(peakHour).padStart(2, '0')}:00 (${p.timeZone}).`);
  if (p.saidBefore) {
    const sentences = p.saidBefore.reduce((a, w) => a + w.sentences, 0);
    const again = p.saidBefore.reduce((a, w) => a + w.saidBefore, 0);
    if (sentences > 0) lines.push(`${((again / sentences) * 100).toFixed(1)}% of typed sentences repeat something said in an earlier session.`);
  }
  return lines;
}
