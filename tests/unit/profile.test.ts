import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { Engine } from '../../src/engine.ts';
import { profileSummary, runProfile, weekStart } from '../../src/insight/profile.ts';
import type { ProfileInput, ProfileReport } from '../../src/insight/profile.ts';
import { runRepeats } from '../../src/insight/repeats.ts';
import { buildDemoStore, hashEmbedder } from '../../src/prove/demo.ts';
import { renderResults } from '../../src/prove/report.ts';

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'loci-profile-test-'));
afterAll(() => fs.rmSync(tmp, { recursive: true, force: true }));

const MIN = 60_000;
/** Monday 2026-03-02, 09:00 UTC. */
const MON9 = Date.UTC(2026, 2, 2, 9, 0, 0);

/** [session, project, role (0 typed, 1 assistant), minutes after MON9] */
type Row = [number, number, 0 | 1, number];
function input(rows: Row[]): ProfileInput {
  return {
    n: rows.length,
    sessionPk: rows.map((r) => r[0]),
    project: rows.map((r) => r[1]),
    role: rows.map((r) => r[2]),
    ts: rows.map((r) => MON9 + r[3] * MIN),
  };
}

describe('session profile on a hand-built history', () => {
  const rows: Row[] = [
    // Session 1, project 7: prompts at 0 and 10, replies between, then a 50-minute gap, then more.
    [1, 7, 0, 0],
    [1, 7, 1, 2],
    [1, 7, 0, 10],
    [1, 7, 0, 10], // the same prompt stored as a second chunk: one prompt, not two
    [1, 7, 1, 12],
    [1, 7, 0, 62],
    [1, 7, 1, 65],
    // Session 2, project 7, the next day (Tuesday) at 14:00: one prompt, one reply.
    [2, 7, 0, 24 * 60 + 5 * 60],
    [2, 7, 1, 24 * 60 + 5 * 60 + 4],
    // Session 3, project 9, eight days later (the following Tuesday) at 09:30: replies only.
    [3, 9, 1, 8 * 24 * 60 + 30],
    [3, 9, 1, 8 * 24 * 60 + 31],
  ];
  const p = runProfile(input(rows), { timeZone: 'UTC', idleMinutes: 30 });

  it('counts sessions, prompts, replies and days', () => {
    // Six reply chunks: three in the first session, one in the second, two in the third.
    expect(p.totals).toMatchObject({ sessions: 3, prompts: 4, replies: 6, activeDays: 3, weeks: 2, projects: 2 });
    expect(p.corpus).toBe(rows.length);
  });

  it('measures active time as the gaps between messages, leaving out the long ones', () => {
    // Session 1: 2 + 8 + 0 + 2 minutes, then a 50-minute gap that does not count, then 3. Session 2: 4. Session 3: 1.
    expect(p.perSession.activeMinutes.max).toBe(15);
    expect(p.totals.activeHours).toBeCloseTo((15 + 4 + 1) / 60, 1);
    // A longer cut-off lets the 50-minute gap in; a shorter one changes nothing here.
    expect(p.sensitivity).toEqual([
      { idleMinutes: 15, activeHours: 0.3 },
      { idleMinutes: 30, activeHours: 0.3 },
      { idleMinutes: 60, activeHours: 1.2 },
    ]);
  });

  it('places prompts on the weekday and hour they were typed', () => {
    expect(p.rhythm).toHaveLength(7);
    for (const row of p.rhythm) expect(row).toHaveLength(24);
    expect(p.rhythm[0]![9]).toBe(2); // Monday 09:00 and 09:10
    expect(p.rhythm[0]![10]).toBe(1); // Monday 10:02
    expect(p.rhythm[1]![14]).toBe(1); // Tuesday 14:00
    expect(p.rhythm.flat().reduce((a, b) => a + b, 0)).toBe(p.totals.prompts);
    // The same moments in another time zone land on other hours, but there are just as many.
    const ny = runProfile(input(rows), { timeZone: 'America/New_York' });
    expect(ny.rhythm.flat().reduce((a, b) => a + b, 0)).toBe(4);
    expect(ny.rhythm[0]![9]).toBe(0);
    expect(ny.rhythm[0]![4]).toBe(2); // 09:00 UTC is 04:00 in New York in March before the clocks change
  });

  it('groups sessions by how many prompts they had', () => {
    expect(p.promptsHistogram.find((b) => b.label === '2 to 3')!.sessions).toBe(1);
    expect(p.promptsHistogram.find((b) => b.label === '1')!.sessions).toBe(1);
    // A session with no typed prompt is counted as a session but has no bucket.
    expect(p.promptsHistogram.reduce((a, b) => a + b.sessions, 0)).toBe(2);
    expect(p.perSession.prompts.max).toBe(3);
  });

  it('adds up week by week', () => {
    expect(p.weekly.map((w) => w.t)).toEqual([weekStart(MON9), weekStart(MON9) + 7 * 24 * 60 * MIN]);
    expect(p.weekly.map((w) => w.sessions)).toEqual([2, 1]);
    expect(p.weekly.map((w) => w.prompts)).toEqual([4, 0]);
    expect(p.weekly.map((w) => w.projects)).toEqual([1, 1]);
    expect(p.weekly.reduce((a, w) => a + w.sessions, 0)).toBe(p.totals.sessions);
    expect(new Date(weekStart(MON9)).getUTCDay()).toBe(1); // weeks start on Monday
  });

  it('reports how concentrated the work is without naming a project', () => {
    expect(p.concentration.topShare).toBeCloseTo(9 / 11, 10);
    expect(p.concentration.top3Share).toBeCloseTo(1, 10);
    expect(JSON.stringify(p)).not.toMatch(/"name"|"title"|"text"/);
  });

  it('carries the weekly "said before" series only when a repeats report is given', () => {
    expect(p.saidBefore).toBeNull();
    const withRepeats = runProfile(input(rows), { timeZone: 'UTC', repeats: { weekly: [{ t: weekStart(MON9), sentences: 10, saidBefore: 3 }] } });
    expect(withRepeats.saidBefore).toEqual([{ t: weekStart(MON9), sentences: 10, saidBefore: 3 }]);
    expect(profileSummary(withRepeats).join('\n')).toContain('30.0% of typed sentences repeat');
  });

  it('does not count program-written prompts as typed', () => {
    // Chunk 0 is the first prompt of session 1. Marked as not typed, it is no longer a prompt.
    const q = runProfile(input(rows), { timeZone: 'UTC', notTyped: new Set([0]) });
    expect(q.totals.prompts).toBe(3);
    expect(q.totals.replies).toBe(6);
    expect(q.rhythm[0]![9]).toBe(1);
  });

  it('copes with nothing at all', () => {
    const empty = runProfile(input([]), { timeZone: 'UTC' });
    expect(empty.totals.sessions).toBe(0);
    expect(empty.perSession.prompts).toEqual({ median: 0, p90: 0, max: 0 });
    expect(empty.weekly).toEqual([]);
  });
});

describe('session profile on the readable demo store', () => {
  let engine: Engine;
  let p: ProfileReport;
  beforeAll(async () => {
    const cfg = await buildDemoStore(path.join(tmp, 'demo'), 3000, 7, { prose: true });
    engine = Engine.open(cfg, { embedder: hashEmbedder(), save: false, kernel: 'js' });
    const repeats = await runRepeats(engine, { embedder: hashEmbedder() });
    p = runProfile(engine, { timeZone: 'UTC', repeats });
  });
  afterAll(() => engine.close());

  it('finds the rhythm the invented history was given', () => {
    // Sessions are sittings of minutes, not days: the demo's messages are minutes apart.
    expect(p.perSession.activeMinutes.median).toBeGreaterThan(10);
    expect(p.perSession.activeMinutes.median).toBeLessThan(120);
    expect(p.totals.sessions).toBeGreaterThan(150);
    // Nobody in the demo works in the small hours, and weekends are quieter than weekdays.
    for (const row of p.rhythm) for (let h = 2; h <= 7; h++) expect(row[h]).toBe(0);
    const perDay = p.rhythm.map((row) => row.reduce((a, b) => a + b, 0));
    const weekday = perDay.slice(0, 5).reduce((a, b) => a + b, 0) / 5;
    const weekend = (perDay[5]! + perDay[6]!) / 2;
    expect(weekend).toBeLessThan(weekday * 0.6);
    expect(perDay.reduce((a, b) => a + b, 0)).toBe(p.totals.prompts);
  });

  it('has a weekly "said before" series that the planted habits show up in', () => {
    expect(p.saidBefore).not.toBeNull();
    const sentences = p.saidBefore!.reduce((a, w) => a + w.sentences, 0);
    const again = p.saidBefore!.reduce((a, w) => a + w.saidBefore, 0);
    expect(sentences).toBeGreaterThan(400);
    // About one prompt in seven carries a planted instruction, so a clear share has been said before.
    expect(again / sentences).toBeGreaterThan(0.05);
    expect(again / sentences).toBeLessThan(0.6);
    for (const w of p.saidBefore!) expect(w.saidBefore).toBeLessThanOrEqual(w.sentences);
  });

  it('writes a results section with numbers and no project name', () => {
    const md = renderResults(null, null, '2026-01-01', null, null, p);
    expect(md).toContain('## 5. Where do the sessions go?');
    expect(md).toContain('| prompts in a session | sessions |');
    expect(md).toContain('deliberately left out of this file');
    for (const project of engine.projects) expect(md).not.toContain(project.name);
    // The hour-by-weekday pattern is personal and must not be published.
    expect(md).not.toMatch(/Mon|Tue|Wed|Thu|Fri|Sat|Sun/);
    expect(md).not.toMatch(/\d\d:00/);
  });

  it('summarises in plain lines for the terminal', () => {
    const lines = profileSummary(p);
    expect(lines[0]).toMatch(/^\d+ sessions on \d+ days over \d+ weeks, 6 projects\.$/);
    expect(lines.join('\n')).toContain('Busiest hour of the week');
  });
});
