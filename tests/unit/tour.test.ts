import fs from 'node:fs';
import { describe, expect, it } from 'vitest';
import { CARDS, FIGURES, RESULTS_URL, parseHash } from '../../web/tour-logic.ts';
import type { TourFacts } from '../../web/tour-logic.ts';

describe('what the address bar can ask for', () => {
  it('opens the tour, optionally at a step', () => {
    expect(parseHash('#tour', 8)).toEqual({ tour: 0, query: null });
    expect(parseHash('#tour=3', 8)).toEqual({ tour: 3, query: null });
    expect(parseHash('tour=3', 8).tour).toBe(3);
    // Out of range or not a number: clamp or start at the beginning, never throw.
    expect(parseHash('#tour=99', 8).tour).toBe(7);
    expect(parseHash('#tour=abc', 8).tour).toBe(0);
    expect(parseHash('#tour=-2', 8).tour).toBe(0);
  });

  it('runs a search, decoded and bounded', () => {
    expect(parseHash('#q=satellite%20orbit', 8)).toEqual({ tour: null, query: 'satellite orbit' });
    expect(parseHash('#q=satellite+orbit', 8).query).toBe('satellite orbit');
    expect(parseHash(`#q=${'a'.repeat(900)}`, 8).query).toHaveLength(512);
    expect(parseHash('#q=', 8).query).toBeNull();
    expect(parseHash('#q=%E0%A4%A', 8).query).toBeNull(); // a broken escape is not a query
  });

  it('ignores anything else', () => {
    expect(parseHash('', 8)).toEqual({ tour: null, query: null });
    expect(parseHash('#', 8)).toEqual({ tour: null, query: null });
    expect(parseHash('#section-2', 8)).toEqual({ tour: null, query: null });
    expect(parseHash('#x=1&tour=2&q=hello', 8)).toEqual({ tour: 2, query: 'hello' });
  });
});

describe('the cards', () => {
  const facts: TourFacts = {
    n: 6000,
    knn: 8,
    layers: [365, 28, 3],
    demo: true,
    query: 'sourdough starter hydration',
    hops: 2,
    compared: 318,
    both: 5,
    meaning: 7,
    keywords: 7,
    folded: 4,
    repeat: { text: 'Keep the answer short.', sessions: 46 },
  };

  it('say something on every card, from the numbers they are given', () => {
    expect(CARDS).toHaveLength(8);
    const all = CARDS.map((c) => c.body(facts).join(' '));
    for (const text of all) {
      expect(text.length).toBeGreaterThan(60);
      expect(text).not.toMatch(/undefined|NaN/);
      expect(text).not.toContain('—'); // no em dashes
    }
    expect(all[0]).toContain('6,000 messages from an invented history');
    expect(all[2]).toContain('365 of the 6,000');
    expect(all[2]).toContain('3 at the very top');
    expect(all[3]).toContain('"sourdough starter hydration"');
    expect(all[3]).toContain('2 long hops');
    expect(all[4]).toContain('compared 318 messages');
    expect(all[5]).toContain('5 were found by both, 7 by meaning only and 7 by keywords only');
    expect(all[6]).toContain('46 separate sessions');
  });

  it('read naturally at the edges', () => {
    const body = (i: number, f: Partial<TourFacts>) => CARDS[i]!.body({ ...facts, ...f }).join(' ');
    expect(body(0, { demo: false })).toContain('from your history');
    expect(body(3, { hops: 1 })).toContain('1 long hop,');
    expect(body(2, { layers: [40] })).not.toContain('third');
    expect(body(5, { both: 1, folded: 0 })).toContain('1 was found by both');
    expect(body(5, { folded: 0 })).not.toContain('here)');
    expect(body(6, { repeat: null })).toContain('Nothing has been said in enough separate sessions');
    // Before a search has run, the card still reads as a sentence.
    expect(body(4, { compared: undefined })).toContain('compared 0 messages');
  });

  it('quote only results that are in the published results file', () => {
    const results = fs.readFileSync(new URL('../../RESULTS.md', import.meta.url), 'utf8');
    for (const [name, figure] of Object.entries(FIGURES)) expect(results, `${name}: ${figure}`).toContain(figure);
    // The habits figure is the count of repeated things, not a stray 79 elsewhere.
    expect(results).toContain(`${FIGURES.habits} things were said in 3 or more sessions`);
    expect(RESULTS_URL).toMatch(/^https:\/\/github\.com\/ethan-goldstein\/loci\//);
    const last = CARDS[CARDS.length - 1]!.body(facts).join(' ');
    expect(last).toContain('98.1%');
    expect(last).toContain('6.38 times faster');
    expect(last).toContain('0.368 against 0.496');
  });
});
