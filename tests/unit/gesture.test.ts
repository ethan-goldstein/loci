import { describe, expect, it } from 'vitest';
import { DEFAULT_DRAG, DRAG_MAPPER, DragAnchor } from '../../web/gesture/drag.ts';
import { DEFAULT_GOVERNOR, Governor } from '../../web/gesture/governor.ts';
import { DEFAULT_MAPPER, mapObservation, stick } from '../../web/gesture/mapper.ts';
import type { HandObservation, Intent, Landmark } from '../../web/gesture/mapper.ts';

/**
 * Ported from SENTINEL's tests (rev-engr-drone/tests/test_mapper.py, test_drag.py and the debounce
 * cases of test_governor.py), with the drone's axes renamed to the camera's.
 */

/** A synthetic hand: all palm points at `palm`, index fingertip at `tip`. */
function hand(gesture: string | null, opts: { palm?: [number, number]; tip?: [number, number]; score?: number } = {}): HandObservation {
  const palm = opts.palm ?? [0.5, 0.5];
  const landmarks: Landmark[] = Array.from({ length: 21 }, () => [palm[0], palm[1], 0]);
  landmarks[8] = [...(opts.tip ?? palm), 0] as unknown as Landmark;
  return { gesture, score: opts.score ?? 0.9, landmarks };
}

const DT = 1 / 30;

describe('mapper', () => {
  it('reports no hand', () => {
    expect(mapObservation(null).kind).toBe('none');
  });

  it('treats a centred open palm as idle', () => {
    const i = mapObservation(hand('Open_Palm'));
    expect(i.kind).toBe('idle');
    expect(i.label).toBe('orbit');
    expect([i.yaw, i.pitch, i.zoom]).toEqual([0, 0, 0]);
  });

  it.each([
    [[0.85, 0.5], 'yaw', 1], // hand right of centre
    [[0.15, 0.5], 'yaw', -1],
    [[0.5, 0.15], 'pitch', 1], // hand above centre
    [[0.5, 0.85], 'pitch', -1],
  ] as const)('open palm at %j drives %s %i', (palm, axis, sign) => {
    const i = mapObservation(hand('Open_Palm', { palm: [...palm] }));
    expect(i.kind).toBe('orbit');
    expect(i[axis]).toBeCloseTo(sign);
    expect(i[axis === 'yaw' ? 'pitch' : 'yaw']).toBe(0);
    expect(i.zoom).toBe(0);
  });

  it('zooms in on thumb up and out on victory', () => {
    expect(mapObservation(hand('Thumb_Up')).zoom).toBe(1);
    expect(mapObservation(hand('Victory')).zoom).toBe(-1);
  });

  it('turns pointing into a cursor at the fingertip', () => {
    const i = mapObservation(hand('Pointing_Up', { tip: [0.8, 0.3] }));
    expect(i.kind).toBe('point');
    expect(i.cursor).toEqual([0.8, 0.3]);
    expect([i.yaw, i.pitch, i.zoom]).toEqual([0, 0, 0]);
  });

  it('holds on a fist', () => {
    expect(mapObservation(hand('Closed_Fist')).kind).toBe('hold');
  });

  it.each([null, 'Thumb_Down', 'ILoveYou'])('ignores gesture %s', (gesture) => {
    const i = mapObservation(hand(gesture, { palm: [0.9, 0.1] }));
    expect(i.kind).toBe('idle');
    expect([i.yaw, i.pitch, i.zoom]).toEqual([0, 0, 0]);
  });

  it('ignores a low-confidence gesture', () => {
    expect(mapObservation(hand('Closed_Fist', { score: 0.3 })).kind).toBe('idle');
  });

  it('has a dead zone and saturates', () => {
    expect(stick(0.05, DEFAULT_MAPPER)).toBe(0);
    expect(stick(DEFAULT_MAPPER.deadZone + DEFAULT_MAPPER.span / 2, DEFAULT_MAPPER)).toBeCloseTo(0.5);
    expect(stick(-0.49, DEFAULT_MAPPER)).toBe(-1);
  });
});

describe('drag anchor', () => {
  function feed(drag: DragAnchor, palm: [number, number] | null, start: number, seconds: number): [Intent, number] {
    let t = start;
    let intent: Intent = mapObservation(null);
    for (let i = 0; i < Math.round(seconds / DT); i++) {
      const obs = palm ? hand('Open_Palm', { palm }) : null;
      const anchor = drag.update(obs, t);
      intent = mapObservation(obs, DRAG_MAPPER, anchor ?? [0.5, 0.5]);
      t += DT;
    }
    return [intent, t];
  }

  it('anchors where the palm first appears', () => {
    const drag = new DragAnchor();
    const [intent] = feed(drag, [0.8, 0.3], 0, 0.1);
    expect(drag.anchor![0]).toBeCloseTo(0.8);
    expect(drag.anchor![1]).toBeCloseTo(0.3);
    expect([intent.yaw, intent.pitch]).toEqual([0, 0]);
  });

  it('steers while the palm moves and stops once it rests', () => {
    const drag = new DragAnchor();
    let [, t] = feed(drag, [0.5, 0.5], 0, 0.5);
    let intent: Intent;
    [intent, t] = feed(drag, [0.65, 0.5], t, DT); // palm jumps right
    expect(intent.kind).toBe('orbit');
    expect(intent.yaw).toBeGreaterThan(0);
    [intent] = feed(drag, [0.65, 0.5], t, DEFAULT_DRAG.stillS + 2 * DT); // ...and rests there
    expect([intent.yaw, intent.pitch]).toEqual([0, 0]);
    expect(drag.anchor![0]).toBeCloseTo(0.65);
  });

  it('pitches up when the palm moves up', () => {
    const drag = new DragAnchor();
    const [, t] = feed(drag, [0.5, 0.5], 0, 0.5);
    const [intent] = feed(drag, [0.5, 0.35], t, DT);
    expect(intent.pitch).toBeGreaterThan(0);
    expect(intent.yaw).toBe(0);
  });

  it('ignores a slowly sagging arm', () => {
    const drag = new DragAnchor();
    let t = 0;
    for (let i = 0; i < 90; i++) {
      const obs = hand('Open_Palm', { palm: [0.5, 0.5 + 0.02 * i * DT] }); // 3 s sinking 0.02 per second
      const intent = mapObservation(obs, DRAG_MAPPER, drag.update(obs, t) ?? [0.5, 0.5]);
      expect([intent.yaw, intent.pitch]).toEqual([0, 0]);
      t += DT;
    }
  });

  it('keeps the anchor through a dropped frame but resets after the hand is lowered', () => {
    const drag = new DragAnchor();
    let [, t] = feed(drag, [0.3, 0.3], 0, 0.2);
    [, t] = feed(drag, null, t, 2 * DT);
    expect(drag.anchor![0]).toBeCloseTo(0.3);
    [, t] = feed(drag, null, t, DEFAULT_DRAG.releaseS + DT);
    expect(drag.anchor).toBeNull();
    feed(drag, [0.7, 0.6], t, DT);
    expect(drag.anchor![0]).toBeCloseTo(0.7);
  });
});

describe('governor', () => {
  const orbit = mapObservation(hand('Open_Palm', { palm: [0.9, 0.5] }));
  const none = mapObservation(null);

  it('needs the same gesture for three frames before acting', () => {
    const g = new Governor();
    expect(g.update(orbit, 0, DT).label).toBe('pending');
    expect(g.update(orbit, DT, DT).label).toBe('pending');
    const third = g.update(orbit, 2 * DT, DT);
    expect(third.label).toBe('orbit');
    expect(third.yaw).toBeGreaterThan(0);
  });

  it('restarts the count when the gesture changes', () => {
    const g = new Governor();
    g.update(orbit, 0, DT);
    g.update(orbit, DT, DT);
    const fist = mapObservation(hand('Closed_Fist'));
    expect(g.update(fist, 2 * DT, DT).label).toBe('pending');
  });

  it('caps output at its authority and ramps up instead of jumping', () => {
    const g = new Governor();
    let t = 0;
    let last = 0;
    for (let i = 0; i < 90; i++) {
      const c = g.update(orbit, t, DT);
      expect(c.yaw).toBeLessThanOrEqual(DEFAULT_GOVERNOR.authority + 1e-9);
      expect(c.yaw - last).toBeLessThanOrEqual(DEFAULT_GOVERNOR.slewPerS * DT + 1e-9);
      last = c.yaw;
      t += DT;
    }
    expect(last).toBeCloseTo(DEFAULT_GOVERNOR.authority);
  });

  it('holds briefly when the hand is lost, then goes to zero', () => {
    const g = new Governor();
    let t = 0;
    for (let i = 0; i < 60; i++, t += DT) g.update(orbit, t, DT);
    const lostAt = t - DT; // the hand was last seen on the previous frame
    expect(g.update(none, t, DT).yaw).toBeGreaterThan(0); // riding out a dropped frame
    for (; t - lostAt <= DEFAULT_GOVERNOR.holdS; t += DT) g.update(none, t, DT);
    const after = g.update(none, t + DT, DT);
    expect([after.yaw, after.pitch, after.zoom]).toEqual([0, 0, 0]);
    expect(after.label).toBe('no hand');
  });

  it('stops immediately on a confirmed fist', () => {
    const g = new Governor();
    let t = 0;
    for (let i = 0; i < 60; i++, t += DT) g.update(orbit, t, DT);
    const fist = mapObservation(hand('Closed_Fist'));
    let c = g.update(fist, t, DT);
    for (let i = 0; i < 3; i++) c = g.update(fist, (t += DT), DT);
    expect(c.hold).toBe(true);
    expect([c.yaw, c.pitch, c.zoom]).toEqual([0, 0, 0]);
  });

  it('selects only after the cursor dwells on one particle, and only once', () => {
    const g = new Governor();
    expect(g.dwell(7, 0).fire).toBe(-1);
    expect(g.dwell(7, 0.5).fire).toBe(-1);
    expect(g.dwell(9, 0.6).fire).toBe(-1); // moved to another particle: the clock restarts
    expect(g.dwell(9, 0.6 + DEFAULT_GOVERNOR.selectHoldS - 0.05).fire).toBe(-1);
    expect(g.dwell(9, 0.6 + DEFAULT_GOVERNOR.selectHoldS).fire).toBe(9);
    expect(g.dwell(9, 5).fire).toBe(-1); // does not fire again while still pointing
    expect(g.dwell(-1, 5.1).progress).toBe(0);
  });
});
