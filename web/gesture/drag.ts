import { palmCenter } from './mapper.ts';
import type { HandObservation, MapperConfig } from './mapper.ts';

/**
 * Drag mode: the view moves while your open palm moves, and holds still when your palm does.
 *
 * Ported from SENTINEL (rev-engr-drone/gesture/drag.py). Open-palm offsets are measured from an
 * anchor instead of the screen centre. The anchor is set where the palm appears, then jumps to the
 * palm whenever it has rested for `stillS`. So a still hand always means centred sticks, wherever
 * it is, and a slowly sagging arm does not steer.
 */

/** Offsets start at the resting palm, so the dead zone only has to absorb tracking jitter. */
export const DRAG_MAPPER: MapperConfig = { deadZone: 0.04, span: 0.15, minScore: 0.55 };

export interface DragConfig {
  /** Palm must rest this long (seconds) before the view stops. */
  stillS: number;
  /** Palm wander (normalised) that still counts as resting. */
  stillTol: number;
  /** Palm missing this long drops the anchor; the next palm starts fresh. */
  releaseS: number;
}

export const DEFAULT_DRAG: DragConfig = { stillS: 0.3, stillTol: 0.02, releaseS: 0.3 };

export class DragAnchor {
  anchor: [number, number] | null = null;
  private rest: [number, number] | null = null;
  private restSince = 0;
  private palmSeenAt: number | null = null;
  private readonly cfg: DragConfig;

  constructor(cfg: DragConfig = DEFAULT_DRAG) {
    this.cfg = cfg;
  }

  /** Feed every frame with the time in seconds. Returns the anchor to measure offsets from, or null. */
  update(obs: HandObservation | null, now: number): [number, number] | null {
    if (obs === null || obs.gesture !== 'Open_Palm') {
      // Ride out a dropped frame or two; after that, the next palm gets a fresh anchor.
      if (this.palmSeenAt === null || now - this.palmSeenAt > this.cfg.releaseS) {
        this.anchor = null;
        this.rest = null;
      }
      return this.anchor;
    }
    const palm = palmCenter(obs.landmarks);
    this.palmSeenAt = now;
    if (this.anchor === null) this.anchor = palm;
    if (this.rest === null || Math.hypot(palm[0] - this.rest[0], palm[1] - this.rest[1]) > this.cfg.stillTol) {
      this.rest = palm; // palm is moving
      this.restSince = now;
    } else if (now - this.restSince >= this.cfg.stillS) {
      this.anchor = palm; // palm has rested: re-centre so the sticks go to zero
    }
    return this.anchor;
  }
}
