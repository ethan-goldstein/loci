import type { Intent } from './mapper.ts';

/**
 * Sits between gestures and the camera, the way SENTINEL's flight governor sat between gestures
 * and the motors (rev-engr-drone/flight/governor.py). The stakes are lower here, but the same
 * rules make hand control feel deliberate instead of twitchy:
 *
 *   - a gesture must be seen for `confirmFrames` frames in a row before it acts (debounce)
 *   - output is capped at `authority` of full deflection
 *   - output can only grow at `slewPerS` per second, so nothing lurches
 *   - when the hand disappears, the last command is held for `holdS` to ride out dropped frames,
 *     then everything goes to zero
 *   - selecting a particle needs the fingertip to dwell on it for `selectHoldS`
 */

export interface GovernorConfig {
  /** Max stick deflection passed on (0..1). */
  authority: number;
  confirmFrames: number;
  holdS: number;
  /** Max increase in stick magnitude per second. */
  slewPerS: number;
  selectHoldS: number;
}

export const DEFAULT_GOVERNOR: GovernorConfig = { authority: 0.8, confirmFrames: 3, holdS: 0.4, slewPerS: 2.5, selectHoldS: 0.9 };

export interface Command {
  yaw: number;
  pitch: number;
  zoom: number;
  /** Confirmed gesture label, or "pending" while debouncing. */
  label: string;
  /** True on fist: the caller should also pause any idle motion. */
  hold: boolean;
  /** Cursor to hover at, when pointing is confirmed. */
  cursor: readonly [number, number] | null;
}

const ZERO = { yaw: 0, pitch: 0, zoom: 0 };

export class Governor {
  private candidate: string | null = null;
  private candidateCount = 0;
  private out = { ...ZERO };
  private lastTarget = { ...ZERO };
  private lastHandAt: number | null = null;
  private dwellTarget = -1;
  private dwellSince = 0;
  private dwellFired = false;
  private readonly cfg: GovernorConfig;

  constructor(cfg: GovernorConfig = DEFAULT_GOVERNOR) {
    this.cfg = cfg;
  }

  private debounce(label: string): boolean {
    if (label === this.candidate) this.candidateCount++;
    else {
      this.candidate = label;
      this.candidateCount = 1;
    }
    return this.candidateCount >= this.cfg.confirmFrames;
  }

  /** Limits how fast each axis may grow in magnitude; shrinking is always immediate. */
  private slew(target: typeof ZERO, dt: number): typeof ZERO {
    const max = this.cfg.slewPerS * dt;
    const step = (cur: number, want: number) => {
      if (Math.abs(want) <= Math.abs(cur) && Math.sign(want) === Math.sign(cur)) return want; // easing off
      if (want === 0) return 0;
      return cur + Math.max(-max, Math.min(max, want - cur));
    };
    return { yaw: step(this.out.yaw, target.yaw), pitch: step(this.out.pitch, target.pitch), zoom: step(this.out.zoom, target.zoom) };
  }

  /** Feed every frame. `now` and `dt` are in seconds. */
  update(intent: Intent, now: number, dt: number): Command {
    let target = { ...ZERO };
    let label = 'pending';
    let hold = false;
    let cursor: Command['cursor'] = null;

    if (intent.kind === 'none') {
      // Hand lost: keep the last command briefly, then stop.
      const lostFor = this.lastHandAt === null ? Infinity : now - this.lastHandAt;
      if (lostFor <= this.cfg.holdS) {
        target = { ...this.lastTarget };
        label = 'holding';
      } else {
        this.candidate = null;
        this.candidateCount = 0;
        label = 'no hand';
      }
    } else {
      this.lastHandAt = now;
      if (this.debounce(intent.label)) {
        label = intent.label;
        const a = this.cfg.authority;
        const clamp = (v: number) => Math.max(-1, Math.min(1, v)) * a;
        target = { yaw: clamp(intent.yaw), pitch: clamp(intent.pitch), zoom: clamp(intent.zoom) };
        hold = intent.kind === 'hold';
        cursor = intent.kind === 'point' ? intent.cursor : null;
      }
      this.lastTarget = target;
    }

    this.out = hold ? { ...ZERO } : this.slew(target, dt);
    return { ...this.out, label, hold, cursor };
  }

  /**
   * Dwell-to-select: returns the particle once the cursor has stayed on it for `selectHoldS`, and
   * only once per dwell. Pass -1 when nothing is under the cursor.
   */
  dwell(particle: number, now: number): { progress: number; fire: number } {
    if (particle < 0) {
      this.dwellTarget = -1;
      this.dwellFired = false;
      return { progress: 0, fire: -1 };
    }
    if (particle !== this.dwellTarget) {
      this.dwellTarget = particle;
      this.dwellSince = now;
      this.dwellFired = false;
    }
    const progress = Math.min(1, (now - this.dwellSince) / this.cfg.selectHoldS);
    if (progress >= 1 && !this.dwellFired) {
      this.dwellFired = true;
      return { progress, fire: particle };
    }
    return { progress, fire: -1 };
  }
}
