/**
 * Turns one frame's hand observation into a camera intent.
 *
 * Ported from SENTINEL (rev-engr-drone/gesture/mapper.py), where the same mapping flew a drone.
 * The stick curve, the palm centre and the gesture vocabulary are unchanged; only the meaning of
 * each axis is new.
 *
 * Gesture vocabulary (selfie view, so "right" is right on screen):
 *   Open_Palm    orbit: palm position is a virtual joystick (left/right = yaw, up/down = pitch)
 *   Thumb_Up     zoom in
 *   Victory      zoom out
 *   Pointing_Up  point: the index fingertip is a cursor over the particles
 *   Closed_Fist  hold: stop everything
 *   anything else, or no hand: idle
 */

export type Landmark = readonly [x: number, y: number, z: number];

export interface HandObservation {
  /** MediaPipe's gesture name, e.g. "Open_Palm". */
  gesture: string | null;
  score: number;
  /** 21 hand landmarks, normalised to 0..1, already mirrored to selfie view. */
  landmarks: readonly Landmark[];
}

export const PALM_POINTS = [0, 5, 9, 13, 17] as const; // wrist + the four knuckles
export const INDEX_TIP = 8;

export type IntentKind = 'none' | 'idle' | 'orbit' | 'zoom' | 'point' | 'hold';

export interface Intent {
  kind: IntentKind;
  /** Stable name used for debouncing, e.g. "orbit", "zoom in". */
  label: string;
  /** Each in -1..1. */
  yaw: number;
  pitch: number;
  zoom: number;
  /** Fingertip position (0..1, selfie view) when pointing. */
  cursor: readonly [number, number] | null;
}

const intent = (kind: IntentKind, label: string, axes: Partial<Pick<Intent, 'yaw' | 'pitch' | 'zoom' | 'cursor'>> = {}): Intent => ({
  kind,
  label,
  yaw: axes.yaw ?? 0,
  pitch: axes.pitch ?? 0,
  zoom: axes.zoom ?? 0,
  cursor: axes.cursor ?? null,
});

export const NO_HAND: Intent = intent('none', 'no hand');

export interface MapperConfig {
  /** Normalised offset from centre that counts as "centred". */
  deadZone: number;
  /** Offset beyond the dead zone where the stick saturates. */
  span: number;
  /** Gesture confidence below this is treated as unrecognised. */
  minScore: number;
}

export const DEFAULT_MAPPER: MapperConfig = { deadZone: 0.12, span: 0.2, minScore: 0.55 };

/** Dead-zoned, linear, saturating stick value for a normalised offset from centre. */
export function stick(offset: number, cfg: MapperConfig): number {
  const mag = Math.abs(offset) - cfg.deadZone;
  if (mag <= 0) return 0;
  return (offset > 0 ? 1 : -1) * Math.min(1, mag / cfg.span);
}

export function palmCenter(landmarks: readonly Landmark[]): [number, number] {
  let x = 0;
  let y = 0;
  for (const i of PALM_POINTS) {
    x += landmarks[i]![0];
    y += landmarks[i]![1];
  }
  return [x / PALM_POINTS.length, y / PALM_POINTS.length];
}

/**
 * `anchor` is the neutral palm position: the screen centre, or where the palm last rested (drag
 * mode, see drag.ts).
 */
export function mapObservation(obs: HandObservation | null, cfg: MapperConfig = DEFAULT_MAPPER, anchor: readonly [number, number] = [0.5, 0.5]): Intent {
  if (obs === null) return NO_HAND;
  const gesture = obs.score >= cfg.minScore ? obs.gesture : null;
  const [cx, cy] = palmCenter(obs.landmarks);
  const dx = cx - anchor[0];
  const dy = anchor[1] - cy; // dy > 0 means the hand is above the anchor

  if (gesture === 'Open_Palm') {
    const yaw = stick(dx, cfg);
    const pitch = stick(dy, cfg);
    return intent(yaw === 0 && pitch === 0 ? 'idle' : 'orbit', 'orbit', { yaw, pitch });
  }
  if (gesture === 'Closed_Fist') return intent('hold', 'hold');
  if (gesture === 'Thumb_Up') return intent('zoom', 'zoom in', { zoom: 1 });
  if (gesture === 'Victory') return intent('zoom', 'zoom out', { zoom: -1 });
  if (gesture === 'Pointing_Up') {
    const tip = obs.landmarks[INDEX_TIP]!;
    return intent('point', 'point', { cursor: [tip[0], tip[1]] });
  }
  return intent('idle', 'idle');
}
