import type { OrbitRig } from '../OrbitRig.ts';
import { DRAG_MAPPER, DragAnchor } from './drag.ts';
import { Governor } from './governor.ts';
import { mapObservation } from './mapper.ts';
import type { HandObservation } from './mapper.ts';
import { createTracker } from './tracker.ts';

/**
 * Connects the webcam to the camera rig: tracker -> drag anchor -> mapper -> governor -> rig.
 * The same pipeline, in the same order, that SENTINEL used to fly a drone.
 */

export interface GestureHost {
  rig: OrbitRig;
  /** Nearest particle to a normalised screen point (0..1), or -1. */
  pickAt: (nx: number, ny: number) => number;
  onPoint: (particle: number) => void;
  onSelect: (particle: number) => void;
}

/** Full-deflection rates: radians per second for yaw and pitch, e-folds per second for zoom. */
const RATE = { yaw: 1.5, pitch: 1.0, zoom: 1.1 };
const FRAME_MS = 1000 / 24;

const CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];

function drawHand(ctx: CanvasRenderingContext2D, obs: HandObservation | null, dwell: number): void {
  const { width: w, height: h } = ctx.canvas;
  ctx.clearRect(0, 0, w, h);
  if (!obs) return;
  // The overlay canvas is mirrored by CSS along with the video, so un-mirror x here.
  const px = (i: number): [number, number] => [(1 - obs.landmarks[i]![0]) * w, obs.landmarks[i]![1] * h];
  ctx.strokeStyle = 'rgba(255, 217, 160, 0.85)';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (const [a, b] of CONNECTIONS) {
    const [ax, ay] = px(a);
    const [bx, by] = px(b);
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
  }
  ctx.stroke();
  if (dwell > 0) {
    const [tx, ty] = px(8);
    ctx.beginPath();
    ctx.arc(tx, ty, 9, -Math.PI / 2, -Math.PI / 2 + dwell * Math.PI * 2);
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2.5;
    ctx.stroke();
  }
}

export async function startGestures(host: GestureHost): Promise<{ stop: () => void }> {
  const view = document.getElementById('gesture-view') as HTMLElement;
  const video = document.getElementById('gesture-video') as HTMLVideoElement;
  const overlay = document.getElementById('gesture-overlay') as HTMLCanvasElement;
  const label = document.getElementById('gesture-label') as HTMLElement;
  const ctx = overlay.getContext('2d');
  if (!ctx) throw new Error('Could not draw the hand overlay');
  if (!navigator.mediaDevices?.getUserMedia) throw new Error('This browser cannot open the camera');

  const tracker = await createTracker();
  let stream: MediaStream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' }, audio: false });
  } catch {
    tracker.close();
    throw new Error('Camera access was not granted');
  }
  video.srcObject = stream;
  await video.play();
  view.hidden = false;

  const drag = new DragAnchor();
  const governor = new Governor();
  const wasAutoRotating = host.rig.autoRotate;
  let running = true;
  let raf = 0;
  let lastRun = 0;
  let lastT = performance.now() / 1000;

  const loop = (nowMs: number) => {
    if (!running) return;
    raf = requestAnimationFrame(loop);
    if (nowMs - lastRun < FRAME_MS) return; // recognition is the expensive part; 24 fps is plenty
    lastRun = nowMs;
    const now = nowMs / 1000;
    const dt = Math.min(0.2, now - lastT);
    lastT = now;

    const obs = tracker.process(video, nowMs);
    const anchor = drag.update(obs, now);
    const cmd = governor.update(mapObservation(obs, DRAG_MAPPER, anchor ?? [0.5, 0.5]), now, dt);

    // Moving the hand right turns the view right; raising it tilts the view up.
    host.rig.rates = { yaw: -cmd.yaw * RATE.yaw, pitch: -cmd.pitch * RATE.pitch, zoom: cmd.zoom * RATE.zoom };
    host.rig.autoRotate = wasAutoRotating && !cmd.hold && cmd.label === 'no hand';

    let dwell = 0;
    if (cmd.cursor) {
      const particle = host.pickAt(cmd.cursor[0], cmd.cursor[1]);
      host.onPoint(particle);
      const d = governor.dwell(particle, now);
      dwell = d.progress;
      if (d.fire >= 0) host.onSelect(d.fire);
    } else {
      governor.dwell(-1, now);
    }
    label.textContent = cmd.label;
    drawHand(ctx, obs, dwell);
  };
  raf = requestAnimationFrame(loop);

  return {
    stop() {
      running = false;
      cancelAnimationFrame(raf);
      for (const track of stream.getTracks()) track.stop();
      video.srcObject = null;
      tracker.close();
      view.hidden = true;
      host.rig.rates = { yaw: 0, pitch: 0, zoom: 0 };
      host.rig.autoRotate = wasAutoRotating;
    },
  };
}
