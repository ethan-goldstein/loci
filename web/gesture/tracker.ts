import { FilesetResolver, GestureRecognizer } from '@mediapipe/tasks-vision';
import type { HandObservation, Landmark } from './mapper.ts';

/**
 * Wraps MediaPipe's gesture recogniser (the same model SENTINEL used from Python) for the browser.
 *
 * The WASM runtime and the model are served from this app's own origin, never a CDN, and video
 * frames are processed in the page. Nothing from the camera is stored or sent anywhere.
 */

const WASM_PATH = '/mediapipe/wasm';
const MODEL_PATH = '/models/gesture_recognizer.task';

export interface Tracker {
  /** Runs recognition on the current video frame. Returns null when no hand is visible. */
  process: (video: HTMLVideoElement, nowMs: number) => HandObservation | null;
  close: () => void;
}

export async function createTracker(): Promise<Tracker> {
  const probe = await fetch(MODEL_PATH, { method: 'HEAD' }).catch(() => null);
  if (!probe || !probe.ok) throw new Error('The gesture model is not installed. Run `npm run assets`, rebuild, and try again.');
  const fileset = await FilesetResolver.forVisionTasks(WASM_PATH);
  const options = (delegate: 'GPU' | 'CPU') => ({ baseOptions: { modelAssetPath: MODEL_PATH, delegate }, runningMode: 'VIDEO' as const, numHands: 1 });
  let recognizer: GestureRecognizer;
  try {
    recognizer = await GestureRecognizer.createFromOptions(fileset, options('GPU'));
  } catch {
    recognizer = await GestureRecognizer.createFromOptions(fileset, options('CPU'));
  }
  let lastTs = 0;
  return {
    process(video, nowMs) {
      if (video.readyState < 2 || video.videoWidth === 0) return null;
      // MediaPipe requires strictly increasing timestamps.
      const ts = Math.max(lastTs + 1, Math.round(nowMs));
      lastTs = ts;
      const result = recognizer.recognizeForVideo(video, ts);
      const hand = result.landmarks[0];
      if (!hand || hand.length < 21) return null;
      const top = result.gestures[0]?.[0];
      // Mirror x so the observation is in selfie view: your right hand is on the right of the frame.
      const landmarks: Landmark[] = hand.map((p) => [1 - p.x, p.y, p.z]);
      const name = top && top.categoryName !== 'None' ? top.categoryName : null;
      return { gesture: name, score: top?.score ?? 0, landmarks };
    },
    close() {
      recognizer.close();
    },
  };
}
