import { createHash } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

/**
 * Copies third-party runtime assets into web/public so the viewer serves them from its own origin
 * (no CDN at run time). Neither is committed; see THIRD_PARTY_NOTICES.md for sources and licences.
 *
 *   - MediaPipe Tasks Vision WASM runtime, from the pinned npm package.
 *   - The gesture recogniser model, from LOCI_GESTURE_MODEL or a sibling SENTINEL checkout. It is
 *     only copied when its SHA-256 matches the published file, so a tampered model never ships.
 */

const root = fileURLToPath(new URL('..', import.meta.url));
const MODEL_SHA256 = '97952348cf6a6a4915c2ea1496b4b37ebabc50cbbf80571435643c455f2b0482';

const wasmSrc = path.join(root, 'node_modules/@mediapipe/tasks-vision/wasm');
const wasmDst = path.join(root, 'web/public/mediapipe/wasm');
fs.mkdirSync(wasmDst, { recursive: true });
let copied = 0;
for (const name of fs.readdirSync(wasmSrc)) {
  if (!/^vision_wasm(_nosimd)?_internal\.(js|wasm)$/.test(name)) continue;
  fs.copyFileSync(path.join(wasmSrc, name), path.join(wasmDst, name));
  copied++;
}
console.log(`mediapipe wasm: copied ${copied} files`);

const modelSrc = process.env.LOCI_GESTURE_MODEL ?? path.join(root, '../rev-engr-drone/assets/gesture_recognizer.task');
const modelDst = path.join(root, 'web/public/models/gesture_recognizer.task');
if (!fs.existsSync(modelSrc)) {
  console.log('gesture model: not found, hand control will be unavailable. Set LOCI_GESTURE_MODEL to its path (see THIRD_PARTY_NOTICES.md).');
} else {
  const sha = createHash('sha256').update(fs.readFileSync(modelSrc)).digest('hex');
  if (sha !== MODEL_SHA256) {
    console.error('gesture model: checksum mismatch, not copied.');
    process.exitCode = 1;
  } else {
    fs.mkdirSync(path.dirname(modelDst), { recursive: true });
    fs.copyFileSync(modelSrc, modelDst);
    console.log('gesture model: copied, checksum verified');
  }
}
