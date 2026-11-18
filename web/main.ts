import * as THREE from 'three';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { Activity } from './Activity.ts';
import '@fontsource-variable/space-grotesk/wght.css';
import '@fontsource-variable/jetbrains-mono/wght.css';
import { STATIC, chunk, loadCorpus, profile, repeats, search } from './api.ts';
import type { Corpus, Mode, ProfileView, RepeatsView } from './api.ts';
import { Cloud } from './Cloud.ts';
import { Hud } from './hud.ts';
import { setupIntro } from './intro.ts';
import type { ToggleName } from './hud.ts';
import { Labels } from './Labels.ts';
import { OrbitRig } from './OrbitRig.ts';
import { Trace } from './Trace.ts';

const canvas = document.getElementById('stage') as HTMLCanvasElement;
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

/** How strongly points away from the focal distance blur, in pixels at twice the focal distance. */
const APERTURE = 9;

function fail(message: string): void {
  const box = document.getElementById('fatal');
  if (!box) return;
  box.hidden = false;
  box.textContent = message;
}

/** A soft vignette drawn inside the scene, so the glow pass and the background share one image. */
function backdrop(): THREE.Mesh {
  const material = new THREE.ShaderMaterial({
    vertexShader: /* glsl */ `varying vec2 vUv; void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`,
    fragmentShader: /* glsl */ `
      precision highp float; varying vec2 vUv;
      void main() {
        float d = distance(vUv, vec2(0.5, 0.55));
        // Chosen so that, after the tone curve, the centre lands near #0b1220 and the edge near #030407.
        vec3 inner = vec3(0.015, 0.0255, 0.0476);
        vec3 outer = vec3(0.004, 0.0054, 0.0092);
        gl_FragColor = vec4(mix(inner, outer, smoothstep(0.0, 0.75, d)), 1.0);
      }`,
    depthTest: false,
    depthWrite: false,
  });
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material);
  mesh.frustumCulled = false;
  mesh.renderOrder = -10;
  return mesh;
}

async function start(): Promise<void> {
  let renderer: THREE.WebGLRenderer;
  try {
    renderer = new THREE.WebGLRenderer({ canvas, antialias: false, alpha: false, powerPreference: 'high-performance' });
  } catch {
    fail('This viewer needs WebGL 2, which this browser or device does not provide.');
    return;
  }
  renderer.setClearColor(0x030407, 1);
  // The shaders already write display-ready colours, so nothing should convert them again on output.
  renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
  // Capped at 2 so high-density phones do not render four times the pixels for no visible gain.
  const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
  renderer.setPixelRatio(pixelRatio);

  let corpus: Corpus;
  try {
    corpus = await loadCorpus();
  } catch (err) {
    fail(`Could not load the corpus from the local server. ${err instanceof Error ? err.message : ''}`);
    return;
  }
  const { manifest, meta } = corpus;
  if (manifest.n === 0) {
    fail('The store is empty. Run `loci ingest` and `loci embed` first.');
    return;
  }

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 200);
  const rig = new OrbitRig(camera, canvas);
  const cloud = new Cloud(renderer, corpus);
  const trace = new Trace(cloud, pixelRatio);
  scene.add(backdrop(), cloud.group, trace.group);
  if (reducedMotion) cloud.alpha = 1; // still settles, just without the idle drift afterwards

  // Glow: render the scene, then add a blurred copy of its bright parts on top. If the device
  // cannot build the pass (no float render targets), the scene is simply drawn without it.
  let composer: EffectComposer | null = null;
  let bloom: UnrealBloomPass | null = null;
  try {
    composer = new EffectComposer(renderer);
    composer.setPixelRatio(pixelRatio);
    composer.addPass(new RenderPass(scene, camera));
    bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.42, 0.5, 0.22);
    composer.addPass(bloom);
    // Thousands of additive points can sum far past white. A tone curve rolls the brightest
    // regions off smoothly instead of clipping them, so dense clusters keep their colour.
    renderer.toneMapping = THREE.ReinhardToneMapping;
    renderer.toneMappingExposure = 3.0; // keeps a lone point near its authored brightness
    composer.addPass(new OutputPass());
  } catch {
    composer = null;
    bloom = null;
  }

  let reveal = 1;
  let playing = false;
  let hovered = -1;
  let pointer: { x: number; y: number } | null = null;
  let pointerMoved = false;
  let downAt: { x: number; y: number } | null = null;
  let searchAbort: AbortController | null = null;
  let chunkAbort: AbortController | null = null;
  let hitSet = new Set<number>();
  let depthOn = true;

  const dateAt = (fraction: number) => {
    const i = Math.min(manifest.n - 1, Math.max(0, Math.round(fraction * (manifest.n - 1))));
    return new Date(meta[i * 3]! * 1000).toISOString().slice(0, 10);
  };
  const describe = (p: number) => `${manifest.projects[meta[p * 3 + 1]!]?.name ?? '?'} · ${new Date(meta[p * 3]! * 1000).toISOString().slice(0, 10)} · ${meta[p * 3 + 2] === 0 ? 'you' : 'assistant'}`;

  const select = async (particle: number) => {
    cloud.selected = particle;
    rig.focus(cloud.positionOf(particle), Math.min(rig.distance, 1.4));
    chunkAbort?.abort();
    chunkAbort = new AbortController();
    try {
      hud.showDetail(await chunk(particle, chunkAbort.signal));
    } catch (err) {
      if (!(err instanceof DOMException && err.name === 'AbortError')) hud.hideDetail();
    }
  };

  const clearSearch = () => {
    searchAbort?.abort();
    hitSet = new Set();
    cloud.setGlow([], false);
    trace.clear();
    hud.clearResults();
    canvas.dataset.hits = '0';
  };

  const runSearch = async (query: string, mode: Mode) => {
    searchAbort?.abort();
    searchAbort = new AbortController();
    hud.setBusy('searching…');
    try {
      const res = await search(query, mode, hud.focusedProject, searchAbort.signal);
      const hits = res.hits.map((h) => h.particle);
      hitSet = new Set(hits);
      // Nodes the search walked through glow faintly; the answers glow fully.
      const glow: [number, number][] = res.trace.map((s) => [s[1], 0.28]);
      for (const h of hits) glow.push([h, 1]);
      cloud.setGlow(glow, true);
      trace.play(res.trace, hits, reducedMotion);
      hud.setResults(res, trace.visited, trace.layers);
      canvas.dataset.hits = String(hits.length);
      canvas.dataset.traceSteps = String(res.trace.length);
      if (hits.length > 0) {
        if (reveal < 1) setReveal(1); // answers may be newer than the scrubber position
        const centre = new THREE.Vector3();
        for (const h of hits) centre.add(cloud.positionOf(h));
        centre.multiplyScalar(1 / hits.length);
        let spread = 0;
        for (const h of hits) spread = Math.max(spread, cloud.positionOf(h).distanceTo(centre));
        // Close enough to read the answers, far enough to keep the path that led to them in view.
        const field = cloud.extent().radius;
        rig.focus(centre, Math.min(8, Math.max(field * 1.15, spread * 2.6 + 0.8)));
      }
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') return;
      hud.setError(err instanceof Error ? err.message : 'Search failed');
    }
  };

  // Split the packed metadata into the columns the chart reads.
  const ts = new Uint32Array(manifest.n);
  const projectOf = new Uint32Array(manifest.n);
  const roleOf = new Uint32Array(manifest.n);
  for (let i = 0; i < manifest.n; i++) {
    ts[i] = meta[i * 3]!;
    projectOf[i] = meta[i * 3 + 1]!;
    roleOf[i] = meta[i * 3 + 2]!;
  }

  const setReveal = (fraction: number) => {
    reveal = Math.min(1, Math.max(0, fraction));
    cloud.reveal = reveal;
    activity?.setReveal(reveal);
    hud.setScrub(reveal, dateAt(reveal), playing);
    canvas.dataset.reveal = reveal.toFixed(3);
    labelsStale = true;
  };

  const applyToggle = (name: ToggleName, on: boolean) => {
    if (name === 'labels') {
      labels.visible = on;
      labelsStale = true;
    } else if (name === 'dust') cloud.dustVisible = on;
    else if (name === 'glow') {
      if (bloom) bloom.enabled = on;
    } else depthOn = on;
    canvas.dataset[name] = on ? '1' : '0';
  };

  let activity: Activity | null = null;
  let labelsStale = true;
  let repeatsCache: RepeatsView | null = null;
  let profileCache: ProfileView | null = null;

  const hud = new Hud(manifest, {
    onSearch: (q, mode) => void runSearch(q, mode),
    onClear: clearSearch,
    onProject: (index) => {
      cloud.focus = index ?? -1;
      labelsStale = true;
    },
    onReveal: (fraction) => {
      playing = false;
      setReveal(fraction);
    },
    onPlay: () => {
      playing = !playing;
      if (playing && reveal >= 1) setReveal(0);
      hud.setScrub(reveal, dateAt(reveal), playing);
    },
    onSelect: (particle) => void select(particle),
    onCloseDetail: () => {
      cloud.selected = -1;
      hud.hideDetail();
    },
    onRepeats: (open) => {
      if (!open) {
        hud.hideRepeats();
        cloud.setGlow([], false);
        canvas.dataset.repeat = '0';
        return;
      }
      clearSearch();
      void (async () => {
        try {
          repeatsCache ??= await repeats();
          hud.showRepeats(repeatsCache);
          canvas.dataset.repeats = String(repeatsCache.items?.length ?? 0);
        } catch (err) {
          hud.showRepeats({ available: false, reason: err instanceof Error ? err.message : 'Could not load repeats.' });
        }
      })();
    },
    onProfile: (open) => {
      if (!open) {
        hud.hideProfile();
        return;
      }
      clearSearch();
      cloud.setGlow([], false);
      canvas.dataset.repeat = '0';
      void (async () => {
        try {
          profileCache ??= await profile();
          hud.showProfile(profileCache);
          canvas.dataset.profile = profileCache.available ? '1' : '0';
        } catch (err) {
          hud.showProfile({ available: false, reason: err instanceof Error ? err.message : 'Could not load the profile.' });
        }
      })();
    },
    onRepeat: (item) => {
      // Light up every chunk where this was said, and frame them.
      trace.clear();
      hitSet = new Set(item.particles);
      cloud.setGlow(item.particles.map((p) => [p, 1] as [number, number]), true);
      canvas.dataset.repeat = String(item.particles.length);
      if (reveal < 1) setReveal(1);
      if (item.particles.length > 0) {
        const centre = new THREE.Vector3();
        for (const p of item.particles) centre.add(cloud.positionOf(p));
        centre.multiplyScalar(1 / item.particles.length);
        let spread = 0;
        for (const p of item.particles) spread = Math.max(spread, cloud.positionOf(p).distanceTo(centre));
        rig.focus(centre, Math.min(9, Math.max(cloud.extent().radius * 1.15, spread * 2.2 + 0.8)));
      }
    },
    onColorMode: (mode) => {
      cloud.setColorMode(mode);
      activity?.setMode(mode);
      canvas.dataset.color = mode;
    },
    onToggle: applyToggle,
  });

  const labels = new Labels(document.getElementById('labels') as HTMLElement, manifest.clusters, corpus.cluster, (centre, radius) => {
    rig.focus(centre, Math.min(6, Math.max(0.7, radius * 2.4)));
  });
  canvas.dataset.labels = '1';
  canvas.dataset.labelCount = String(labels.count);
  canvas.dataset.dustCount = String(cloud.dustCount);

  activity = new Activity(
    document.getElementById('activity') as HTMLCanvasElement,
    document.getElementById('activity-tip') as HTMLElement,
    document.getElementById('activity-table') as HTMLElement,
    { ts, project: projectOf, role: roleOf, projectNames: manifest.projects.map((p) => p.name) },
    (fraction) => {
      playing = false;
      setReveal(fraction);
    },
  );

  // Apply the saved view settings.
  cloud.setColorMode(hud.view.color);
  activity.setMode(hud.view.color);
  canvas.dataset.color = hud.view.color;
  for (const name of ['labels', 'dust', 'glow', 'depth'] as const) applyToggle(name, hud.view.toggles[name]);
  setReveal(1);

  // --- pointer: hover and click ---------------------------------------------------------------
  canvas.addEventListener('pointermove', (e) => {
    pointer = { x: e.clientX, y: e.clientY };
    pointerMoved = true;
  });
  canvas.addEventListener('pointerleave', () => {
    pointer = null;
    hovered = -1;
    cloud.hover = -1;
    hud.hideTooltip();
  });
  canvas.addEventListener('pointerdown', (e) => {
    downAt = { x: e.clientX, y: e.clientY };
    canvas.classList.add('dragging');
  });
  canvas.addEventListener('pointerup', (e) => {
    canvas.classList.remove('dragging');
    // A click is a press and release in nearly the same place; anything else was a drag.
    if (downAt && Math.hypot(e.clientX - downAt.x, e.clientY - downAt.y) < 5 && hovered >= 0) void select(hovered);
    downAt = null;
  });

  const projected = new THREE.Vector3();
  const viewProj = new THREE.Matrix4();
  /** Nearest visible particle to a screen point, within `radius` CSS pixels, or -1. */
  const pick = (x: number, y: number, radius: number): number => {
    const rect = canvas.getBoundingClientRect();
    viewProj.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    const focus = hud.focusedProject;
    const limit = Math.round(reveal * (manifest.n - 1));
    let best = -1;
    let bestD = radius * radius;
    const pos = cloud.positions;
    for (let i = 0; i <= limit; i++) {
      if (focus !== null && meta[i * 3 + 1] !== focus) continue;
      projected.set(pos[i * 4]!, pos[i * 4 + 1]!, pos[i * 4 + 2]!).applyMatrix4(viewProj);
      if (projected.z < -1 || projected.z > 1) continue;
      const sx = rect.left + (projected.x * 0.5 + 0.5) * rect.width;
      const sy = rect.top + (1 - (projected.y * 0.5 + 0.5)) * rect.height;
      let d = (sx - x) ** 2 + (sy - y) ** 2;
      if (hitSet.has(i)) d *= 0.35; // answers are easier to grab
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    }
    return best;
  };

  // --- keyboard -------------------------------------------------------------------------------
  window.addEventListener('keydown', (e) => {
    if (e.key === '/' && !hud.searchHasFocus) {
      e.preventDefault();
      hud.focusSearch();
    } else if (e.key === 'Escape') {
      clearSearch();
      hud.hideRepeats();
      hud.hideProfile();
      cloud.selected = -1;
      hud.hideDetail();
      (document.activeElement as HTMLElement | null)?.blur();
    }
  });

  // --- hand gestures: loaded only when switched on ---------------------------------------------
  // Not in the public demo: it ships no camera code, no WebAssembly and no gesture model.
  const gestureToggle = document.getElementById('gesture-toggle') as HTMLButtonElement;
  let gestures: { stop: () => void } | null = null;
  // Written out in full so the bundler can drop the branch, and with it the camera code, from the demo build.
  if (import.meta.env.MODE === 'pages') (document.getElementById('gesture') as HTMLElement).hidden = true;
  else gestureToggle.addEventListener('click', async () => {
    if (gestures) {
      gestures.stop();
      gestures = null;
      gestureToggle.textContent = 'hand control: off';
      gestureToggle.setAttribute('aria-pressed', 'false');
      rig.rates = { yaw: 0, pitch: 0, zoom: 0 };
      return;
    }
    gestureToggle.textContent = 'hand control: starting…';
    try {
      const { startGestures } = await import('./gesture/control.ts');
      gestures = await startGestures({
        rig,
        pickAt: (nx, ny) => {
          const rect = canvas.getBoundingClientRect();
          return pick(rect.left + nx * rect.width, rect.top + ny * rect.height, 40);
        },
        onPoint: (particle) => {
          hovered = particle;
          cloud.hover = particle;
        },
        onSelect: (particle) => void select(particle),
      });
      gestureToggle.textContent = 'hand control: on';
      gestureToggle.setAttribute('aria-pressed', 'true');
    } catch (err) {
      gestureToggle.textContent = 'hand control: off';
      hud.setError(err instanceof Error ? err.message : 'Could not start hand control');
    }
  });

  // --- resize and render loop -----------------------------------------------------------------
  let viewW = 1;
  let viewH = 1;
  const resize = () => {
    viewW = canvas.clientWidth;
    viewH = canvas.clientHeight;
    renderer.setSize(viewW, viewH, false);
    composer?.setSize(viewW, viewH);
    camera.aspect = viewW / Math.max(1, viewH);
    camera.updateProjectionMatrix();
    cloud.resize(pixelRatio, viewH);
    trace.resize(pixelRatio, viewH);
  };
  // The canvas can change size without the window doing so (side panels, split views).
  new ResizeObserver(resize).observe(canvas);
  resize();

  // Frame the whole field once it has had a moment to arrange itself.
  const fit = () => {
    const { centre, radius } = cloud.extent();
    rig.focus(centre, Math.min(12, Math.max(1.2, radius * 2.35)));
  };
  let fitted = false;

  // Debug hook for tuning and tests, only with ?debug in the URL. Exposes rendering internals,
  // never conversation text.
  if (new URLSearchParams(location.search).has('debug')) {
    (window as unknown as Record<string, unknown>).__loci = { cloud, rig, trace, fit, bloom, labels };
  }

  let last = performance.now();
  let raf = 0;
  let seenReadbacks = -1;
  const frame = (now: number) => {
    raf = requestAnimationFrame(frame);
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    const time = now / 1000;

    if (playing) {
      setReveal(reveal + dt / 14);
      if (reveal >= 1) {
        playing = false;
        hud.setScrub(1, dateAt(1), false);
      }
    }
    rig.update(dt, reducedMotion);
    camera.updateMatrixWorld();
    // Whatever the camera is looking at is what is in focus.
    cloud.setLens(rig.distance, depthOn ? APERTURE : 0);
    cloud.update(dt, time, reducedMotion);
    trace.update(dt, time);
    if (!fitted && cloud.frame === 240) {
      fitted = true;
      fit();
    }

    if (pointer && pointerMoved && !rig.isDragging) {
      pointerMoved = false;
      hovered = pick(pointer.x, pointer.y, 16);
      cloud.hover = hovered;
      if (hovered >= 0) hud.showTooltip(pointer.x, pointer.y, describe(hovered));
      else hud.hideTooltip();
      canvas.style.cursor = hovered >= 0 ? 'pointer' : '';
    }

    // Label centres follow the particles: recount whenever fresh positions arrive from the GPU,
    // or when what is shown has changed.
    if (labelsStale || cloud.readbacks !== seenReadbacks) {
      labels.recount(cloud.positions, Math.round(reveal * (manifest.n - 1)), hud.focusedProject, meta);
      labelsStale = false;
      seenReadbacks = cloud.readbacks;
    }
    labels.place(camera, viewW, viewH, rig.distance);

    if (composer) composer.render(dt);
    else renderer.render(scene, camera);

    // Hooks for the end-to-end test: counts and a few live positions, nothing private.
    if (cloud.frame % 20 === 0) {
      canvas.dataset.frame = String(cloud.frame);
      canvas.dataset.alpha = cloud.alpha.toFixed(4);
      canvas.dataset.sample = Array.from(cloud.positions.subarray(0, 12))
        .map((v) => v.toFixed(4))
        .join(',');
    }
  };
  canvas.dataset.particles = String(manifest.n);
  if (STATIC) {
    // The welcome panel's "try" buttons press the real controls, so they show what a visitor would do.
    setupIntro({
      search: (query) => hud.search(query),
      repeats: () => document.getElementById('open-repeats')?.click(),
      play: () => document.getElementById('play')?.click(),
    });
  }
  canvas.dataset.ready = '1';
  raf = requestAnimationFrame(frame);

  // Stop drawing while the tab is hidden; resume cleanly when it returns.
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) cancelAnimationFrame(raf);
    else {
      last = performance.now();
      raf = requestAnimationFrame(frame);
    }
  });
  window.addEventListener('pagehide', () => {
    cancelAnimationFrame(raf);
    gestures?.stop();
    rig.dispose();
    trace.dispose();
    cloud.dispose();
    composer?.dispose();
    renderer.dispose();
  });
}

void start();
