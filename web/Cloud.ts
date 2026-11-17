import * as THREE from 'three';
import type { Corpus } from './api.ts';
import { OTHER, SLOTS, SERIES, SPEAKER, hexRgb, timeRgb } from './palette.ts';
import type { ColorMode, Rgb } from './palette.ts';

/**
 * The particle field: one point per chunk, arranged by a force simulation that runs entirely on
 * the GPU.
 *
 * Positions live in a float texture (one texel per particle). Each frame a full-screen pass reads
 * that texture and writes the next state into a second one, then the two swap (ping-pong). The
 * forces are the index's own structure:
 *
 *   - each particle is drawn to its place in a UMAP-style layout of the HNSW neighbour graph
 *     (computed once on the server, src/index/layout.ts)
 *   - it is also tied by springs to its nearest neighbours in that graph, so clusters move and
 *     settle together rather than as loose dust
 *   - and pushed away from a few randomly chosen particles per frame (negative sampling), which
 *     keeps points from stacking on top of each other
 *
 * Particles start at a PCA projection of their embeddings and flow into the graph layout, so what
 * you watch settling is the search index arranging itself: things the index considers neighbours
 * end up next to each other. Points and edge lines read their positions from the same texture in
 * the vertex shader, so the CPU never touches per-particle positions except for a periodic
 * readback used for picking.
 *
 * Three layers are drawn:
 *
 *   core   one point per chunk. Its position is the data.
 *   dust   one small point per sentence in the chunk, scattered around the core and slowly
 *          orbiting it. The NUMBER of dust points is real (longer chunks look denser); their
 *          scatter is decoration and carries no meaning.
 *   edges  each chunk to its nearest few neighbours in the index.
 *
 * Depth of field is done per point rather than as a screen pass: a point away from the focal
 * distance is drawn larger and proportionally fainter, so its light is spread, not added.
 */

const EDGE_NEIGHBOURS = 4;

const QUAD_VERT = /* glsl */ `varying vec2 vUv; void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`;

const SIM_FRAG = /* glsl */ `
  precision highp float;
  uniform sampler2D uPos; uniform sampler2D uHome; uniform sampler2D uKnnA; uniform sampler2D uKnnB;
  uniform float uSize; uniform float uCount; uniform float uAlpha; uniform float uFrame;
  uniform float uRest; uniform float uAttract; uniform float uRepel; uniform float uAnchor; uniform float uRange;
  varying vec2 vUv;

  vec2 uvOf(float idx) { float y = floor(idx / uSize); return (vec2(idx - y * uSize, y) + 0.5) / uSize; }
  float hash(vec2 p) { vec3 q = fract(vec3(p.xyx) * 0.1031); q += dot(q, q.yzx + 33.33); return fract((q.x + q.y) * q.z); }

  vec3 spring(vec3 p, float n) {
    vec3 d = texture2D(uPos, uvOf(n)).xyz - p;
    float dist = length(d);
    if (dist < 1e-6) return vec3(0.0); // padding entries point at the particle itself
    return (d / dist) * (dist - uRest) * uAttract;
  }

  void main() {
    vec4 self = texture2D(uPos, vUv);
    vec2 texel = floor(vUv * uSize);
    float idx = texel.y * uSize + texel.x;
    if (idx >= uCount || uAlpha <= 0.0) { gl_FragColor = self; return; }
    vec3 p = self.xyz;
    vec4 a = texture2D(uKnnA, vUv);
    vec4 b = texture2D(uKnnB, vUv);
    vec3 f = spring(p, a.x) + spring(p, a.y) + spring(p, a.z) + spring(p, a.w)
           + spring(p, b.x) + spring(p, b.y) + spring(p, b.z) + spring(p, b.w);
    for (int s = 0; s < 6; s++) {
      float r = hash(vec2(idx * 0.6180339 + float(s) * 19.19, uFrame * 0.7548777 + float(s) * 7.31));
      vec3 d = p - texture2D(uPos, uvOf(floor(r * uCount))).xyz;
      float d2 = dot(d, d);
      f += d * uRepel / ((0.0006 + d2) * (1.0 + uRange * d2));
    }
    // The pull toward this particle's place in the graph layout.
    f += (texture2D(uHome, vUv).xyz - p) * uAnchor;
    vec3 step = f * uAlpha;
    float len = length(step);
    if (len > 0.04) step *= 0.04 / len; // never let one frame throw a particle across the scene
    gl_FragColor = vec4(p + step, self.w);
  }
`;

/** Shared by core and dust: blur radius in pixels for a point at this depth. */
const COC = /* glsl */ `
  uniform float uFocusDist; uniform float uAperture;
  float circleOfConfusion(float depth, float scale) {
    return uAperture * abs(depth - uFocusDist) / depth * scale;
  }
`;

const POINT_VERT = /* glsl */ `
  attribute vec2 ref; attribute vec3 aColor; attribute float aRank; attribute float aRole; attribute float aSeed;
  attribute float aGlow; attribute float aProject; attribute float aExpose;
  uniform sampler2D uPos; uniform float uReveal; uniform float uPixelRatio; uniform float uScale; uniform float uTime;
  uniform float uDim; uniform float uFocus; uniform float uHover; uniform float uSelected;
  ${COC}
  varying vec3 vColor; varying float vAlpha; varying float vGlow;
  void main() {
    vec3 p = texture2D(uPos, ref).xyz;
    // Display-only drift so a settled field still feels alive; the simulation state is untouched.
    p += 0.004 * vec3(sin(uTime * 0.6 + aSeed * 40.0), cos(uTime * 0.5 + aSeed * 63.0), sin(uTime * 0.7 + aSeed * 91.0));
    vec4 mv = modelViewMatrix * vec4(p, 1.0);
    gl_Position = projectionMatrix * mv;
    float depth = max(0.15, -mv.z);
    float shown = step(aRank, uReveal);
    // Chunks close to the scrubber's edge flash as they arrive.
    float fresh = smoothstep(0.012, 0.0, uReveal - aRank) * step(uReveal, 0.9999);
    float focus = (uFocus < 0.0 || abs(aProject - uFocus) < 0.5) ? 1.0 : 0.12;
    float isHover = abs(aRank - uHover) < 1e-7 ? 1.0 : 0.0;
    float isSel = abs(aRank - uSelected) < 1e-7 ? 1.0 : 0.0;
    float glow = max(max(aGlow, isHover), isSel);
    float base = mix(1.0, 1.55, 1.0 - aRole); // what you typed is drawn a little larger
    float twinkle = 0.88 + 0.12 * sin(uTime * 1.7 + aSeed * 120.0);
    float scale = uPixelRatio * uScale;
    // Perspective attenuation, referenced to a viewing distance of 3 units.
    float sharp = (3.0 * base + 6.0 * glow + 4.5 * fresh) * twinkle * scale * 3.0 / depth;
    // Answers and the point under the cursor stay mostly in focus so they can be read.
    float size = sharp + circleOfConfusion(depth, scale) * (1.0 - 0.75 * glow);
    gl_PointSize = clamp(size, 1.0, 30.0);
    // Spreading a point over a bigger disc must not add light: scale alpha by the area ratio.
    float spread = clamp((sharp * sharp) / (size * size), 0.07, 1.0);
    vColor = mix(aColor, vec3(1.0), 0.55 * glow + 0.5 * fresh);
    vGlow = glow;
    // aExpose turns crowded regions down so they do not burn out to white under additive blending.
    vAlpha = shown * focus * mix(uDim, 1.0, clamp(glow + fresh, 0.0, 1.0)) * (0.7 + 0.2 * base) * spread * mix(aExpose, 1.0, glow);
  }
`;

const POINT_FRAG = /* glsl */ `
  precision highp float;
  varying vec3 vColor; varying float vAlpha; varying float vGlow;
  void main() {
    vec2 uv = gl_PointCoord - 0.5;
    float d = length(uv) * 2.0;
    float core = smoothstep(1.0, 0.12, d);
    float halo = exp(-d * d * 3.0) * (0.42 + 0.5 * vGlow);
    float a = (core + halo) * vAlpha;
    if (a < 0.004) discard;
    gl_FragColor = vec4(vColor * a, a);
  }
`;

const DUST_VERT = /* glsl */ `
  attribute vec2 ref; attribute vec3 aColor; attribute float aRank; attribute float aProject; attribute vec3 aOffset; attribute float aSeed;
  attribute float aExpose;
  uniform sampler2D uPos; uniform float uReveal; uniform float uPixelRatio; uniform float uScale; uniform float uTime;
  uniform float uDim; uniform float uFocus; uniform float uSpread;
  ${COC}
  varying vec3 vColor; varying float vAlpha;
  void main() {
    // Each dust point circles its chunk slowly, at its own speed.
    float a = uTime * (0.12 + 0.22 * aSeed) + aSeed * 6.2831853;
    float ca = cos(a); float sa = sin(a);
    vec3 o = vec3(aOffset.x * ca - aOffset.z * sa, aOffset.y, aOffset.x * sa + aOffset.z * ca);
    // Scatter shrinks where chunks are packed tightly, so dust stays with its own chunk.
    vec4 mv = modelViewMatrix * vec4(texture2D(uPos, ref).xyz + o * uSpread * aExpose, 1.0);
    gl_Position = projectionMatrix * mv;
    float depth = max(0.15, -mv.z);
    float scale = uPixelRatio * uScale;
    float sharp = (1.3 + 0.6 * aSeed) * scale * 3.0 / depth;
    float size = sharp + circleOfConfusion(depth, scale);
    gl_PointSize = clamp(size, 1.0, 22.0);
    float spread = clamp((sharp * sharp) / (size * size), 0.05, 1.0);
    float focus = (uFocus < 0.0 || abs(aProject - uFocus) < 0.5) ? 1.0 : 0.08;
    vColor = aColor;
    vAlpha = step(aRank, uReveal) * focus * uDim * 0.5 * spread * aExpose;
  }
`;

const DUST_FRAG = /* glsl */ `
  precision highp float;
  varying vec3 vColor; varying float vAlpha;
  void main() {
    float d = length(gl_PointCoord - 0.5) * 2.0;
    float a = exp(-d * d * 2.6) * vAlpha;
    if (a < 0.004) discard;
    gl_FragColor = vec4(vColor * a, a);
  }
`;

const EDGE_VERT = /* glsl */ `
  attribute vec2 ref; attribute vec3 aColor; attribute float aRank; attribute float aProject;
  uniform sampler2D uPos; uniform float uReveal; uniform float uFocus; uniform float uOpacity;
  varying vec3 vColor; varying float vAlpha;
  void main() {
    vec4 mv = modelViewMatrix * vec4(texture2D(uPos, ref).xyz, 1.0);
    gl_Position = projectionMatrix * mv;
    float focus = (uFocus < 0.0 || abs(aProject - uFocus) < 0.5) ? 1.0 : 0.08;
    vColor = aColor;
    vAlpha = step(aRank, uReveal) * focus * uOpacity * clamp(2.2 / max(0.4, -mv.z), 0.25, 1.6);
  }
`;

const EDGE_FRAG = /* glsl */ `
  precision highp float;
  varying vec3 vColor; varying float vAlpha;
  void main() { gl_FragColor = vec4(vColor * vAlpha, vAlpha); }
`;

export class Cloud {
  readonly n: number;
  readonly size: number;
  readonly group = new THREE.Group();
  /** Latest positions read back from the GPU (n * 4 floats, xyz + unused). Used for picking. */
  readonly positions: Float32Array;
  /** How many dust points are drawn in total. */
  readonly dustCount: number;
  /** Simulation step size: 1 when freshly started, decaying to a small floor. */
  alpha = 1;
  frame = 0;
  /** Increases every time `positions` is refreshed from the GPU. */
  readbacks = 0;

  private rtA: THREE.WebGLRenderTarget;
  private rtB: THREE.WebGLRenderTarget;
  private readonly simScene = new THREE.Scene();
  private readonly simCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  private readonly simMaterial: THREE.ShaderMaterial;
  private readonly pointMaterial: THREE.ShaderMaterial;
  private readonly dustMaterial: THREE.ShaderMaterial;
  private readonly edgeMaterial: THREE.ShaderMaterial;
  private readonly points: THREE.Points;
  private readonly dust: THREE.Points;
  private readonly edges: THREE.LineSegments;
  private readonly glow: THREE.BufferAttribute;
  private readonly textures: THREE.Texture[] = [];
  private readonly renderer: THREE.WebGLRenderer;
  private readonly rank: Float32Array;
  private readonly meta: Uint32Array;
  /** Which particle each edge vertex and each dust point belongs to, for recolouring. */
  private readonly edgeOwner: Uint32Array;
  private readonly dustOwner: Uint32Array;
  private readbackPending = true;

  constructor(renderer: THREE.WebGLRenderer, corpus: Corpus) {
    this.renderer = renderer;
    const { manifest, pos, home, knn, meta, sentences } = corpus;
    this.meta = meta;
    const n = (this.n = manifest.n);
    const size = (this.size = manifest.textureSize);
    const texels = size * size;
    this.positions = new Float32Array(texels * 4);

    // Float render targets need an extension on WebGL2; fall back to half floats without it.
    const floatOk = renderer.extensions.has('EXT_color_buffer_float');
    const type = floatOk ? THREE.FloatType : THREE.HalfFloatType;
    const mk = () =>
      new THREE.WebGLRenderTarget(size, size, { type, format: THREE.RGBAFormat, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter, depthBuffer: false, stencilBuffer: false });
    this.rtA = mk();
    this.rtB = mk();

    const seed = new Float32Array(texels * 4);
    const homeData = new Float32Array(texels * 4);
    const knnA = new Float32Array(texels * 4);
    const knnB = new Float32Array(texels * 4);
    for (let i = 0; i < n; i++) {
      seed[i * 4] = pos[i * 3]!;
      seed[i * 4 + 1] = pos[i * 3 + 1]!;
      seed[i * 4 + 2] = pos[i * 3 + 2]!;
      homeData[i * 4] = home[i * 3]!;
      homeData[i * 4 + 1] = home[i * 3 + 1]!;
      homeData[i * 4 + 2] = home[i * 3 + 2]!;
      for (let j = 0; j < 4; j++) {
        knnA[i * 4 + j] = knn[i * manifest.knn + j] ?? i;
        knnB[i * 4 + j] = knn[i * manifest.knn + 4 + j] ?? i;
      }
    }
    this.positions.set(seed);
    const tex = (data: Float32Array) => {
      const t = new THREE.DataTexture(data, size, size, THREE.RGBAFormat, THREE.FloatType);
      t.minFilter = t.magFilter = THREE.NearestFilter;
      t.needsUpdate = true;
      this.textures.push(t);
      return t;
    };
    const seedTex = tex(seed);

    // Rest length scales with the typical spacing of n points in a unit-ish ball.
    const rest = 0.5 * Math.cbrt(1 / Math.max(n, 1));
    this.simMaterial = new THREE.ShaderMaterial({
      uniforms: {
        uPos: { value: seedTex },
        uHome: { value: tex(homeData) },
        uKnnA: { value: tex(knnA) },
        uKnnB: { value: tex(knnB) },
        uSize: { value: size },
        uCount: { value: n },
        uAlpha: { value: 0 },
        uFrame: { value: 0 },
        uRest: { value: rest },
        uAttract: { value: 0.02 },
        uRepel: { value: 0.00025 },
        uAnchor: { value: 0.05 },
        uRange: { value: 40 },
      },
      vertexShader: QUAD_VERT,
      fragmentShader: SIM_FRAG,
      depthTest: false,
      depthWrite: false,
    });
    this.simScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.simMaterial));
    // Prime both targets with the starting positions (uAlpha = 0 copies the input through).
    for (const rt of [this.rtA, this.rtB]) {
      renderer.setRenderTarget(rt);
      renderer.render(this.simScene, this.simCamera);
    }
    renderer.setRenderTarget(null);

    // ---- exposure -------------------------------------------------------------------------
    // How crowded each particle's neighbourhood is in the settled layout: its mean distance to
    // its graph neighbours, relative to the median. Crowded regions are drawn fainter and sparse
    // ones stronger, which keeps dense clusters from saturating without dimming everything.
    const spacing = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      let sum = 0;
      let count = 0;
      for (let j = 0; j < manifest.knn; j++) {
        const k = knn[i * manifest.knn + j]!;
        if (k === i || k >= n) continue;
        sum += Math.hypot(home[i * 3]! - home[k * 3]!, home[i * 3 + 1]! - home[k * 3 + 1]!, home[i * 3 + 2]! - home[k * 3 + 2]!);
        count++;
      }
      spacing[i] = count ? sum / count : 0;
    }
    const sorted = Float32Array.from(spacing).sort();
    const median = sorted[Math.floor(n / 2)] || 1;
    // Also account for the field as a whole: a tightly packed layout needs less light per point.
    const typical = Math.cbrt(1 / Math.max(n, 1));
    const global = Math.min(1, Math.max(0.15, (median / typical) * 1.6));
    const expose = new Float32Array(n);
    for (let i = 0; i < n; i++) expose[i] = Math.min(1.5, Math.max(0.3, spacing[i]! / median)) * global;

    // ---- core points -----------------------------------------------------------------------
    const ref = new Float32Array(n * 2);
    const rank = (this.rank = new Float32Array(n));
    const role = new Float32Array(n);
    const seedAttr = new Float32Array(n);
    const project = new Float32Array(n);
    const hash01 = (i: number, salt: number) => (Math.imul(i + 1, 2654435761 + salt * 40503) >>> 0) / 4294967296;
    for (let i = 0; i < n; i++) {
      ref[i * 2] = ((i % size) + 0.5) / size;
      ref[i * 2 + 1] = (Math.floor(i / size) + 0.5) / size;
      rank[i] = n > 1 ? i / (n - 1) : 0;
      role[i] = meta[i * 3 + 2]!;
      seedAttr[i] = hash01(i, 0);
      project[i] = meta[i * 3 + 1]!;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    geo.setAttribute('ref', new THREE.BufferAttribute(ref, 2));
    geo.setAttribute('aColor', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    geo.setAttribute('aRank', new THREE.BufferAttribute(rank, 1));
    geo.setAttribute('aRole', new THREE.BufferAttribute(role, 1));
    geo.setAttribute('aSeed', new THREE.BufferAttribute(seedAttr, 1));
    geo.setAttribute('aProject', new THREE.BufferAttribute(project, 1));
    geo.setAttribute('aExpose', new THREE.BufferAttribute(expose, 1));
    this.glow = new THREE.BufferAttribute(new Float32Array(n), 1);
    this.glow.setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute('aGlow', this.glow);

    const lens = { uFocusDist: { value: 4 }, uAperture: { value: 0 } };
    const shared = () => ({
      uPos: { value: this.rtA.texture },
      uReveal: { value: 1 },
      uPixelRatio: { value: renderer.getPixelRatio() },
      uScale: { value: 1 },
      uTime: { value: 0 },
      uDim: { value: 1 },
      uFocus: { value: -1 },
      uFocusDist: lens.uFocusDist,
      uAperture: lens.uAperture,
    });
    const additive = { transparent: true, depthTest: false, depthWrite: false, blending: THREE.AdditiveBlending } as const;

    this.pointMaterial = new THREE.ShaderMaterial({
      uniforms: { ...shared(), uHover: { value: -1 }, uSelected: { value: -1 } },
      vertexShader: POINT_VERT,
      fragmentShader: POINT_FRAG,
      ...additive,
    });
    this.points = new THREE.Points(geo, this.pointMaterial);
    this.points.frustumCulled = false;
    this.points.renderOrder = 2;

    // ---- dust: one small point per sentence, scattered around its chunk ---------------------
    let dustCount = 0;
    for (let i = 0; i < n; i++) dustCount += Math.max(0, (sentences[i] ?? 1) - 1); // the core point stands for the first sentence
    this.dustCount = dustCount;
    const dRef = new Float32Array(dustCount * 2);
    const dRank = new Float32Array(dustCount);
    const dProject = new Float32Array(dustCount);
    const dOffset = new Float32Array(dustCount * 3);
    const dSeed = new Float32Array(dustCount);
    const dExpose = new Float32Array(dustCount);
    this.dustOwner = new Uint32Array(dustCount);
    let d = 0;
    for (let i = 0; i < n; i++) {
      const extra = Math.max(0, (sentences[i] ?? 1) - 1);
      for (let s = 0; s < extra; s++, d++) {
        // A deterministic point inside a unit ball, denser towards the centre.
        const u = hash01(d, 1) * 2 - 1;
        const phi = hash01(d, 2) * Math.PI * 2;
        const r = Math.pow(hash01(d, 3), 0.6);
        const w = Math.sqrt(1 - u * u);
        dOffset[d * 3] = r * w * Math.cos(phi);
        dOffset[d * 3 + 1] = r * u;
        dOffset[d * 3 + 2] = r * w * Math.sin(phi);
        dRef[d * 2] = ref[i * 2]!;
        dRef[d * 2 + 1] = ref[i * 2 + 1]!;
        dRank[d] = rank[i]!;
        dProject[d] = project[i]!;
        dSeed[d] = hash01(d, 4);
        dExpose[d] = expose[i]!;
        this.dustOwner[d] = i;
      }
    }
    const dGeo = new THREE.BufferGeometry();
    dGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(dustCount * 3), 3));
    dGeo.setAttribute('ref', new THREE.BufferAttribute(dRef, 2));
    dGeo.setAttribute('aColor', new THREE.BufferAttribute(new Float32Array(dustCount * 3), 3));
    dGeo.setAttribute('aRank', new THREE.BufferAttribute(dRank, 1));
    dGeo.setAttribute('aProject', new THREE.BufferAttribute(dProject, 1));
    dGeo.setAttribute('aOffset', new THREE.BufferAttribute(dOffset, 3));
    dGeo.setAttribute('aSeed', new THREE.BufferAttribute(dSeed, 1));
    dGeo.setAttribute('aExpose', new THREE.BufferAttribute(dExpose, 1));
    this.dustMaterial = new THREE.ShaderMaterial({
      // Scatter radius: a little over the typical gap between neighbouring chunks.
      uniforms: { ...shared(), uSpread: { value: 1.6 * Math.cbrt(1 / Math.max(n, 1)) } },
      vertexShader: DUST_VERT,
      fragmentShader: DUST_FRAG,
      ...additive,
    });
    this.dust = new THREE.Points(dGeo, this.dustMaterial);
    this.dust.frustumCulled = false;
    this.dust.renderOrder = 1;

    // ---- edges: each particle to its nearest few graph neighbours ---------------------------
    const pairs: number[] = [];
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < EDGE_NEIGHBOURS; j++) {
        const k = knn[i * manifest.knn + j]!;
        if (k !== i && k < n) pairs.push(i, k);
      }
    }
    const eRef = new Float32Array(pairs.length * 2);
    const eRank = new Float32Array(pairs.length);
    const eProject = new Float32Array(pairs.length);
    this.edgeOwner = Uint32Array.from(pairs);
    for (let e = 0; e < pairs.length; e += 2) {
      const a = pairs[e]!;
      const b = pairs[e + 1]!;
      const later = Math.max(rank[a]!, rank[b]!); // an edge appears once both ends exist
      for (let v = 0; v < 2; v++) {
        const i = v === 0 ? a : b;
        eRef[(e + v) * 2] = ref[i * 2]!;
        eRef[(e + v) * 2 + 1] = ref[i * 2 + 1]!;
        eRank[e + v] = later;
        eProject[e + v] = project[a]!;
      }
    }
    const eGeo = new THREE.BufferGeometry();
    eGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(pairs.length * 3), 3));
    eGeo.setAttribute('ref', new THREE.BufferAttribute(eRef, 2));
    eGeo.setAttribute('aColor', new THREE.BufferAttribute(new Float32Array(pairs.length * 3), 3));
    eGeo.setAttribute('aRank', new THREE.BufferAttribute(eRank, 1));
    eGeo.setAttribute('aProject', new THREE.BufferAttribute(eProject, 1));
    this.edgeMaterial = new THREE.ShaderMaterial({
      uniforms: { uPos: { value: this.rtA.texture }, uReveal: { value: 1 }, uFocus: { value: -1 }, uOpacity: { value: 0.2 } },
      vertexShader: EDGE_VERT,
      fragmentShader: EDGE_FRAG,
      ...additive,
    });
    this.edges = new THREE.LineSegments(eGeo, this.edgeMaterial);
    this.edges.frustumCulled = false;
    this.edges.renderOrder = 0;

    this.group.add(this.edges, this.dust, this.points);
    this.setColorMode('project');
  }

  /** Simulation uniforms, exposed for tuning from the debug hook. */
  get forces(): Record<string, THREE.IUniform> {
    return this.simMaterial.uniforms;
  }

  /** Colour of one particle under a colouring mode. */
  colorOf(particle: number, mode: ColorMode): Rgb {
    if (mode === 'time') return timeRgb(this.rank[particle] ?? 0);
    if (mode === 'speaker') return hexRgb(this.meta[particle * 3 + 2] === 0 ? SPEAKER.human : SPEAKER.assistant);
    const project = this.meta[particle * 3 + 1]!;
    return hexRgb(project < SLOTS ? SERIES[project]! : OTHER);
  }

  /** Recolours every layer. Colour always follows the entity: a project, a moment, or a speaker. */
  setColorMode(mode: ColorMode): void {
    const perParticle = new Float32Array(this.n * 3);
    for (let i = 0; i < this.n; i++) perParticle.set(this.colorOf(i, mode), i * 3);
    const fill = (geometry: THREE.BufferGeometry, owner: ArrayLike<number> | null) => {
      const attr = geometry.getAttribute('aColor') as THREE.BufferAttribute;
      const arr = attr.array as Float32Array;
      if (!owner) arr.set(perParticle);
      else for (let v = 0; v < owner.length; v++) arr.set(perParticle.subarray(owner[v]! * 3, owner[v]! * 3 + 3), v * 3);
      attr.needsUpdate = true;
    };
    fill(this.points.geometry, null);
    fill(this.dust.geometry, this.dustOwner);
    fill(this.edges.geometry, this.edgeOwner);
  }

  /** Centre and framing radius of the field from the last readback. */
  extent(): { centre: THREE.Vector3; radius: number } {
    const centre = new THREE.Vector3();
    const pos = this.positions;
    for (let i = 0; i < this.n; i++) {
      centre.x += pos[i * 4]!;
      centre.y += pos[i * 4 + 1]!;
      centre.z += pos[i * 4 + 2]!;
    }
    centre.multiplyScalar(1 / Math.max(this.n, 1));
    // 97th-percentile distance from the centre: a radius that frames nearly everything without
    // letting a few stragglers zoom the camera out.
    const dist = new Float32Array(this.n);
    for (let i = 0; i < this.n; i++) dist[i] = Math.hypot(pos[i * 4]! - centre.x, pos[i * 4 + 1]! - centre.y, pos[i * 4 + 2]! - centre.z);
    dist.sort();
    return { centre, radius: dist[Math.floor(this.n * 0.97)] ?? 1 };
  }

  /** The texture holding current positions, for other layers (trace, markers) to read. */
  get positionTexture(): THREE.Texture {
    return this.rtA.texture;
  }

  /** UV of a particle's texel in the position texture. */
  refOf(particle: number): [number, number] {
    return [((particle % this.size) + 0.5) / this.size, (Math.floor(particle / this.size) + 0.5) / this.size];
  }

  /** World position of a particle from the last readback. */
  positionOf(particle: number, out = new THREE.Vector3()): THREE.Vector3 {
    return out.set(this.positions[particle * 4]!, this.positions[particle * 4 + 1]!, this.positions[particle * 4 + 2]!);
  }

  rankOf(particle: number): number {
    return this.rank[particle] ?? -1;
  }

  /** Restarts the settling animation. */
  reheat(alpha = 1): void {
    this.alpha = Math.max(this.alpha, alpha);
  }

  private each(name: string, value: number): void {
    for (const m of [this.pointMaterial, this.dustMaterial, this.edgeMaterial]) {
      const u = m.uniforms[name];
      if (u) u.value = value;
    }
  }

  set reveal(v: number) {
    this.each('uReveal', v);
  }

  get reveal(): number {
    return this.pointMaterial.uniforms.uReveal!.value as number;
  }

  /** Focus one project (dims the rest), or -1 for all. */
  set focus(project: number) {
    this.each('uFocus', project);
  }

  set hover(particle: number) {
    this.pointMaterial.uniforms.uHover!.value = particle < 0 ? -1 : this.rankOf(particle);
  }

  set selected(particle: number) {
    this.pointMaterial.uniforms.uSelected!.value = particle < 0 ? -1 : this.rankOf(particle);
  }

  /** Lens: the distance that is in focus, and how strongly everything else blurs (0 turns it off). */
  setLens(focusDistance: number, aperture: number): void {
    this.pointMaterial.uniforms.uFocusDist!.value = focusDistance;
    this.pointMaterial.uniforms.uAperture!.value = aperture;
  }

  set dustVisible(on: boolean) {
    this.dust.visible = on;
  }

  /** Marks particles to glow (search hits = 1, nodes the search walked through = a fraction). */
  setGlow(entries: Iterable<[particle: number, amount: number]>, dimOthers: boolean): void {
    const arr = this.glow.array as Float32Array;
    arr.fill(0);
    for (const [p, amount] of entries) if (p >= 0 && p < this.n) arr[p] = Math.max(arr[p]!, amount);
    this.glow.needsUpdate = true;
    this.pointMaterial.uniforms.uDim!.value = dimOthers ? 0.3 : 1;
    this.dustMaterial.uniforms.uDim!.value = dimOthers ? 0.22 : 1;
    this.edgeMaterial.uniforms.uOpacity!.value = dimOthers ? 0.06 : 0.2;
  }

  resize(pixelRatio: number, height: number): void {
    for (const m of [this.pointMaterial, this.dustMaterial]) {
      m.uniforms.uPixelRatio!.value = pixelRatio;
      // Keeps points the same apparent size whatever the canvas height.
      m.uniforms.uScale!.value = height / 760;
    }
  }

  /** Advances the simulation one step and reads positions back when needed. */
  update(dt: number, time: number, settle: boolean): void {
    const floor = settle ? 0 : 0.012;
    // Exponential cool-down: energetic for the first seconds, then a faint residual motion.
    this.alpha = floor + (this.alpha - floor) * Math.exp(-dt * 0.55);
    const sim = this.simMaterial.uniforms;
    sim.uPos!.value = this.rtA.texture;
    sim.uAlpha!.value = this.alpha;
    sim.uFrame!.value = this.frame++;
    this.renderer.setRenderTarget(this.rtB);
    this.renderer.render(this.simScene, this.simCamera);
    this.renderer.setRenderTarget(null);
    [this.rtA, this.rtB] = [this.rtB, this.rtA];
    for (const m of [this.pointMaterial, this.dustMaterial, this.edgeMaterial]) m.uniforms.uPos!.value = this.rtA.texture;
    this.pointMaterial.uniforms.uTime!.value = time;
    this.dustMaterial.uniforms.uTime!.value = time;

    // Picking needs CPU positions. Read often while the layout is moving, rarely once it settles.
    const every = this.alpha > 0.05 ? 6 : 45;
    if (this.readbackPending || this.frame % every === 0) {
      this.renderer.readRenderTargetPixels(this.rtA, 0, 0, this.size, this.size, this.positions);
      this.readbackPending = false;
      this.readbacks++;
    }
  }

  dispose(): void {
    this.rtA.dispose();
    this.rtB.dispose();
    for (const t of this.textures) t.dispose();
    this.points.geometry.dispose();
    this.dust.geometry.dispose();
    this.edges.geometry.dispose();
    this.simMaterial.dispose();
    this.pointMaterial.dispose();
    this.dustMaterial.dispose();
    this.edgeMaterial.dispose();
  }
}
