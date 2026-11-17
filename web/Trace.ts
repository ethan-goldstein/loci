import * as THREE from 'three';
import type { TraceStep } from './api.ts';
import type { Cloud } from './Cloud.ts';

/**
 * Draws a search as it actually happened inside the index.
 *
 * The server returns the HNSW trace: every node the search evaluated, the node it was expanding at
 * the time, and the layer. Each step becomes a line segment from parent to node. The segments are
 * revealed in order, so you watch the search drop through the sparse upper layers and then fan out
 * across layer 0 toward the answer. Nothing here is decorative: every line is an edge the search
 * really followed.
 *
 * Like the cloud, segments read positions from the simulation texture, so the path stays attached
 * to the particles while they move.
 */

const VERT = /* glsl */ `
  attribute vec2 ref; attribute float aOrder; attribute float aLayer;
  uniform sampler2D uPos; uniform float uProgress; uniform float uFade;
  varying float vAlpha; varying float vLayer;
  void main() {
    gl_Position = projectionMatrix * modelViewMatrix * vec4(texture2D(uPos, ref).xyz, 1.0);
    float age = uProgress - aOrder;           // how long ago this step was drawn, in path units
    float shown = step(0.0, age);
    float head = exp(-age * 9.0);              // bright where the search currently is
    vAlpha = shown * (0.34 + 0.66 * head) * uFade;
    vLayer = aLayer;
  }
`;

const FRAG = /* glsl */ `
  precision highp float;
  varying float vAlpha; varying float vLayer;
  void main() {
    // Upper layers (the coarse descent) are warm; layer 0 (the fine search) is cool white.
    vec3 col = mix(vec3(0.72, 0.9, 1.0), vec3(1.0, 0.78, 0.36), clamp(vLayer, 0.0, 1.0));
    gl_FragColor = vec4(col * vAlpha, vAlpha);
  }
`;

const MARK_VERT = /* glsl */ `
  attribute vec2 ref; attribute float aRankInList;
  uniform sampler2D uPos; uniform float uPixelRatio; uniform float uScale; uniform float uTime; uniform float uShow;
  varying float vAlpha; varying float vPhase;
  void main() {
    vec4 mv = modelViewMatrix * vec4(texture2D(uPos, ref).xyz, 1.0);
    gl_Position = projectionMatrix * mv;
    float pulse = 0.5 + 0.5 * sin(uTime * 3.0 - aRankInList * 0.55);
    gl_PointSize = clamp((16.0 + 6.0 * pulse) * uPixelRatio * uScale * 3.0 / max(0.3, -mv.z) * (1.2 - 0.03 * aRankInList), 8.0, 34.0);
    vAlpha = uShow * (1.0 - 0.035 * aRankInList);
    vPhase = pulse;
  }
`;

const MARK_FRAG = /* glsl */ `
  precision highp float;
  varying float vAlpha; varying float vPhase;
  void main() {
    float d = length(gl_PointCoord - 0.5) * 2.0;
    float ring = smoothstep(0.16, 0.0, abs(d - (0.62 + 0.18 * vPhase)));
    float a = ring * vAlpha;
    if (a < 0.01) discard;
    gl_FragColor = vec4(vec3(1.0, 0.95, 0.82) * a, a);
  }
`;

export class Trace {
  readonly group = new THREE.Group();
  /** Number of nodes the last search evaluated, and how many layers it crossed. */
  visited = 0;
  layers = 0;

  private readonly lines: THREE.LineSegments;
  private readonly marks: THREE.Points;
  private readonly lineMaterial: THREE.ShaderMaterial;
  private readonly markMaterial: THREE.ShaderMaterial;
  private readonly cloud: Cloud;
  private progress = 0;
  private duration = 1.6;
  private active = false;

  constructor(cloud: Cloud, pixelRatio: number) {
    this.cloud = cloud;
    this.lineMaterial = new THREE.ShaderMaterial({
      uniforms: { uPos: { value: cloud.positionTexture }, uProgress: { value: 0 }, uFade: { value: 0 } },
      vertexShader: VERT,
      fragmentShader: FRAG,
      transparent: true,
      depthTest: false,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    this.lines = new THREE.LineSegments(new THREE.BufferGeometry(), this.lineMaterial);
    this.lines.frustumCulled = false;
    this.lines.renderOrder = 3;

    this.markMaterial = new THREE.ShaderMaterial({
      uniforms: { uPos: { value: cloud.positionTexture }, uPixelRatio: { value: pixelRatio }, uScale: { value: 1 }, uTime: { value: 0 }, uShow: { value: 0 } },
      vertexShader: MARK_VERT,
      fragmentShader: MARK_FRAG,
      transparent: true,
      depthTest: false,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    this.marks = new THREE.Points(new THREE.BufferGeometry(), this.markMaterial);
    this.marks.frustumCulled = false;
    this.marks.renderOrder = 4;
    this.group.add(this.lines, this.marks);
  }

  /** Starts animating a new search. `hits` are particle indices, best first. */
  play(trace: readonly TraceStep[], hits: readonly number[], instant: boolean): void {
    const steps = trace.filter((s) => s[2] >= 0 && s[1] < this.cloud.n && s[2] < this.cloud.n);
    const maxLayer = trace.reduce((m, s) => Math.max(m, s[0]), 0);
    this.visited = trace.length;
    this.layers = new Set(trace.map((s) => s[0])).size;

    const ref = new Float32Array(steps.length * 4);
    const order = new Float32Array(steps.length * 2);
    const layer = new Float32Array(steps.length * 2);
    steps.forEach(([l, node, parent], i) => {
      ref.set(this.cloud.refOf(parent), i * 4);
      ref.set(this.cloud.refOf(node), i * 4 + 2);
      const t = steps.length > 1 ? i / (steps.length - 1) : 0;
      // The far end of a segment appears a touch after its start, so lines seem to grow.
      order[i * 2] = t;
      order[i * 2 + 1] = t + 0.5 / Math.max(steps.length, 1);
      layer[i * 2] = layer[i * 2 + 1] = maxLayer > 0 ? Math.min(1, l / maxLayer) : 0;
    });
    this.replace(this.lines, { ref: [ref, 2], aOrder: [order, 1], aLayer: [layer, 1] }, steps.length * 2);

    const mRef = new Float32Array(hits.length * 2);
    const mRank = new Float32Array(hits.length);
    hits.forEach((p, i) => {
      mRef.set(this.cloud.refOf(p), i * 2);
      mRank[i] = i;
    });
    this.replace(this.marks, { ref: [mRef, 2], aRankInList: [mRank, 1] }, hits.length);

    this.duration = Math.min(2.4, 0.6 + steps.length * 0.006);
    this.progress = instant ? 1.2 : 0;
    this.active = true;
    this.lineMaterial.uniforms.uFade!.value = 1;
  }

  private replace(obj: THREE.LineSegments | THREE.Points, attrs: Record<string, [Float32Array, number]>, vertices: number): void {
    obj.geometry.dispose();
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(vertices * 3), 3));
    for (const [name, [data, size]] of Object.entries(attrs)) geo.setAttribute(name, new THREE.BufferAttribute(data, size));
    obj.geometry = geo;
  }

  clear(): void {
    this.active = false;
    this.progress = 0;
    this.lineMaterial.uniforms.uFade!.value = 0;
    this.markMaterial.uniforms.uShow!.value = 0;
    this.visited = 0;
    this.layers = 0;
  }

  resize(pixelRatio: number, height: number): void {
    this.markMaterial.uniforms.uPixelRatio!.value = pixelRatio;
    this.markMaterial.uniforms.uScale!.value = height / 760;
  }

  update(dt: number, time: number): void {
    this.lineMaterial.uniforms.uPos!.value = this.cloud.positionTexture;
    this.markMaterial.uniforms.uPos!.value = this.cloud.positionTexture;
    this.markMaterial.uniforms.uTime!.value = time;
    if (!this.active) return;
    this.progress = Math.min(1.25, this.progress + dt / this.duration);
    this.lineMaterial.uniforms.uProgress!.value = this.progress;
    // Hit rings fade in as the path arrives.
    this.markMaterial.uniforms.uShow!.value = Math.min(1, Math.max(0, (this.progress - 0.75) * 4));
  }

  /** True once the path animation has fully played. */
  get done(): boolean {
    return !this.active || this.progress >= 1.2;
  }

  dispose(): void {
    this.lines.geometry.dispose();
    this.marks.geometry.dispose();
    this.lineMaterial.dispose();
    this.markMaterial.dispose();
  }
}
