import * as THREE from 'three';

/**
 * Floating names for regions of the field.
 *
 * The server splits the settled layout into clusters and names each with its most distinctive
 * words. Here each name is pinned to the live centre of its cluster's visible particles and
 * drawn as ordinary DOM text over the canvas, so it stays crisp and selectable-free.
 *
 * Label words come from conversation text, so they are written with `textContent`.
 */

export interface ClusterInfo {
  index: number;
  size: number;
  words: string[];
}

interface Entry {
  info: ClusterInfo;
  el: HTMLButtonElement;
  centre: THREE.Vector3;
  /** Visible members at the last recount. */
  visible: number;
  /** Largest distance from the centre, for framing when clicked. */
  radius: number;
}

export class Labels {
  private readonly entries: Entry[] = [];
  private readonly root: HTMLElement;
  private readonly clusterOf: Uint8Array;
  private readonly projected = new THREE.Vector3();
  private on = true;

  constructor(root: HTMLElement, clusters: readonly ClusterInfo[], clusterOf: Uint8Array, onPick: (centre: THREE.Vector3, radius: number) => void) {
    this.root = root;
    this.clusterOf = clusterOf;
    root.replaceChildren();
    // Largest first: when two labels would collide, the bigger cluster keeps its place.
    for (const info of [...clusters].sort((a, b) => b.size - a.size)) {
      if (info.words.length === 0) continue; // nothing distinctive to say about this region
      const el = document.createElement('button');
      el.type = 'button';
      el.className = 'cluster-label';
      el.textContent = info.words.join(' · ');
      el.title = `${info.size.toLocaleString('en-US')} chunks. Click to fly here.`;
      const entry: Entry = { info, el, centre: new THREE.Vector3(), visible: 0, radius: 0.2 };
      el.addEventListener('click', () => onPick(entry.centre.clone(), entry.radius));
      root.append(el);
      this.entries.push(entry);
    }
  }

  get count(): number {
    return this.entries.length;
  }

  set visible(on: boolean) {
    this.on = on;
    this.root.hidden = !on;
  }

  get visible(): boolean {
    return this.on;
  }

  /**
   * Recomputes each cluster's centre from the particles that are currently shown.
   * `positions` is the GPU readback (stride 4); `limit` is the last revealed particle index;
   * `focus` restricts to one project (or null).
   */
  recount(positions: Float32Array, limit: number, focus: number | null, meta: Uint32Array): void {
    if (!this.on) return;
    const byIndex = new Map<number, Entry>();
    for (const e of this.entries) {
      e.centre.set(0, 0, 0);
      e.visible = 0;
      e.radius = 0;
      byIndex.set(e.info.index, e);
    }
    for (let i = 0; i <= limit; i++) {
      if (focus !== null && meta[i * 3 + 1] !== focus) continue;
      const e = byIndex.get(this.clusterOf[i]!);
      if (!e) continue;
      e.centre.x += positions[i * 4]!;
      e.centre.y += positions[i * 4 + 1]!;
      e.centre.z += positions[i * 4 + 2]!;
      e.visible++;
    }
    for (const e of this.entries) if (e.visible > 0) e.centre.multiplyScalar(1 / e.visible);
    for (let i = 0; i <= limit; i++) {
      const e = byIndex.get(this.clusterOf[i]!);
      if (!e || e.visible === 0) continue;
      const d = Math.hypot(positions[i * 4]! - e.centre.x, positions[i * 4 + 1]! - e.centre.y, positions[i * 4 + 2]! - e.centre.z);
      if (d > e.radius) e.radius = d;
    }
  }

  /** Places the labels for this frame. Cheap: a few dozen projections and style writes. */
  place(camera: THREE.PerspectiveCamera, width: number, height: number, cameraDistance: number): void {
    if (!this.on) return;
    const taken: { x: number; y: number }[] = [];
    for (const e of this.entries) {
      let show = e.visible >= 8;
      let x = 0;
      let y = 0;
      let depth = 0;
      if (show) {
        this.projected.copy(e.centre).project(camera);
        show = this.projected.z > -1 && this.projected.z < 1;
        x = (this.projected.x * 0.5 + 0.5) * width;
        y = (1 - (this.projected.y * 0.5 + 0.5)) * height;
        depth = camera.position.distanceTo(e.centre);
        // Keep labels clear of the edges and of each other.
        show = show && x > 40 && x < width - 40 && y > 126 && y < height - 150 && !taken.some((t) => Math.abs(t.x - x) < 130 && Math.abs(t.y - y) < 22);
      }
      if (!show) {
        if (e.el.style.opacity !== '0') {
          e.el.style.opacity = '0';
          e.el.style.pointerEvents = 'none';
        }
        continue;
      }
      taken.push({ x, y });
      // Nearer regions read stronger; regions still filling in under the time slider read fainter.
      const near = Math.min(1, Math.max(0.35, cameraDistance / Math.max(0.3, depth)));
      const filled = Math.min(1, e.visible / Math.max(8, e.info.size * 0.25));
      e.el.style.opacity = (0.3 + 0.6 * near * filled).toFixed(2);
      e.el.style.pointerEvents = 'auto';
      e.el.style.transform = `translate(${x.toFixed(1)}px, ${y.toFixed(1)}px) translate(-50%, -50%)`;
    }
  }
}
