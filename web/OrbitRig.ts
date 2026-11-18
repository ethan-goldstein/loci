import * as THREE from 'three';

/**
 * A small orbit camera: spherical coordinates around a target, with damping. It accepts three
 * kinds of input and treats them the same way: pointer drags, the scroll wheel, and continuous
 * rates (radians per second) that the gesture controller sets each frame.
 */
export class OrbitRig {
  readonly camera: THREE.PerspectiveCamera;
  readonly target = new THREE.Vector3();
  theta = 0.6;
  phi = 1.25;
  distance = 4.2;
  /** Continuous input, e.g. from hand gestures: yaw, pitch (rad/s) and zoom (fraction/s). */
  rates = { yaw: 0, pitch: 0, zoom: 0 };
  autoRotate = true;

  private vTheta = 0;
  private vPhi = 0;
  private goal: { target: THREE.Vector3; distance: number } | null = null;
  private idle = 0;
  private dragging = false;
  private lastX = 0;
  private lastY = 0;
  private readonly minDistance = 0.25;
  private readonly maxDistance = 14;
  private readonly cleanup: (() => void)[] = [];

  constructor(camera: THREE.PerspectiveCamera, el: HTMLElement) {
    this.camera = camera;
    const on = <K extends keyof HTMLElementEventMap>(type: K, fn: (e: HTMLElementEventMap[K]) => void, opts?: AddEventListenerOptions) => {
      el.addEventListener(type, fn, opts);
      this.cleanup.push(() => el.removeEventListener(type, fn, opts));
    };
    on('pointerdown', (e) => {
      if (e.button !== 0) return;
      this.dragging = true;
      this.lastX = e.clientX;
      this.lastY = e.clientY;
      el.setPointerCapture(e.pointerId);
      this.touch();
    });
    on('pointermove', (e) => {
      if (!this.dragging) return;
      const dx = e.clientX - this.lastX;
      const dy = e.clientY - this.lastY;
      this.lastX = e.clientX;
      this.lastY = e.clientY;
      this.vTheta = -dx * 0.005;
      this.vPhi = -dy * 0.005;
      this.theta += this.vTheta;
      this.phi += this.vPhi;
      this.touch();
    });
    const end = (e: PointerEvent) => {
      this.dragging = false;
      if (el.hasPointerCapture(e.pointerId)) el.releasePointerCapture(e.pointerId);
    };
    on('pointerup', end);
    on('pointercancel', end);
    on(
      'wheel',
      (e) => {
        e.preventDefault();
        this.distance *= Math.exp(e.deltaY * 0.0012);
        this.goal = null;
        this.touch();
      },
      { passive: false },
    );
  }

  /** Any direct input pauses the idle rotation for a while. */
  touch(): void {
    this.idle = 0;
    this.goal = null;
  }

  private homeView: { target: THREE.Vector3; distance: number } | null = null;

  /** Remembers the view that frames the whole field. */
  setHome(point: THREE.Vector3, distance: number): void {
    this.homeView = { target: point.clone(), distance };
  }

  /** Eases back to the view that frames the whole field, if one has been set. */
  home(): void {
    if (this.homeView) this.focus(this.homeView.target, this.homeView.distance);
  }

  /** Eases the camera to look at `point` from `distance` away. */
  focus(point: THREE.Vector3, distance: number): void {
    this.goal = { target: point.clone(), distance };
    this.idle = 0;
  }

  get isDragging(): boolean {
    return this.dragging;
  }

  update(dt: number, reducedMotion: boolean): void {
    const active = this.rates.yaw !== 0 || this.rates.pitch !== 0 || this.rates.zoom !== 0;
    if (active) {
      this.theta += this.rates.yaw * dt;
      this.phi += this.rates.pitch * dt;
      this.distance *= Math.exp(-this.rates.zoom * dt);
      this.idle = 0;
      this.goal = null;
    }
    if (!this.dragging) {
      // Inertia after a drag.
      this.theta += this.vTheta;
      this.phi += this.vPhi;
      const decay = Math.exp(-dt * 6);
      this.vTheta *= decay;
      this.vPhi *= decay;
    }
    if (this.goal) {
      const k = 1 - Math.exp(-dt * 3.2);
      this.target.lerp(this.goal.target, k);
      this.distance += (this.goal.distance - this.distance) * k;
      if (this.target.distanceTo(this.goal.target) < 1e-3 && Math.abs(this.distance - this.goal.distance) < 1e-3) this.goal = null;
    }
    this.idle += dt;
    if (this.autoRotate && !reducedMotion && !this.dragging && !active && this.idle > 2.5) this.theta += dt * 0.045;

    this.phi = Math.min(Math.PI - 0.08, Math.max(0.08, this.phi));
    this.distance = Math.min(this.maxDistance, Math.max(this.minDistance, this.distance));
    const s = Math.sin(this.phi);
    this.camera.position.set(
      this.target.x + this.distance * s * Math.sin(this.theta),
      this.target.y + this.distance * Math.cos(this.phi),
      this.target.z + this.distance * s * Math.cos(this.theta),
    );
    this.camera.lookAt(this.target);
  }

  dispose(): void {
    for (const fn of this.cleanup) fn();
    this.cleanup.length = 0;
  }
}
