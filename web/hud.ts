import type { ChunkResponse, Manifest, Mode, ProfileView, RepeatItem, RepeatsView, SearchResponse } from './api.ts';
import { renderProfile } from './Profile.ts';
import { SLOTS, SPEAKER, TIME_GRADIENT, projectColor } from './palette.ts';
import type { ColorMode } from './palette.ts';

/**
 * The DOM side of the viewer. Everything that came from a conversation is written with
 * `textContent`, never as HTML: recalled text is untrusted input and must not become markup.
 */

export interface HudHandlers {
  onSearch: (query: string, mode: Mode) => void;
  onClear: () => void;
  onProject: (index: number | null) => void;
  onReveal: (fraction: number) => void;
  onPlay: () => void;
  onSelect: (particle: number) => void;
  onCloseDetail: () => void;
  onColorMode: (mode: ColorMode) => void;
  onToggle: (name: ToggleName, on: boolean) => void;
  /** The repeats panel was opened (true) or closed (false). */
  onRepeats: (open: boolean) => void;
  /** One repeated thing was chosen: light up where it was said. */
  onRepeat: (item: RepeatItem) => void;
  /** The profile panel was opened (true) or closed (false). */
  onProfile: (open: boolean) => void;
}

export type ToggleName = 'labels' | 'dust' | 'glow' | 'depth';
const TOGGLES: ToggleName[] = ['labels', 'dust', 'glow', 'depth'];
const COLOR_MODES: ColorMode[] = ['project', 'time', 'speaker'];
const PREFS_KEY = 'loci.view';

interface Prefs {
  color: ColorMode;
  toggles: Record<ToggleName, boolean>;
}

/** View preferences are a per-viewer convenience; the page works the same if storage is unavailable. */
function loadPrefs(): Prefs {
  const prefs: Prefs = { color: 'project', toggles: { labels: true, dust: true, glow: true, depth: true } };
  try {
    const raw = JSON.parse(localStorage.getItem(PREFS_KEY) ?? 'null') as Partial<Prefs> | null;
    if (raw && COLOR_MODES.includes(raw.color as ColorMode)) prefs.color = raw.color as ColorMode;
    for (const t of TOGGLES) if (typeof raw?.toggles?.[t] === 'boolean') prefs.toggles[t] = raw.toggles[t];
  } catch {
    /* private window or blocked storage: use the defaults */
  }
  return prefs;
}

function savePrefs(prefs: Prefs): void {
  try {
    localStorage.setItem(PREFS_KEY, JSON.stringify(prefs));
  } catch {
    /* nothing to do */
  }
}

function el<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) throw new Error(`missing #${id}`);
  return node as T;
}

function make<K extends keyof HTMLElementTagNameMap>(tag: K, className?: string, text?: string): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

const day = (iso: string) => iso.slice(0, 10);

export class Hud {
  mode: Mode = 'hybrid';
  private readonly manifest: Manifest;
  private readonly handlers: HudHandlers;
  private readonly input = el<HTMLInputElement>('q');
  private readonly results = el<HTMLElement>('results');
  private readonly resultsList = el<HTMLOListElement>('results-list');
  private readonly resultsStats = el<HTMLElement>('results-stats');
  private readonly detail = el<HTMLElement>('detail');
  private readonly tooltip = el<HTMLElement>('tooltip');
  private readonly scrub = el<HTMLInputElement>('scrub');
  private readonly scrubDate = el<HTMLOutputElement>('scrub-date');
  private readonly playButton = el<HTMLButtonElement>('play');
  private readonly legend = el<HTMLElement>('legend');
  private readonly projectIndex = new Map<string, number>();
  private focused: number | null = null;
  private readonly prefs: Prefs = loadPrefs();

  constructor(manifest: Manifest, handlers: HudHandlers) {
    this.manifest = manifest;
    this.handlers = handlers;
    for (const p of manifest.projects) this.projectIndex.set(p.name, p.index);

    const span = manifest.n ? `${new Date(manifest.from).toISOString().slice(0, 10)} to ${new Date(manifest.to).toISOString().slice(0, 10)}` : 'empty';
    el('summary').textContent = `${manifest.n.toLocaleString('en-US')} chunks · ${manifest.projects.length} projects · ${manifest.sessions} sessions · ${span}`;
    el('demo-badge').hidden = !manifest.demo;

    el<HTMLFormElement>('search-form').addEventListener('submit', (e) => {
      e.preventDefault();
      this.submit();
    });
    const searchModes = [...document.querySelectorAll<HTMLButtonElement>('#search-modes button')];
    for (const b of searchModes) {
      b.addEventListener('click', () => {
        this.mode = (b.dataset.mode as Mode) ?? 'hybrid';
        for (const other of searchModes) other.setAttribute('aria-checked', String(other === b));
        if (this.input.value.trim()) this.submit();
      });
    }
    const colorModes = [...document.querySelectorAll<HTMLButtonElement>('#color-modes button')];
    for (const b of colorModes) {
      b.addEventListener('click', () => {
        const mode = b.dataset.color as ColorMode;
        if (!COLOR_MODES.includes(mode)) return;
        this.prefs.color = mode;
        savePrefs(this.prefs);
        this.applyColorMode();
        handlers.onColorMode(mode);
      });
    }
    for (const name of TOGGLES) {
      const b = el<HTMLButtonElement>(`toggle-${name}`);
      b.setAttribute('aria-pressed', String(this.prefs.toggles[name]));
      b.addEventListener('click', () => {
        const on = !this.prefs.toggles[name];
        this.prefs.toggles[name] = on;
        b.setAttribute('aria-pressed', String(on));
        savePrefs(this.prefs);
        handlers.onToggle(name, on);
      });
    }
    el('clear').addEventListener('click', () => {
      this.input.value = '';
      handlers.onClear();
    });
    el('detail-close').addEventListener('click', () => handlers.onCloseDetail());
    el('open-repeats').addEventListener('click', () => handlers.onRepeats(el('repeats').hidden));
    el('repeats-close').addEventListener('click', () => handlers.onRepeats(false));
    el('open-profile').addEventListener('click', () => handlers.onProfile(el('profile').hidden));
    el('profile-close').addEventListener('click', () => handlers.onProfile(false));
    this.scrub.addEventListener('input', () => handlers.onReveal(Number(this.scrub.value) / 1000));
    this.playButton.addEventListener('click', () => handlers.onPlay());
    this.buildLegend();
    this.applyColorMode();
  }

  /** The saved view settings, for the caller to apply once at start-up. */
  get view(): Readonly<Prefs> {
    return this.prefs;
  }

  /** Updates the legend swatches and the small key under the colour switch. */
  private applyColorMode(): void {
    const mode = this.prefs.color;
    for (const b of document.querySelectorAll<HTMLButtonElement>('#color-modes button')) b.setAttribute('aria-checked', String(b.dataset.color === mode));
    // Project swatches only mean something while colour encodes the project.
    [...this.legend.querySelectorAll<HTMLElement>('.dot')].forEach((dot, i) => {
      dot.style.background = mode === 'project' ? projectColor(this.manifest.projects[i]!.index) : 'transparent';
      dot.style.outline = mode === 'project' ? 'none' : '1px solid var(--line-strong)';
    });
    const key = el('mode-key');
    key.replaceChildren();
    const day = (ms: number) => new Date(ms).toISOString().slice(0, 10);
    if (mode === 'time') {
      const bar = make('div', 'ramp');
      bar.style.background = TIME_GRADIENT;
      const ends = make('div', 'ramp-ends');
      ends.append(make('span', undefined, day(this.manifest.from)), make('span', undefined, day(this.manifest.to)));
      key.append(bar, ends);
    } else if (mode === 'speaker') {
      for (const [name, color] of [['you', SPEAKER.human], ['assistant', SPEAKER.assistant]] as const) {
        const row = make('span', 'key-item');
        const dot = make('span', 'dot');
        dot.style.background = color;
        row.append(dot, make('span', undefined, name));
        key.append(row);
      }
    } else {
      const rest = this.manifest.projects.length - SLOTS;
      key.append(make('span', 'muted', rest > 0 ? `the ${SLOTS} largest projects have a colour; ${rest} others share grey. Click a project to isolate it.` : 'one colour per project'));
    }
  }

  private submit(): void {
    const q = this.input.value.trim();
    if (q) this.handlers.onSearch(q.slice(0, 512), this.mode);
    else this.handlers.onClear();
  }

  private buildLegend(): void {
    this.legend.replaceChildren();
    for (const p of this.manifest.projects) {
      const b = make('button');
      b.type = 'button';
      b.setAttribute('aria-pressed', 'false');
      const dot = make('span', 'dot');
      dot.style.background = projectColor(p.index);
      b.append(dot, make('span', 'name', p.name), make('span', 'count', p.count.toLocaleString('en-US')));
      b.title = `Show only ${p.name}`;
      b.addEventListener('click', () => {
        this.focused = this.focused === p.index ? null : p.index;
        for (const [i, child] of [...this.legend.children].entries()) child.setAttribute('aria-pressed', String(this.focused === this.manifest.projects[i]!.index));
        this.handlers.onProject(this.focused);
      });
      this.legend.append(b);
    }
  }

  get focusedProject(): number | null {
    return this.focused;
  }

  /** Types a query into the box and runs it, as if the person had. */
  search(query: string): void {
    this.input.value = query;
    this.submit();
  }

  focusSearch(): void {
    this.input.focus();
    this.input.select();
  }

  get searchHasFocus(): boolean {
    return document.activeElement === this.input;
  }

  private colorFor(project: string): string {
    return projectColor(this.projectIndex.get(project) ?? 0);
  }

  setBusy(text: string): void {
    this.hideRepeats();
    this.hideProfile();
    this.results.hidden = false;
    this.resultsStats.textContent = text;
  }

  setResults(res: SearchResponse, visited: number, layers: number): void {
    this.results.hidden = false;
    const parts = [`${res.hits.length} results`, `${res.ms} ms`];
    if (res.dense === 'hnsw') parts.push(`graph search visited ${visited} nodes over ${layers} layers`);
    else if (res.dense === 'exact') parts.push('exact scan (narrow filter)');
    if (res.folded > 0) parts.push(`${res.folded} near-${res.folded === 1 ? 'copy' : 'copies'} folded`);
    if (res.degraded) parts.push('embedding model unreachable: keywords only');
    this.resultsStats.textContent = parts.join(' · ');
    this.resultsList.replaceChildren();
    res.hits.forEach((h, i) => {
      const li = make('li');
      const b = make('button');
      b.type = 'button';
      const meta = make('div', 'hit-meta');
      const dot = make('span', 'dot');
      dot.style.background = this.colorFor(h.project);
      const sim = h.similarity === null ? '' : ` · ${h.similarity.toFixed(2)}`;
      meta.append(dot, make('span', undefined, `${i + 1} · ${h.project} · ${day(h.date)} · ${h.role}${sim}`));
      b.append(meta, make('div', 'hit-text', h.text));
      b.addEventListener('click', () => this.handlers.onSelect(h.particle));
      li.append(b);
      this.resultsList.append(li);
    });
    if (res.hits.length === 0) this.resultsList.append(make('li', 'muted', 'Nothing matched.'));
  }

  /** Shows the things said again and again, most widespread first. Text is the person's own. */
  /** The "where the time goes" panel: figures and small charts, from counts only. */
  showProfile(view: ProfileView): void {
    this.results.hidden = true; // one right-hand panel at a time
    this.hideRepeats();
    el('profile').hidden = false;
    el('open-profile').setAttribute('aria-pressed', 'true');
    const body = el('profile-body');
    if (!view.available || !view.profile) {
      body.replaceChildren();
      el('profile-stats').textContent = view.reason ?? 'Not available.';
      return;
    }
    const t = view.profile.totals;
    el('profile-stats').textContent = `${t.sessions.toLocaleString('en-US')} sessions · ${t.prompts.toLocaleString('en-US')} prompts · ${t.weeks} weeks · counts only, no text`;
    renderProfile(body, el('profile-tip'), view.profile);
  }

  hideProfile(): void {
    el('profile').hidden = true;
    el('profile-tip').hidden = true;
    el('open-profile').setAttribute('aria-pressed', 'false');
  }

  showRepeats(view: RepeatsView): void {
    this.results.hidden = true; // one right-hand panel at a time
    this.hideProfile();
    const panel = el('repeats');
    const list = el<HTMLOListElement>('repeats-list');
    panel.hidden = false;
    el('open-repeats').setAttribute('aria-pressed', 'true');
    list.replaceChildren();
    if (!view.available || !view.items) {
      el('repeats-stats').textContent = view.reason ?? 'Not available.';
      return;
    }
    el('repeats-stats').textContent = `${view.groups} things said in ${view.minSessions}+ sessions on 2+ days · ${((view.share ?? 0) * 100).toFixed(1)}% of what you typed`;
    const day = (ms: number) => new Date(ms).toISOString().slice(0, 10);
    for (const item of view.items) {
      const li = make('li');
      const b = make('button');
      b.type = 'button';
      const meta = make('div', 'hit-meta');
      meta.append(make('span', `kind kind-${item.kind}`, item.kind), make('span', undefined, `${item.occurrences}× · ${item.sessions} sessions · ${item.projects.length} project${item.projects.length === 1 ? '' : 's'} · ${day(item.first)} to ${day(item.last)}`));
      b.append(meta, make('div', 'hit-text', item.text));
      b.title = item.variants.length ? `Also phrased as: ${item.variants.join(' | ')}` : '';
      b.addEventListener('click', () => {
        for (const other of list.querySelectorAll('button')) other.removeAttribute('aria-current');
        b.setAttribute('aria-current', 'true');
        this.handlers.onRepeat(item);
      });
      li.append(b);
      list.append(li);
    }
    if (view.items.length === 0) list.append(make('li', 'muted', 'Nothing was said in enough separate sessions.'));
  }

  hideRepeats(): void {
    el('repeats').hidden = true;
    el('open-repeats').setAttribute('aria-pressed', 'false');
  }

  get repeatsOpen(): boolean {
    return !el('repeats').hidden;
  }

  setError(message: string): void {
    this.results.hidden = false;
    this.resultsStats.textContent = message;
    this.resultsList.replaceChildren();
  }

  clearResults(): void {
    this.results.hidden = true;
    this.resultsList.replaceChildren();
  }

  showDetail(c: ChunkResponse): void {
    this.detail.hidden = false;
    el('detail-meta').textContent = `${c.project} · ${day(c.date)} · ${c.role} · chunk ${c.chunkId} · index layer ${c.level}`;
    el('detail-session').textContent = c.session ?? 'untitled session';
    el('detail-text').textContent = c.text;
    const list = el<HTMLUListElement>('detail-neighbours');
    list.replaceChildren();
    for (const nb of c.neighbours) {
      const li = make('li');
      const b = make('button', undefined, `${nb.similarity.toFixed(2)} · ${nb.project} · ${nb.text.replace(/\s+/g, ' ').slice(0, 110)}`);
      b.type = 'button';
      b.addEventListener('click', () => this.handlers.onSelect(nb.particle));
      li.append(b);
      list.append(li);
    }
    this.detail.scrollTop = 0;
  }

  hideDetail(): void {
    this.detail.hidden = true;
  }

  showTooltip(x: number, y: number, text: string): void {
    this.tooltip.hidden = false;
    this.tooltip.textContent = text;
    this.tooltip.style.left = `${Math.min(window.innerWidth - 240, x + 14)}px`;
    this.tooltip.style.top = `${Math.max(8, y - 28)}px`;
  }

  hideTooltip(): void {
    this.tooltip.hidden = true;
  }

  setScrub(fraction: number, label: string, playing: boolean): void {
    const v = String(Math.round(fraction * 1000));
    if (this.scrub.value !== v) this.scrub.value = v;
    this.scrubDate.textContent = label;
    this.playButton.textContent = playing ? 'pause' : 'play';
  }

  fatal(message: string): void {
    const box = el('fatal');
    box.hidden = false;
    box.textContent = message;
  }
}
