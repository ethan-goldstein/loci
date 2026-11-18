import { CARDS, RESULTS_URL } from './tour-logic.ts';
import type { TourFacts } from './tour-logic.ts';

/**
 * The guided tour: a column of short cards that scrolls over the map. Each card coming into view
 * changes what the map shows, using the same calls a visitor's own clicks would make.
 *
 * Scrolling drives it, and nothing depends on scrolling: the next and back buttons, the step
 * dots and the arrow keys do the same thing, Esc and the exit button always leave, and people
 * who ask for reduced motion get each state without the animation.
 */

export interface TourHost {
  facts: TourFacts;
  /** Called on entering and leaving: clear anything a step lit and put the camera home. */
  reset(): void;
  /** Called when the tour opens (true) and closes (false), so the map can make room for the cards. */
  shown?(open: boolean): void;
  /** One function per card, in the order of CARDS. Each may add what it learned to `facts`. */
  steps: (() => void | Promise<void>)[];
}

export interface Tour {
  open(step?: number): void;
  close(): void;
  readonly active: boolean;
}

const el = <K extends keyof HTMLElementTagNameMap>(tag: K, className?: string, text?: string): HTMLElementTagNameMap[K] => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
};

export function setupTour(host: TourHost, canvas: HTMLCanvasElement, reducedMotion: boolean): Tour {
  const root = document.getElementById('tour') as HTMLElement;
  const scroller = document.getElementById('tour-scroll') as HTMLElement;
  const dots = document.getElementById('tour-dots') as HTMLElement;
  const exit = document.getElementById('tour-exit') as HTMLButtonElement;
  const count = CARDS.length;

  let active = false;
  let current = -1;
  let token = 0;
  let settle = 0;
  let returnTo: HTMLElement | null = null;
  /** While a button or key is moving the view to a step, scroll positions on the way are not choices. */
  let heading: { step: number; until: number } | null = null;

  // ---- build the cards ---------------------------------------------------------------------
  const sections: HTMLElement[] = [];
  const bodies: HTMLElement[] = [];
  const dotButtons: HTMLButtonElement[] = [];
  CARDS.forEach((card, i) => {
    const section = el('section', 'tour-step');
    section.dataset.step = String(i);
    const box = el('div', 'tour-card');
    box.append(el('p', 'tour-count', `${i + 1} of ${count}`), el('h2', undefined, card.title));
    const body = el('div', 'tour-body');
    bodies.push(body);
    box.append(body);
    if (i === count - 1) {
      const link = el('a', undefined, 'Read the measured results');
      link.href = RESULTS_URL;
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
      const p = el('p', 'tour-link');
      p.append(link);
      box.append(p);
    }
    const nav = el('div', 'tour-nav');
    const back = el('button', 'ghost', 'back');
    back.type = 'button';
    back.disabled = i === 0;
    back.addEventListener('click', () => go(i - 1));
    const next = el('button', i === count - 1 ? 'primary' : 'ghost', i === count - 1 ? 'Explore the map' : 'next');
    next.type = 'button';
    next.addEventListener('click', () => (i === count - 1 ? close() : go(i + 1)));
    nav.append(back, next);
    box.append(nav);
    section.append(box);
    scroller.append(section);
    sections.push(section);

    const dot = el('button', 'tour-dot');
    dot.type = 'button';
    dot.setAttribute('aria-label', `Step ${i + 1}: ${card.title}`);
    dot.addEventListener('click', () => go(i));
    dots.append(dot);
    dotButtons.push(dot);
  });

  const write = (i: number) => {
    const body = bodies[i]!;
    body.replaceChildren(...CARDS[i]!.body(host.facts).map((line) => el('p', undefined, line)));
  };

  /** Makes step `i` the live one: runs its action, then fills in what the action found. */
  const activate = (i: number) => {
    if (!active || i === current) return;
    current = i;
    const mine = ++token;
    canvas.dataset.tourStep = String(i);
    dotButtons.forEach((d, j) => d.setAttribute('aria-current', String(j === i)));
    write(i);
    window.clearTimeout(settle);
    // A fast scroll passes several cards; only the one it rests on does its work.
    settle = window.setTimeout(
      () => {
        if (mine !== token) return;
        void (async () => {
          try {
            await host.steps[i]!();
          } catch {
            /* a step that fails leaves its card with the text it already has */
          }
          if (mine === token) {
            write(i);
            canvas.dataset.tourDone = String(i);
          }
        })();
      },
      reducedMotion ? 0 : 120,
    );
  };

  const go = (i: number) => {
    const target = Math.min(count - 1, Math.max(0, i));
    heading = { step: target, until: performance.now() + 1200 };
    sections[target]!.scrollIntoView({ behavior: reducedMotion ? 'auto' : 'smooth', block: 'start' });
    // Scrolling will trigger the observer, but do not wait for it: a keyboard user should not have to.
    activate(target);
  };

  // The card crossing the middle of the view is the live one.
  const observer = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (!e.isIntersecting) continue;
        const step = Number((e.target as HTMLElement).dataset.step);
        if (heading && performance.now() < heading.until) {
          // On the way to a step someone asked for: cards passed on the way do not take over.
          if (step !== heading.step) continue;
          heading = null;
        }
        activate(step);
      }
    },
    { root: scroller, rootMargin: '-45% 0px -45% 0px' },
  );
  for (const s of sections) observer.observe(s);
  // Once the view has come to rest, scrolling by hand chooses the step again.
  scroller.addEventListener('scrollend', () => {
    heading = null;
  });

  const onKey = (e: KeyboardEvent) => {
    if (!active) return;
    if (e.key === 'Escape') {
      e.stopPropagation();
      close();
    } else if (e.key === 'ArrowDown' || e.key === 'ArrowRight' || e.key === 'PageDown') {
      e.preventDefault();
      go(current + 1);
    } else if (e.key === 'ArrowUp' || e.key === 'ArrowLeft' || e.key === 'PageUp') {
      e.preventDefault();
      go(current - 1);
    }
  };
  // Capture, so Esc leaves the tour before the viewer's own Esc handler clears anything else.
  window.addEventListener('keydown', onKey, true);
  exit.addEventListener('click', () => close());

  function open(step = 0): void {
    if (active) return;
    returnTo = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    active = true;
    current = -1;
    host.reset();
    document.body.classList.add('touring');
    root.hidden = false;
    host.shown?.(true);
    canvas.dataset.tour = '1';
    const start = Math.min(count - 1, Math.max(0, step));
    heading = { step: start, until: performance.now() + 400 };
    scroller.scrollTop = sections[start]!.offsetTop;
    activate(start);
    exit.focus();
  }

  function close(): void {
    if (!active) return;
    active = false;
    token++;
    window.clearTimeout(settle);
    root.hidden = true;
    document.body.classList.remove('touring');
    host.shown?.(false);
    canvas.dataset.tour = '0';
    delete canvas.dataset.tourStep;
    delete canvas.dataset.tourDone;
    host.reset();
    returnTo?.focus();
  }

  return {
    open,
    close,
    get active() {
      return active;
    },
  };
}
