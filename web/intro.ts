/**
 * The public demo's welcome panel: what this is, that the data is invented, and three things to
 * try. Shown on a first visit and reachable afterwards from the "about" button. The local viewer
 * never shows it; there the data is your own and needs no introduction.
 */

export interface IntroActions {
  search: (query: string) => void;
  repeats: () => void;
  play: () => void;
  tour: () => void;
}

const SEEN = 'loci.intro.seen';

function seen(): boolean {
  try {
    return localStorage.getItem(SEEN) === '1';
  } catch {
    return false; // storage blocked: just show it
  }
}

/** `quiet` skips the first-visit welcome, for when the address already asks for a tour or a search. */
export function setupIntro(actions: IntroActions, quiet = false): void {
  const box = document.getElementById('intro');
  const open = document.getElementById('about-open');
  const close = document.getElementById('intro-close');
  if (!box || !open || !close) return;

  let returnTo: HTMLElement | null = null;
  const show = () => {
    returnTo = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    box.hidden = false;
    close.focus();
  };
  const hide = () => {
    if (box.hidden) return;
    box.hidden = true;
    try {
      localStorage.setItem(SEEN, '1');
    } catch {
      /* not remembered; harmless */
    }
    returnTo?.focus();
  };

  open.hidden = false;
  open.addEventListener('click', show);
  close.addEventListener('click', hide);
  // A click on the dimmed area around the card closes it; a click inside does not.
  box.addEventListener('click', (e) => {
    if (e.target === box) hide();
  });
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') hide();
  });

  for (const b of box.querySelectorAll<HTMLButtonElement>('button[data-try]')) {
    b.addEventListener('click', () => {
      hide();
      if (b.dataset.try === 'search') actions.search(b.dataset.query ?? '');
      else if (b.dataset.try === 'repeats') actions.repeats();
      else if (b.dataset.try === 'play') actions.play();
      else if (b.dataset.try === 'tour') actions.tour();
    });
  }

  if (!quiet && !seen()) show();
}
