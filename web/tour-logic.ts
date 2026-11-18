/**
 * The parts of the guided tour that need no page: what the address bar asks for, and the words on
 * each card. Kept apart from tour.ts so they can be tested without a browser.
 */

export const RESULTS_URL = 'https://github.com/ethan-goldstein/loci/blob/main/RESULTS.md';

/**
 * The measured results quoted on the last card. They are the author's numbers, not the demo's,
 * and each one must appear in RESULTS.md as written here: a unit test checks that, so the card
 * cannot drift from the published file.
 */
export const FIGURES = {
  /** Recall@10 of the graph against an exact scan, at the setting quoted in the README. */
  recall: '0.9814',
  /** How much faster the WebAssembly SIMD inner loop makes that search. */
  kernel: '6.38x',
  /** Project-precision@10 of hybrid retrieval, and of the recency baseline that beat it. */
  hybrid: '0.368',
  recency: '0.496',
  /** Things said in three or more separate sessions. */
  habits: '79',
} as const;

export interface HashRequest {
  /** Open the tour, at this step (0 is the first). Null when the address does not ask for it. */
  tour: number | null;
  /** Run this search on load. Null when the address does not ask for one. */
  query: string | null;
}

/** Reads `#tour`, `#tour=3` and `#q=some+words` from a location hash. Anything else is ignored. */
export function parseHash(hash: string, steps: number): HashRequest {
  const out: HashRequest = { tour: null, query: null };
  const body = hash.startsWith('#') ? hash.slice(1) : hash;
  if (!body) return out;
  for (const part of body.split('&')) {
    const eq = part.indexOf('=');
    const key = eq < 0 ? part : part.slice(0, eq);
    const raw = eq < 0 ? '' : part.slice(eq + 1);
    if (key === 'tour') {
      const n = /^\d{1,2}$/.test(raw) ? Number(raw) : 0;
      out.tour = Math.min(Math.max(0, n), Math.max(0, steps - 1));
    } else if (key === 'q') {
      let text: string;
      try {
        text = decodeURIComponent(raw.replace(/\+/g, ' '));
      } catch {
        continue; // a broken escape: not a query
      }
      text = text.trim().slice(0, 512);
      if (text) out.query = text;
    }
  }
  return out;
}

/** What the map knows by the time a card is shown. Every number here is read from the index or a search. */
export interface TourFacts {
  n: number;
  /** Links drawn per point. */
  knn: number;
  /** How many points reach layer 1 or higher, layer 2 or higher, and so on. */
  layers: number[];
  demo: boolean;
  query?: string;
  /** Hops the search made on the upper layers. */
  hops?: number;
  /** Comparisons it made on the bottom layer. */
  compared?: number;
  both?: number;
  meaning?: number;
  keywords?: number;
  folded?: number;
  repeat?: { text: string; sessions: number } | null;
}

export interface Card {
  title: string;
  body: (f: TourFacts) => string[];
}

const n = (x: number | undefined) => (x ?? 0).toLocaleString('en-US');
const plural = (x: number | undefined, one: string, many: string) => ((x ?? 0) === 1 ? one : many);

export const CARDS: Card[] = [
  {
    title: 'Each point is one message',
    body: (f) => [
      `${n(f.n)} messages from ${f.demo ? 'an invented' : 'your'} history of working with a coding assistant. Larger points are prompts someone typed. The rest are replies.`,
      'Where a point sits is not decoration. Messages that mean similar things are close together, because the layout is built from the search index itself.',
    ],
  },
  {
    title: 'The index links neighbours',
    body: (f) => [
      `This message is linked to the ${n(f.knn)} messages most like it. Every message has links like these, and they are the faint lines you see.`,
      'Searching means walking those links: from a message, step to whichever neighbour is closer to what you asked for.',
    ],
  },
  {
    title: 'A few points live higher up',
    body: (f) => {
      const [l1 = 0, l2 = 0, l3 = 0] = f.layers;
      return [
        `Walking link by link from a random start would be slow. So ${n(l1)} of the ${n(f.n)} messages are also placed on a second, sparser layer${l2 > 0 ? `, ${n(l2)} of those on a third` : ''}${l3 > 0 ? `, and ${n(l3)} at the very top` : ''}.`,
        'Up there the links are long. One step can cross the whole map. This layered graph is called HNSW, and here it is written from scratch.',
      ];
    },
  },
  {
    title: 'A search starts at the top',
    body: (f) => [
      `The search is "${f.query ?? ''}". It enters at the top layer and takes ${n(f.hops)} long ${plural(f.hops, 'hop', 'hops')}, each time to the neighbour closest to the query.`,
      'These lines are the real path: every one is a link the search followed.',
    ],
  },
  {
    title: 'Then it looks around carefully',
    body: (f) => [
      `On the bottom layer it compared ${n(f.compared)} messages and kept the closest. The bright points are the answers.`,
      `An exact search would have compared all ${n(f.n)}. The graph finds almost the same answers while reading a small part of the map.`,
    ],
  },
  {
    title: 'Keywords vote too',
    body: (f) => [
      `Meaning is one ranking. A second ranking, BM25, scores the exact words. Of each one's top twelve here, ${n(f.both)} ${plural(f.both, 'was', 'were')} found by both, ${n(f.meaning)} by meaning only and ${n(f.keywords)} by keywords only.`,
      `The two lists are merged by rank, and a result that is a near-copy of one already shown is folded away${(f.folded ?? 0) > 0 ? ` (${n(f.folded)} here)` : ''}.`,
    ],
  },
  {
    title: 'What keeps getting repeated',
    body: (f) =>
      f.repeat
        ? [
            `"${f.repeat.text}" was said in ${n(f.repeat.sessions)} separate sessions. The lit points are every time.`,
            'LOCI compares every sentence you typed with every other, finds the ones you keep repeating, and drafts standing instructions from them so you can say them once.',
          ]
        : ['Nothing has been said in enough separate sessions yet.', 'LOCI compares every sentence you typed with every other and finds the ones you keep repeating.'],
  },
  {
    title: 'What the numbers say',
    body: () => [
      `Measured on the author's real history, not this demo: the graph finds ${(Number(FIGURES.recall) * 100).toFixed(1)}% of the true nearest neighbours, and a hand-assembled WebAssembly kernel makes that search ${FIGURES.kernel.replace('x', '')} times faster.`,
      `Finding the right neighbourhood did not make retrieval useful by itself. It scored ${FIGURES.hybrid} against ${FIGURES.recency} for simply returning the last ten things worked on, and a preregistered test found no measurable gain from handing recalled context to a model.`,
      `What did hold up: ${FIGURES.habits} things said in three or more separate sessions. The results file reports all of it as measured.`,
    ],
  },
];
