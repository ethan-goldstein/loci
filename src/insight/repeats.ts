import type { Embedder } from '../embed/ollama.ts';
import type { Engine } from '../engine.ts';
import { dot } from '../index/hnsw.ts';
import { calibrate } from '../index/simstats.ts';
import type { Threshold } from '../index/simstats.ts';
import { tokenize } from '../index/tokenize.ts';

/**
 * Repeat-finder: which things do you keep telling your assistant, session after session?
 *
 * The unit is a sentence you typed, not a whole prompt, because a standing instruction ("run the
 * tests before you commit") usually rides along inside a longer message about something else.
 *
 *   1. Take every prompt you typed, split it into sentences, and drop what is not prose
 *      (pasted code, logs, paths) or carries no content ("ok go ahead").
 *   2. Embed each sentence with the local model.
 *   3. Decide how similar two sentences must be to count as the same thing. That threshold is
 *      not hand-picked: it is set from the data, as a number of robust standard deviations above
 *      the similarity of a typical pair of unrelated sentences.
 *   4. Compare every pair exactly (no approximate index: at this size it is affordable and there
 *      is nothing to second-guess) and group sentences around the ones with the most matches.
 *   5. Keep groups that span several separate sessions on more than one day. Saying something
 *      three times in one sitting is emphasis, and the same text in many sessions on a single day
 *      is usually a script or a batch of parallel runs; saying it in different sessions on
 *      different days is a habit.
 *
 * The report also says how many matching pairs would be expected by chance under that threshold,
 * so "you repeat yourself" is a measured claim rather than an impression.
 *
 * Everything stays local. The text in the output is yours and private: it is written to the store
 * directory, never to the repository.
 */

/**
 * instruction  something to do or not do, the raw material for a standing rule
 * task         the opening of a piece of work you start again and again ("let's update the changelog")
 * question     something you keep asking
 * statement    anything else
 */
export type Kind = 'instruction' | 'task' | 'question' | 'statement';

export interface RepeatGroup {
  /** The most central phrasing, in your own words. */
  text: string;
  kind: Kind;
  occurrences: number;
  sessions: number;
  projects: string[];
  /** Milliseconds since the epoch. */
  first: number;
  last: number;
  /** Chunk nodes that contain one of the group's sentences, ascending. */
  nodes: number[];
  /** Up to three other ways you phrased it. */
  variants: string[];
}

export interface RepeatsReport {
  generated: string;
  /** Chunks in the index when this was computed; used to tell whether the report is stale. */
  corpus: number;
  prompts: number;
  sentences: number;
  minSessions: number;
  threshold: Threshold;
  pairs: { crossSession: number; above: number; expectedByChance: number };
  groups: number;
  sentencesInGroups: number;
  share: number;
  byKind: Record<Kind, number>;
  /** Prompts left out because they are a repeated template (written by a program, not typed). */
  templatedPrompts: number;
  /** Groups left out because every occurrence fell on one calendar day. */
  sameDayBursts: number;
  /** How many groups span exactly this many sessions (the last bucket is "this many or more"). */
  spread: { sessions: number; groups: number }[];
  top: RepeatGroup[];
}

export interface Sentence {
  node: number;
  text: string;
}

/** Words that carry no content on their own: function words, acknowledgements, go-aheads. */
const FILLER = new Set(
  `a an the this that these those it its i me my we us our you your he she they them their
  is are was were be been being am do does did done doing have has had having will would shall should can could may might must
  and or but so if then than as at by for from in into of off on onto out over to up with about after again before
  not no yes yeah yep ok okay sure please thanks thank lets let go going ahead continue proceed keep now next just also too very really
  what which who how why when where there here all any some more most much many few other same such only own
  good great nice cool perfect awesome sounds fine right well lol haha hmm hey hi hello
  im ive id ill dont doesnt didnt cant wont isnt thats whats youre its s t d ll re ve m`.split(/\s+/),
);

/** Splits a prompt into sentence-like pieces and strips list markers and wrapping quotes. */
export function splitSentences(text: string): string[] {
  return text
    .split(/(?<=[.!?])\s+|\n+/)
    .map((s) =>
      s
        .replace(/^\s*(?:[-*•]|\d{1,2}[.)])\s+/, '')
        .replace(/^["'“”‘’`\s]+|["'“”‘’`\s]+$/g, '')
        .trim(),
    )
    .filter((s) => s.length > 0);
}

/** True for a sentence worth comparing: readable prose with some content in it. */
export function isCandidate(s: string): boolean {
  if (s.length < 12 || s.length > 200) return false;
  // Code, logs, markup, paths and redacted material are not things you "say".
  if (/[{}<>;=|\\`]|:\/\/|\[REDACTED|~\/|\/[\w.-]+\/[\w.-]+|^\s*[$#>@]/.test(s)) return false;
  const visible = s.replace(/\s/g, '');
  const letters = visible.replace(/[^\p{L}]/gu, '').length;
  if (letters / visible.length < 0.72) return false;
  const tokens = tokenize(s);
  if (tokens.length < 3 || tokens.length > 40) return false;
  return tokens.filter((t) => t.length >= 2 && !FILLER.has(t)).length >= 2;
}

export function classify(s: string): Kind {
  const t = s.toLowerCase().trim();
  if (
    t.endsWith('?') ||
    /^(what|why|how|when|where|which|who|is|are|was|were|does|do(?! not\b)|did|can|could|would|will|should|have|has)\b/.test(t)
  ) {
    // "can you add tests" is a request, not a question, even with a question mark.
    return /^(can|could|would|will) you\b/.test(t) ? 'instruction' : 'question';
  }
  // "let's work on X", "now X": the start of a piece of work, not a rule about how to do it.
  if (/^(ok\s+|okay\s+|so\s+|and\s+)*(let'?s|now|next|time to|back to)\b/.test(t)) return 'task';
  if (
    /^(please\s+|and\s+|also\s+|then\s+|ok\s+|okay\s+)*(always|never|no|only|don'?t|do not|make sure|ensure|use|avoid|keep|stop|remember|add|remove|run|fix|check|write|update|show|give|tell|explain|create|build|make|put|move|change|set|test|verify|clean|delete|rename|refactor|commit|push|deploy|install|try|start|open|read|look|find|continue|i want|i need|i'?d like|we need|you should|you need|you must|it should|it needs|be|respond|answer|reply)\b/.test(
      t,
    ) ||
    /\b(always|never|don'?t|do not|make sure|must|should not|shouldn'?t|no more|instead of)\b/.test(t)
  ) {
    return 'instruction';
  }
  return 'statement';
}

export { calibrate };

/** Upper tail of the standard normal distribution. */
export function normalTail(z: number): number {
  // Abramowitz and Stegun 26.2.17.
  const t = 1 / (1 + 0.2316419 * Math.abs(z));
  const poly = t * (0.31938153 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
  const tail = (Math.exp((-z * z) / 2) / Math.sqrt(2 * Math.PI)) * poly;
  return z >= 0 ? tail : 1 - tail;
}

/**
 * Exact comparison of every pair. Returns, for each sentence, the others at or above `tau`, and
 * counts of cross-session pairs (all, and above the threshold).
 */
export function neighboursAbove(
  vectors: Float32Array,
  dim: number,
  n: number,
  tau: number,
  sessionOf: ArrayLike<number>,
  onProgress?: (done: number, total: number) => void,
): { adjacency: number[][]; crossSession: number; above: number } {
  const adjacency: number[][] = Array.from({ length: n }, () => []);
  let crossSession = 0;
  let above = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const cross = sessionOf[i] !== sessionOf[j];
      if (cross) crossSession++;
      if (dot(vectors, i * dim, vectors, j * dim, dim) >= tau) {
        adjacency[i]!.push(j);
        adjacency[j]!.push(i);
        if (cross) above++;
      }
    }
    if (onProgress && (i % 200 === 0 || i === n - 1)) onProgress(i + 1, n);
  }
  return { adjacency, crossSession, above };
}

/**
 * Leader clustering: repeatedly take the unassigned sentence with the most matches and group it
 * with its unassigned matches. Every member is within the threshold of its group's leader, so
 * groups cannot drift the way chained (single-link) clusters do.
 */
export function leaderClusters(adjacency: readonly (readonly number[])[]): number[][] {
  const n = adjacency.length;
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => adjacency[b]!.length - adjacency[a]!.length || a - b);
  const taken = new Uint8Array(n);
  const clusters: number[][] = [];
  for (const leader of order) {
    if (taken[leader]) continue;
    taken[leader] = 1;
    const members = [leader];
    for (const j of adjacency[leader]!) {
      if (!taken[j]) {
        taken[j] = 1;
        members.push(j);
      }
    }
    clusters.push(members);
  }
  return clusters;
}

export interface RepeatsOptions {
  embedder: Embedder;
  /** A group must appear in at least this many separate sessions. */
  minSessions?: number;
  /** Robust standard deviations above the typical pair that count as "the same thing". */
  z?: number;
  top?: number;
  onProgress?: (stage: string, done: number, total: number) => void;
}

/**
 * Prompts that are a program's template rather than something typed: a long prompt whose opening
 * (digits masked) is shared by prompts in several other sessions. People repeat short phrases;
 * they do not retype the same three-hundred-character preamble in session after session.
 */
export function templatedPrompts(engine: Engine, minLength = 300, minSessions = 3): Set<number> {
  const key = (text: string) => text.slice(0, 60).toLowerCase().replace(/\d+/g, '#').replace(/\s+/g, ' ').trim();
  const sessionsByKey = new Map<string, Set<number>>();
  const nodesByKey = new Map<string, number[]>();
  for (let node = 0; node < engine.n; node++) {
    if (engine.role[node] !== 0 || engine.texts[node]!.length < minLength) continue;
    const k = key(engine.texts[node]!);
    (sessionsByKey.get(k) ?? sessionsByKey.set(k, new Set()).get(k)!).add(engine.sessionPk[node]!);
    (nodesByKey.get(k) ?? nodesByKey.set(k, []).get(k)!).push(node);
  }
  const out = new Set<number>();
  for (const [k, sessions] of sessionsByKey) if (sessions.size >= minSessions) for (const node of nodesByKey.get(k)!) out.add(node);
  return out;
}

/** Collects the candidate sentences from every prompt the person typed. */
export function candidateSentences(engine: Engine): { sentences: Sentence[]; prompts: number; templated: number } {
  const sentences: Sentence[] = [];
  const templates = templatedPrompts(engine);
  // A templated prompt that was split into several chunks: later chunks of the same turn go too.
  const templatedTurns = new Set<string>();
  const turnOf = new Map<number, string>();
  for (const row of engine.store.db.prepare("SELECT id, uuid FROM chunks WHERE role = 'human'").all() as { id: number; uuid: string }[]) {
    turnOf.set(Number(row.id) - 1, String(row.uuid));
  }
  for (const node of templates) {
    const turn = turnOf.get(node);
    if (turn) templatedTurns.add(turn);
  }
  let prompts = 0;
  let templated = 0;
  for (let node = 0; node < engine.n; node++) {
    if (engine.role[node] !== 0) continue;
    if (templates.has(node) || templatedTurns.has(turnOf.get(node) ?? '')) {
      templated++;
      continue;
    }
    prompts++;
    // The same sentence twice in one chunk counts once.
    const seen = new Set<string>();
    for (const s of splitSentences(engine.texts[node]!)) {
      const key = s.toLowerCase().replace(/\s+/g, ' ');
      if (seen.has(key) || !isCandidate(s)) continue;
      seen.add(key);
      sentences.push({ node, text: s });
    }
  }
  return { sentences, prompts, templated };
}

export async function runRepeats(engine: Engine, opts: RepeatsOptions): Promise<RepeatsReport> {
  const minSessions = opts.minSessions ?? 3;
  const z = opts.z ?? 5;
  const { sentences, prompts, templated } = candidateSentences(engine);
  const n = sentences.length;
  const empty: RepeatsReport = {
    generated: new Date().toISOString(),
    corpus: engine.n,
    prompts,
    sentences: n,
    minSessions,
    threshold: { median: 0, sigma: 0, z, tau: 1, sampledPairs: 0 },
    pairs: { crossSession: 0, above: 0, expectedByChance: 0 },
    groups: 0,
    sentencesInGroups: 0,
    share: 0,
    byKind: { instruction: 0, task: 0, question: 0, statement: 0 },
    templatedPrompts: templated,
    sameDayBursts: 0,
    spread: [],
    top: [],
  };
  if (n < 2) return empty;

  // Embed. The 'cluster' kind asks the model for embeddings suited to comparing texts with each other.
  let dim = 0;
  let vectors = new Float32Array(0);
  for (let i = 0; i < n; i += 64) {
    const batch = await opts.embedder.embed(
      sentences.slice(i, i + 64).map((s) => s.text),
      'cluster',
    );
    if (dim === 0) {
      dim = batch[0]!.length;
      vectors = new Float32Array(n * dim);
    }
    batch.forEach((v, j) => vectors.set(v, (i + j) * dim));
    opts.onProgress?.('embedding sentences', Math.min(i + 64, n), n);
  }

  const sessionOf = Int32Array.from(sentences, (s) => engine.sessionPk[s.node]!);
  const threshold = calibrate(vectors, dim, n, sessionOf, z);
  const { adjacency, crossSession, above } = neighboursAbove(vectors, dim, n, threshold.tau, sessionOf, (d, t) => opts.onProgress?.('comparing sentences', d, t));

  const groups: RepeatGroup[] = [];
  let sentencesInGroups = 0;
  let sameDayBursts = 0;
  for (const members of leaderClusters(adjacency)) {
    const sessions = new Set(members.map((m) => sessionOf[m]!));
    if (sessions.size < minSessions) continue;
    const times = members.map((m) => engine.ts[sentences[m]!.node]!);
    if (new Set(times.map((t) => Math.floor(t / 86_400_000))).size < 2) {
      sameDayBursts++;
      continue;
    }
    sentencesInGroups += members.length;
    const leader = members[0]!;
    const projects = [...new Set(members.map((m) => engine.projectName(engine.project[sentences[m]!.node]!)))].sort();
    // Other phrasings: distinct from the leader's wording, most recent first.
    const leaderKey = sentences[leader]!.text.toLowerCase();
    const variants = [...new Map(members.slice(1).map((m) => [sentences[m]!.text.toLowerCase(), m])).entries()]
      .filter(([key]) => key !== leaderKey)
      .map(([, m]) => m)
      .sort((a, b) => engine.ts[sentences[b]!.node]! - engine.ts[sentences[a]!.node]!)
      .slice(0, 3)
      .map((m) => sentences[m]!.text);
    groups.push({
      text: sentences[leader]!.text,
      kind: classify(sentences[leader]!.text),
      occurrences: members.length,
      sessions: sessions.size,
      projects,
      first: Math.min(...times),
      last: Math.max(...times),
      nodes: [...new Set(members.map((m) => sentences[m]!.node))].sort((a, b) => a - b),
      variants,
    });
  }
  groups.sort((a, b) => b.sessions - a.sessions || b.occurrences - a.occurrences || b.last - a.last || a.text.localeCompare(b.text));

  const byKind: Record<Kind, number> = { instruction: 0, task: 0, question: 0, statement: 0 };
  for (const g of groups) byKind[g.kind]++;
  const buckets = [3, 4, 5, 6, 8, 12].filter((b) => b >= minSessions);
  if (buckets[0] !== minSessions) buckets.unshift(minSessions);
  const spread = buckets.map((lo, i) => {
    const hi = buckets[i + 1] ?? Infinity;
    return { sessions: lo, groups: groups.filter((g) => g.sessions >= lo && g.sessions < hi).length };
  });

  return {
    ...empty,
    threshold,
    pairs: { crossSession, above, expectedByChance: crossSession * normalTail(threshold.z) },
    groups: groups.length,
    sentencesInGroups,
    share: sentencesInGroups / n,
    byKind,
    sameDayBursts,
    spread,
    top: groups.slice(0, opts.top ?? 60),
  };
}

const day = (ms: number) => new Date(ms).toISOString().slice(0, 10);

/**
 * Draft standing rules, one per recurring instruction, each in the person's own words with the
 * evidence beside it. A starting point to edit, not something to paste in unread.
 */
export function draftRules(report: RepeatsReport, limit = 20): string {
  const rules = report.top.filter((g) => g.kind === 'instruction').slice(0, limit);
  const lines = [
    '# Draft standing rules',
    '',
    `Things you told your assistant in at least ${report.minSessions} separate sessions, in your own words.`,
    'Review and reword before adding any of these to a CLAUDE.md or other instruction file.',
    '',
  ];
  if (rules.length === 0) lines.push('No recurring instructions found.');
  for (const g of rules) {
    lines.push(`- ${g.text}`);
    lines.push(`  - said ${g.occurrences} times in ${g.sessions} sessions across ${g.projects.length} project${g.projects.length === 1 ? '' : 's'} (${g.projects.join(', ')}), ${day(g.first)} to ${day(g.last)}`);
    for (const v of g.variants) lines.push(`  - also: ${v}`);
  }
  return `${lines.join('\n')}\n`;
}
