import { rng } from './synth.ts';
import type { SynthDoc } from './synth.ts';

/**
 * Readable invented conversations for the public demo.
 *
 * The unit-test fixture (synth.ts) is a deliberate word salad: easy to reason about, unpleasant
 * to read. A visitor to the demo should see something that looks like the real thing, so this
 * generator writes short, plausible exchanges about six made-up projects. Every sentence is built
 * from the lists below; nothing here comes from anyone's actual history.
 *
 * Two properties matter:
 *   - A reply is coherent: each issue carries its own cause and fix, so the explanation always
 *     matches the symptom.
 *   - Prompts do not repeat by accident. Each is dealt from a shuffled deck in which no two cards
 *     share more than one of their parts, and cards that do share a part are worded differently
 *     around it. So what the repeat-finder finds in the demo is what was planted on purpose
 *     (PROSE_HABITS here and PLANTED in demo.ts), not an artefact of a small vocabulary.
 *
 * Pure TypeScript with no Node dependencies. Deterministic for a seed.
 */

/** What goes wrong (after "The <thing> "), why (after "because "), what was done (after "I "). */
type Issue = readonly [symptom: string, cause: string, fix: string];

interface Project {
  /** Parts of the project, in the same order as `issues`: things[i] is where issues[i] lives. */
  things: readonly string[];
  issues: readonly Issue[];
  /** Circumstances, completing "when ...". */
  conditions: readonly string[];
  /** Things to do next (base form, completing "I want to ..."). */
  goals: readonly string[];
  /** What is true afterwards, completing "... now." */
  checks: readonly string[];
}

export const PROSE_PROJECTS: Record<string, Project> = {
  orbit: {
    things: ['satellite pass predictor', 'ground track map', 'SGP4 propagator', 'TLE loader', 'orbit plot', 'pass alert', 'ephemeris cache', 'sky chart'],
    issues: [
      ['drifts a few seconds further off every day', 'the TLE epoch was read as local time', 'parsed the epoch as UTC'],
      ['puts the satellite on the wrong side of the planet', 'degrees and radians were mixed in the inclination', 'converted the inclination to radians once at load'],
      ['shows one satellite at the position of another', 'the propagator kept state between satellites', 'gave each satellite its own propagator state'],
      ['drops satellites whose TLE is valid', 'the checksum skipped the last column', 'included the last column in the TLE checksum'],
      ['reports a perigee below the surface', "Kepler's equation was solved with too few iterations", "iterated Kepler's equation to a tolerance"],
      ['skips passes that cross midnight', 'the pass window was cut at the end of the UTC day', 'let a pass window run across the day boundary'],
      ['recomputes every orbit on each frame', 'the cache key left out the satellite number', 'keyed the ephemeris cache by satellite and epoch'],
      ['freezes while the catalogue loads', 'the whole catalogue was propagated on the main thread', 'moved the SGP4 loop into a worker'],
    ],
    conditions: ['the TLE epoch is more than a week old', 'the orbit is nearly circular', 'two satellites share a launch number', 'the observer is near a pole', 'the ephemeris cache is cold', 'the inclination is above ninety degrees', 'the apogee is far above the perigee', 'a pass starts just before midnight'],
    goals: ['propagate the whole catalogue in a worker', 'cache the ephemeris between sessions', 'add a test against a published orbit', 'show apogee and perigee on the plot', 'refresh the TLE set once a day', 'draw the ground track for the next three passes', 'filter the catalogue by inclination', 'export upcoming passes as a calendar'],
    checks: ['the predicted pass matches the published one to the second', 'the orbit plot stays smooth at sixty frames a second', 'every TLE in the test set parses', 'perigee and apogee agree with the reference table', 'each satellite keeps its own track'],
  },
  kitchen: {
    things: ['hydration calculator', 'sourdough planner', 'proof timer', 'starter schedule', 'flour inventory', 'bake planner', 'bake history', 'recipe scaler'],
    issues: [
      ['rounds the hydration to a whole percent', 'dough weights were stored as whole grams', 'kept flour and water weights to a tenth of a gram'],
      ['leaves the starter out of the hydration', "the starter's own flour and water were never counted", "counted the starter's flour and water in the hydration"],
      ['shows times an hour off', 'timers were saved in local time', 'stored every timer as a UTC instant'],
      ['forgets the levain feeding after a restart', 'the levain plan was only held in memory', 'saved the levain plan on every change'],
      ['lists flour that ran out last week', 'the inventory never subtracted a finished bake', 'subtracted flour whenever a bake is logged'],
      ['schedules the bake before the starter peaks', 'the proof estimate ignored room temperature', 'scaled the proof time by temperature'],
      ['loses the crumb photos on sync', 'photos were uploaded before the bake had an id', 'queued crumb photos until the bake is saved'],
      ['doubles the salt when a recipe is scaled', 'salt was scaled by dough weight and again by flour weight', "scaled salt by flour weight only, as a baker's percentage"],
    ],
    conditions: ['the recipe mixes two kinds of flour', 'the kitchen is colder than usual', 'a bake is moved to the next day', 'the starter was fed late', 'the oven is still preheating', 'the phone is offline', 'the loaf is scaled past two kilos', 'two bakes overlap in the same evening'],
    goals: ['scale a sourdough recipe by flour weight', 'warn when the starter is past its peak', 'log crust colour with each bake', 'plan the proof around room temperature', 'export the bake history', 'suggest a hydration for the flour on hand', 'add a levain build to the schedule', 'compare two bakes in one view'],
    checks: ['the hydration matches a hand calculation', 'the proof timer survives a restart', 'the flour inventory balances after ten bakes', 'the starter schedule lines up with the bake', 'scaled recipes keep the same percentages'],
  },
  parser: {
    things: ['tokenizer', 'recursive descent parser', 'syntax error reporter', 'precedence table', 'AST builder', 'lexer', 'keyword matcher', 'parse cache'],
    issues: [
      ['drops the last token of the file', 'the loop stopped one step before the end', 'moved the bounds check ahead of the read'],
      ['loops forever on nested brackets', 'a grammar rule called itself first', 'rewrote the left-recursive rule as a loop'],
      ['reports the wrong line for a syntax error', 'line numbers were counted after the token was read', 'recorded the line when each token starts'],
      ['binds minus tighter than multiply', 'two rows of the precedence table were swapped', 'put the precedence rows back in order'],
      ['allocates a node for every space', 'whitespace tokens were kept in the tree', 'dropped whitespace in the tokenizer'],
      ['accepts an unterminated string', 'the lexer treated an escaped quote as the closing quote', 'handled escapes inside strings in the lexer'],
      ['reads a keyword prefix as a keyword', 'keywords were matched before the longest identifier', 'matched the longest identifier first and then checked keywords'],
      ['parses the whole file again after a one-line edit', 'the parse cache was keyed by file time instead of content', 'keyed the parse cache by a hash of the source'],
    ],
    conditions: ['the input ends without a newline', 'two operators are adjacent', 'a comment contains a quote', 'the grammar has a left-recursive rule', 'the file is larger than a megabyte', 'an identifier starts with a keyword', 'brackets are nested ten deep', 'a string spans two lines'],
    goals: ['add error recovery to the parser', 'write tests for the lexer', 'print the AST as indented text', 'turn the precedence table into data', 'reuse the parse of unchanged files', 'report every syntax error in one pass', 'fuzz the tokenizer with random input', 'document the grammar in one page'],
    checks: ['the grammar tests pass', 'nested input parses in one pass', 'each syntax error points at the right column', 'the parser handles the large file in under a second', 'the tokenizer round-trips every sample file'],
  },
  garden: {
    things: ['watering schedule', 'harvest tally', 'seedling tray tracker', 'soil moisture reader', 'mulch calculator', 'tomato bed planner', 'trellis layout', 'compost log'],
    issues: [
      ['waters a bed right after rain', "the schedule read yesterday's forecast", 'fetched the forecast just before each watering'],
      ['counts the same harvest twice', 'tally entries were keyed by time alone', 'keyed harvest entries by bed and time'],
      ['shows seedlings that were already planted out', 'planted trays were never marked done', 'marked a tray done when it is planted out'],
      ['reads the soil as always dry', 'the sensor value was read before it settled', 'averaged the soil sensor over a minute'],
      ['orders twice the mulch needed', 'feet were treated as metres in the mulch sum', 'converted bed sizes to metres on entry'],
      ['forgets which bed had tomatoes last year', 'crop history was stored per season instead of per bed', 'kept crop history for every bed'],
      ['puts tall plants where they shade the tomato rows', 'the layout ignored which way the bed faces', 'sorted rows by height from the sunny side'],
      ['reports the compost as ready too early', 'the log used the newest temperature instead of the weekly average', 'judged the compost on its weekly average temperature'],
    ],
    conditions: ['two beds share one sensor', 'the forecast changes during the day', 'a seedling tray is split across beds', 'the sensor battery is low', 'a bed is measured in feet', 'a harvest is entered from the phone', 'the first frost comes early', 'a bed is left fallow for a season'],
    goals: ['skip watering when rain is forecast', 'rotate tomato beds year to year', 'track compost temperature', 'plan the trellis before planting', 'chart the harvest by week', 'send a reminder to prune and mulch', 'label each seedling tray with its sowing date', 'estimate the harvest from last year'],
    checks: ['no bed is watered within a day of rain', 'the harvest total matches the paper notes', 'the trellis plan fits the bed', 'the compost log shows a clean weekly curve', 'every tomato bed has a different crop from last year'],
  },
  network: {
    things: ['socket pool', 'retry policy', 'packet framer', 'latency monitor', 'stream buffer', 'timeout handler', 'connection manager', 'read loop'],
    issues: [
      ['leaks a socket on every failed handshake', 'the socket was only closed on the success path', 'closed the socket in a finally block'],
      ['retries without any backoff', 'the backoff timer was reset by each attempt', 'doubled the backoff after each retry and added jitter'],
      ['splits a packet across two reads', 'the framer assumed one read per packet', 'buffered partial packets in the framer'],
      ['reports a latency of zero', 'latency used wall-clock time instead of a monotonic clock', 'switched the latency monitor to a monotonic clock'],
      ['stalls once it is full', 'nothing slowed the sender when the buffer filled', 'paused the stream while the buffer is full'],
      ['fires a timeout after the reply arrives', 'the timeout was never cancelled', 'cancelled the timeout when the reply lands'],
      ['opens a new connection for every request', 'idle sockets were closed instead of returned to the pool', 'returned idle sockets to the pool'],
      ['hangs when the server goes quiet', 'reads had no deadline', 'put a deadline on every read'],
    ],
    conditions: ['the server closes the connection mid-stream', 'the first TCP packet is dropped', 'a reply is larger than the buffer', 'the clock jumps backwards', 'ten clients connect at once', 'the handshake takes longer than a second', 'the network switches from wifi to mobile', 'a proxy sits in between'],
    goals: ['add jitter to the backoff', 'measure latency per request', 'reuse sockets across requests', 'cap the stream buffer and apply back-pressure', 'log every retry with its reason', 'time out a handshake that stalls', 'replay a captured packet trace in a test', 'expose pool size as a setting'],
    checks: ['no socket is left open after a thousand failed handshakes', 'retries spread out instead of arriving together', 'packets survive being split at any byte', 'latency stays positive when the clock changes', 'the pool holds steady under load'],
  },
  music: {
    things: ['synth voice', 'chord detector', 'tempo tracker', 'octave switch', 'reverb unit', 'rhythm grid', 'voice allocator', 'melody editor'],
    issues: [
      ['clicks when a note ends', 'the voice stopped at a non-zero sample', 'added a short release to every synth voice'],
      ['hears a minor chord as major', 'the chord detector ignored the third', 'weighted the third when naming a chord'],
      ['drifts off the tempo after a minute', 'tempo was estimated once and never updated', 'tracked tempo over a sliding window'],
      ['doubles the bass an octave up', 'the octave switch was applied twice', 'applied the octave switch once per note'],
      ['smears every note in the mix', 'the reverb fed its own output back at full level', 'lowered the reverb feedback and cut its low end'],
      ['snaps notes to the wrong beat', 'grid positions were rounded down instead of to the nearest', 'rounded melody notes to the nearest grid line'],
      ['drops notes when many play at once', 'the oldest voice was cut without a fade', 'faded out the oldest voice before reusing it'],
      ['plays the melody a semitone flat', 'the tuning table started from the wrong reference pitch', 'rebuilt the tuning table from concert pitch'],
    ],
    conditions: ['two notes overlap', 'the tempo changes mid-song', 'the reverb tail is longer than the bar', 'more than eight voices play', 'the scale is neither major nor minor', 'audio is buffered in small blocks', 'the melody jumps an octave', 'the song is in triple time'],
    goals: ['fade each synth voice out instead of cutting it', 'detect the scale of a melody', 'sync the reverb to the tempo', 'let the rhythm grid swing', 'keep the bass in mono in the mix', 'show the chord names above the melody', 'add a metronome that follows the tempo tracker', 'export the melody as sheet music'],
    checks: ['notes end without a click', 'the chord detector agrees with the sheet music', 'the tempo holds for the whole song', 'the mix stays clear with eight voices playing', 'the melody plays in tune'],
  },
};

export const PROSE_TOPICS = Object.keys(PROSE_PROJECTS);

/**
 * Habits planted in the public demo on top of PLANTED, so the "what you keep repeating" panel
 * shows more than one kind of thing: instructions, questions and a way of starting work. Entries
 * repeat to set how often each one is said. All invented.
 */
export const PROSE_HABITS = [
  'Use metric units everywhere in the output.',
  'What did you change in that last step and why?',
  'Use metric units everywhere in the output please.',
  "Let's pick up where we left off last time.",
  'Do not touch the config files in this pass.',
  'Use metric units everywhere in the output.',
  'Is there a simpler way to do this?',
  'What exactly did you change in that last step and why?',
  'Do not touch the config files in this pass please.',
  "Let's pick up where we left off last time.",
  'Use metric units everywhere in the output.',
  'Is there a simpler way to do this one?',
] as const;

/** Where or when a problem was seen. Each adds its own words, which keeps prompts apart. */
const DETAILS = ['on the staging build', 'in the nightly run', 'on my laptop', 'after the last merge', 'since the weekend', 'on the release branch', 'with the sample data', 'in the demo account', 'during the import', 'on a fresh install'] as const;

/** What a part of a project does in an awkward case (after "the <thing> "). */
const OUTCOMES = [
  'falls back to the last good value and logs a warning',
  'refuses the input and says why',
  'keeps the previous result until new data arrives',
  'retries once and then reports the failure',
  'marks the result as approximate',
  'asks before doing anything that cannot be undone',
  'skips that item and carries on with the rest',
  'queues the work until the condition clears',
] as const;

const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

export interface ProseDoc extends SynthDoc {
  role: 'human' | 'assistant';
}

/**
 * Deals the numbers 0..size-1 in a shuffled order and says how many times the deck has been
 * through. A second pass only happens when a project gets more prompts than it has cards.
 */
function deck(size: number, rand: () => number): () => { card: number; pass: number } {
  const cards = Array.from({ length: size }, (_, i) => i);
  let at = size;
  let pass = -1;
  return () => {
    if (at === size) {
      for (let i = size - 1; i > 0; i--) {
        const j = Math.floor(rand() * (i + 1));
        [cards[i], cards[j]] = [cards[j]!, cards[i]!];
      }
      at = 0;
      pass++;
    }
    return { card: cards[at++]!, pass };
  };
}

/** One prompt in this many documents is typed by the person; the rest are replies. */
const PROMPT_EVERY = 6;

/**
 * `n` invented documents in chronological order, one an hour. `planted` are standing instructions
 * to sprinkle through the prompts, alongside PROSE_HABITS.
 */
export function proseDocs(n: number, seed: number, planted: readonly string[] = []): ProseDoc[] {
  const rand = rng(seed);
  const pick = <T>(xs: readonly T[]): T => xs[Math.floor(rand() * xs.length)]!;
  // Per project: 64 problems (issue x condition), 56 plans (ordered pairs of goals) and 64
  // questions (thing x condition). Within a deck, two cards share at most one part.
  const decks = new Map(PROSE_TOPICS.map((t) => [t, { problem: deck(64, rand), plan: deck(56, rand), question: deck(64, rand) }]));

  // The wording around a card, and the part of the project it names, are chosen from the card
  // itself: two cards that share a part (the same issue, say) never share their wording, and two
  // cards with the same wording never name the same part. The wording is kept short so that it is
  // the subject matter, not the turn of phrase, that makes two prompts alike.
  const problem = (topic: string, p: Project): string => {
    const { card, pass } = decks.get(topic)!.problem();
    const s = card % 8;
    const c = card >> 3;
    // Where it was noticed, which need not be where the fault lies.
    const thing = p.things[(s + 2 * c + pass) % 8]!;
    const symptom = p.issues[s]![0];
    const condition = p.conditions[c]!;
    const detail = pick(DETAILS);
    switch ((s + c + pass * 3) % 8) {
      case 0:
        return `The ${thing} ${symptom} when ${condition}, ${detail}.`;
      case 1:
        return `Can you look at why the ${thing} ${symptom} when ${condition}?`;
      case 2:
        return `Something is off with the ${thing}, it ${symptom} whenever ${condition}.`;
      case 3:
        return `I noticed ${detail} that the ${thing} ${symptom} if ${condition}.`;
      case 4:
        return `Bug: the ${thing} ${symptom} while ${condition}.`;
      case 5:
        return `Still seeing this ${detail}, the ${thing} ${symptom} when ${condition}.`;
      case 6:
        return `Low priority, but the ${thing} ${symptom} every time ${condition}.`;
      default:
        return `A strange one ${detail}: the ${thing} ${symptom} once ${condition}.`;
    }
  };

  const plan = (topic: string, p: Project): string => {
    const { card, pass } = decks.get(topic)!.plan();
    const a = card % 8;
    const b = (a + 1 + Math.floor(card / 8)) % 8; // any goal but the first one
    const thing = p.things[(a + 3 + pass) % 8]!;
    const first = p.goals[a]!;
    const second = p.goals[b]!;
    switch ((a + 3 * b + pass * 3) % 8) {
      case 0:
        return `Next, ${first} and after that ${second}, once the ${thing} is stable.`;
      case 1:
        return `Now that the ${thing} works, let's ${first} and then ${second}.`;
      case 2:
        return `Two jobs for today around the ${thing}: ${first}, then ${second}.`;
      case 3:
        return `Could we ${first} before we ${second}, since the ${thing} depends on it?`;
      case 4:
        return `Plan: ${first} and ${second}, leaving the ${thing} alone.`;
      case 5:
        return `Ahead of the ${thing} work, ${first}, then ${second}.`;
      case 6:
        return `When you are done with the ${thing}, please ${first} and afterwards ${second}.`;
      default:
        return `This week: ${first}, ${second}, and later the ${thing}.`;
    }
  };

  const question = (topic: string, p: Project): string => {
    const { card, pass } = decks.get(topic)!.question();
    const t = card % 8;
    const c = card >> 3;
    const thing = p.things[t]!;
    const condition = p.conditions[c]!;
    switch ((t + c + pass * 3) % 8) {
      case 0:
        return `How should the ${thing} behave when ${condition}?`;
      case 1:
        return `What should the ${thing} show if ${condition}?`;
      case 2:
        return `Is the ${thing} tested for when ${condition}?`;
      case 3:
        return `Does the ${thing} cope when ${condition}?`;
      case 4:
        return `What happens in the ${thing} when ${condition}?`;
      case 5:
        return `Why would the ${thing} fail when ${condition}?`;
      case 6:
        return `Remind me what the ${thing} does when ${condition}.`;
      default:
        return `Describe the ${thing} for the case where ${condition}.`;
    }
  };

  // Replies come in four kinds, so a search does not return twelve copies of one sentence.
  const reply = (p: Project): string => {
    const kind = rand();
    if (kind < 0.3) {
      // What was wrong and what was done about it, told six ways.
      const j = Math.floor(rand() * 8);
      const thing = p.things[j]!;
      const [symptom, cause, fix] = p.issues[j]!;
      const condition = pick(p.conditions);
      const out: string[] = [];
      switch (Math.floor(rand() * 6)) {
        case 0:
          out.push(`The ${thing} ${symptom} because ${cause}.`, `I ${fix}.`);
          break;
        case 1:
          out.push(`The cause was that ${cause}, which is why the ${thing} ${symptom}.`, `I ${fix}.`);
          break;
        case 2:
          out.push('Found it.', `${cap(cause)}, so the ${thing} ${symptom}.`, `I ${fix} and added a test that fails without the change.`);
          break;
        case 3:
          out.push(`When ${condition}, the ${thing} ${symptom}.`, `The reason is that ${cause}.`, `I ${fix}.`);
          break;
        case 4:
          out.push(`I ${fix}.`, `The ${thing} no longer ${symptom}, including when ${condition}.`);
          break;
        default:
          out.push(`Short version: ${cause}.`, `That is why the ${thing} ${symptom}, so I ${fix}.`);
      }
      if (rand() < 0.5) out.push(`${cap(pick(p.checks))} now.`);
      if (rand() < 0.25) out.push(`A sensible next step is to ${pick(p.goals)}.`);
      return out.join(' ');
    }
    if (kind < 0.65) {
      // An answer about behaviour.
      const thing = pick(p.things);
      const condition = pick(p.conditions);
      const outcome = pick(OUTCOMES);
      if (rand() < 0.5) return `When ${condition}, the ${thing} ${outcome}.${rand() < 0.5 ? ` ${cap(pick(p.checks))}, so that case is covered.` : ''}`;
      return `The ${thing} ${outcome} if ${condition}.${rand() < 0.4 ? ` It showed up when ${pick(p.conditions)}.` : ''}`;
    }
    if (kind < 0.85) {
      // An order of work, worded three ways so the wording is not what these have in common.
      const a = Math.floor(rand() * 8);
      const b = (a + 1 + Math.floor(rand() * 7)) % 8;
      const first = p.goals[a]!;
      const second = p.goals[b]!;
      const thing = pick(p.things);
      const form = rand();
      if (form < 0.34) return `Order: ${first}, then ${second}. The ${thing} stays as it is.`;
      if (form < 0.67) return `Start with the ${thing}: ${first}. After that, ${second}.`;
      return `${cap(first)} first. That unblocks the work to ${second}, and the ${thing} is not affected.`;
    }
    // A test report.
    return `I added a test for the case where ${pick(p.conditions)}. ${cap(pick(p.checks))} now.${rand() < 0.5 ? ` A sensible next step is to ${pick(p.goals)}.` : ''}`;
  };

  const out: ProseDoc[] = [];
  const start = Date.UTC(2026, 0, 1);
  let prompts = 0;
  for (let i = 0; i < n; i++) {
    const topic = pick(PROSE_TOPICS);
    const p = PROSE_PROJECTS[topic]!;
    const role = i % PROMPT_EVERY === 0 ? 'human' : 'assistant';
    let text: string;
    if (role === 'human') {
      const kind = rand();
      text = kind < 0.4 ? problem(topic, p) : kind < 0.6 ? plan(topic, p) : kind < 0.9 ? question(topic, p) : `${problem(topic, p)} ${plan(topic, p)}`;
      // The habits: about one prompt in seven carries a standing instruction, one in eleven
      // another habit. Real history looks like this; it is what the repeat-finder is for.
      if (planted.length > 0 && prompts % 7 === 3) text += ` ${planted[Math.floor(prompts / 7) % planted.length]!}`;
      else if (prompts % 11 === 5) text += ` ${PROSE_HABITS[Math.floor(prompts / 11) % PROSE_HABITS.length]!}`;
      prompts++;
    } else {
      text = reply(p);
    }
    out.push({ topic, text, ts: start + i * 3_600_000, role });
  }
  return out;
}
