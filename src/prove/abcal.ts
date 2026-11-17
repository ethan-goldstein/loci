/**
 * Calibration set for the A/B judge. Every line here is invented: no real project, person or
 * conversation. Each item is a question, a reference answer and five candidate answers whose
 * correctness is known by construction, so a person can check the labels by reading them.
 *
 *   exact      states the reference almost word for word            correct
 *   reworded   states the same fact in other words or units          correct
 *   wrong      the same sentence with a different value or name      not correct
 *   abstain    "I do not know."                                      not correct
 *   other      a confident sentence about something else             not correct
 *
 * The first DEV_ITEMS items were used while writing the judge prompt. The rest were not, and only
 * they produce the numbers that are reported.
 */

export interface CalItem {
  question: string;
  reference: string;
  exact: string;
  reworded: string;
  wrong: string;
  other: string;
}

export const ABSTAIN = 'I do not know.';
export const DEV_ITEMS = 10;

const item = (project: string, q: string, reference: string, exact: string, reworded: string, wrong: string, other: string): CalItem => ({
  question: `In the project ${project}: ${q}`,
  reference,
  exact,
  reworded,
  wrong,
  other,
});

export const CALIBRATION: readonly CalItem[] = [
  // Development items (0 to 9).
  item('orbit', 'What step size does the propagator use by default?', '30 seconds', 'The propagator uses a default step size of 30 seconds.', 'It advances in 30 s steps unless configured otherwise.', 'The propagator uses a default step size of 45 seconds.', 'The propagator reads its orbital elements from a TLE file.'),
  item('kitchen', 'What hydration is the starter fed at?', '80 percent', 'The starter is fed at 80 percent hydration.', 'It is kept at a hydration of 80%.', 'The starter is fed at 65 percent hydration.', 'The starter lives in a glass jar on the counter.'),
  item('parser', 'Which parsing technique does the expression parser use?', 'Pratt parsing', 'The expression parser uses Pratt parsing.', 'It is a Pratt parser.', 'The expression parser uses an LR(1) table.', 'The parser reports errors with line and column numbers.'),
  item('garden', 'How often does the watering schedule run?', 'every two days', 'The watering schedule runs every two days.', 'It waters once every two days.', 'The watering schedule runs every week.', 'The watering schedule is stored as a cron string.'),
  item('network', 'What port does the relay server listen on?', '7443', 'The relay server listens on port 7443.', 'It binds to 7443.', 'The relay server listens on port 8080.', 'The relay server only accepts loopback connections.'),
  item('music', 'What tempo is the demo track set to?', '96 BPM', 'The demo track is set to 96 BPM.', 'Its tempo is 96 beats per minute.', 'The demo track is set to 140 BPM.', 'The demo track is in a minor key.'),
  item('orbit', 'Which library parses the TLE files?', 'sgp4-lite', 'TLE files are parsed with sgp4-lite.', 'Parsing is handled by the sgp4-lite package.', 'TLE files are parsed with satellite-js.', 'The TLE files are refreshed once a day.'),
  item('kitchen', 'How long is the dough proofed before baking?', 'eleven hours', 'The dough is proofed for eleven hours.', 'It proofs for 11 hours before the bake.', 'The dough is proofed for three hours.', 'The dough is shaped into a round loaf.'),
  item('parser', 'What token ends a statement?', 'a semicolon', 'A statement ends with a semicolon.', 'Statements are terminated by a semicolon token.', 'A statement ends with a newline.', 'Statements can be nested inside blocks.'),
  item('garden', 'Which bed are the tomatoes planted in?', 'bed C', 'The tomatoes are planted in bed C.', 'They are in bed C.', 'The tomatoes are planted in bed A.', 'The tomatoes are pruned to a single stem.'),

  // Held-out items (10 to 29): the reported calibration.
  item('network', 'How many times does the client retry a failed request?', 'five times', 'The client retries a failed request five times.', 'It makes 5 retries before giving up.', 'The client retries a failed request twice.', 'The client logs every failed request.'),
  item('music', 'Which scale does the melody generator default to?', 'D dorian', 'The melody generator defaults to D dorian.', 'By default it writes melodies in D dorian.', 'The melody generator defaults to C major.', 'The melody generator outputs MIDI.'),
  item('orbit', 'What reference frame are the positions stored in?', 'TEME', 'Positions are stored in the TEME frame.', 'They are kept in TEME coordinates.', 'Positions are stored in the ECEF frame.', 'Positions are stored as 64-bit floats.'),
  item('kitchen', 'What temperature is the oven preheated to?', '245 degrees', 'The oven is preheated to 245 degrees.', 'Preheat goes up to 245°.', 'The oven is preheated to 180 degrees.', 'The oven is preheated with the dutch oven inside.'),
  item('parser', 'How many precedence levels does the grammar define?', 'nine', 'The grammar defines nine precedence levels.', 'There are 9 levels of precedence.', 'The grammar defines four precedence levels.', 'Precedence is resolved in the lexer.'),
  item('garden', 'What sensor measures soil moisture?', 'a capacitive probe', 'Soil moisture is measured with a capacitive probe.', 'A capacitive probe reads the soil moisture.', 'Soil moisture is measured with a resistive fork sensor.', 'Soil moisture is logged every ten minutes.'),
  item('network', 'What backoff does the reconnect logic use?', 'exponential backoff with jitter', 'Reconnects use exponential backoff with jitter.', 'It backs off exponentially, with jitter added.', 'Reconnects use a fixed one-second delay.', 'Reconnects are counted in the metrics endpoint.'),
  item('music', 'How long is the reverb tail?', '2.4 seconds', 'The reverb tail is 2.4 seconds long.', 'The tail rings out for 2.4 s.', 'The reverb tail is 0.5 seconds long.', 'The reverb is applied after the compressor.'),
  item('orbit', 'How many satellites does the default catalogue load?', '1200', 'The default catalogue loads 1200 satellites.', 'It loads 1200 of them at start-up.', 'The default catalogue loads 5000 satellites.', 'The catalogue is cached in the browser for one hour.'),
  item('kitchen', 'Which flour does the recipe scaler assume?', 'strong white flour', 'The scaler assumes strong white flour.', 'It is written for strong white flour.', 'The scaler assumes wholemeal rye flour.', 'The scaler rounds every weight to the nearest gram.'),
  item('parser', 'What does the lexer do with comments?', 'drops them', 'The lexer drops them.', 'Comments are dropped by the lexer and never reach the parser.', 'The lexer keeps them as trivia tokens.', 'Comments start with two slashes.'),
  item('garden', 'What is the compost ratio of browns to greens?', 'three to one', 'The compost ratio is three to one.', 'Browns to greens is 3 to 1.', 'The compost ratio is one to one.', 'The compost bin is turned on Sundays.'),
  item('network', 'What is the handshake timeout?', '8 seconds', 'The handshake timeout is 8 seconds.', 'The handshake gives up after 8 s.', 'The handshake timeout is 30 seconds.', 'The handshake uses TLS 1.3.'),
  item('music', 'What sample rate does the synth render at?', '48 kHz', 'The synth renders at 48 kHz.', 'Rendering happens at a 48 kHz sample rate.', 'The synth renders at 22 kHz.', 'The synth has four oscillators.'),
  item('orbit', 'What caused the perigee calculation bug?', 'a degrees-to-radians mix-up', 'The bug was a degrees-to-radians mix-up.', 'Degrees and radians were mixed up in the perigee code.', 'The bug was an off-by-one error in the loop.', 'The perigee value is shown in kilometres.'),
  item('kitchen', 'Where does the recipe app store its timers?', 'in localStorage', 'The timers are stored in localStorage.', "It keeps them in the browser's localStorage.", 'The timers are stored in a Postgres table.', 'The timers ring with a short bell sound.'),
  item('parser', 'Which file holds the AST node definitions?', 'nodes.ts', 'The AST node definitions are in nodes.ts.', 'They live in the file nodes.ts.', 'The AST node definitions are in grammar.ts.', 'Each AST node carries a source span.'),
  item('garden', 'Which month does the planner start seedlings?', 'February', 'The planner starts seedlings in February.', 'Seedlings are started in February.', 'The planner starts seedlings in April.', 'Seedlings are kept under a grow light.'),
  item('network', 'Which header carries the request id?', 'x-trace-id', 'The request id is carried in the x-trace-id header.', 'It travels in x-trace-id.', 'The request id is carried in the x-request-token header.', 'Request ids are random 128-bit values.'),
  item('music', 'Which note is the bass line tuned down to?', 'low B', 'The bass line is tuned down to low B.', 'It goes down to a low B.', 'The bass line is tuned down to low E.', 'The bass line follows the kick drum.'),
];

export interface CalCase {
  question: string;
  reference: string;
  answer: string;
  correct: boolean;
}

/** The judged cases for a slice of the set: five per item, two correct and three not. */
export function calibrationCases(items: readonly CalItem[]): CalCase[] {
  return items.flatMap((it) => [
    { question: it.question, reference: it.reference, answer: it.exact, correct: true },
    { question: it.question, reference: it.reference, answer: it.reworded, correct: true },
    { question: it.question, reference: it.reference, answer: it.wrong, correct: false },
    { question: it.question, reference: it.reference, answer: ABSTAIN, correct: false },
    { question: it.question, reference: it.reference, answer: it.other, correct: false },
  ]);
}

/**
 * Invented passages for developing the question-writing prompt. They read like the assistant side
 * of a coding session and are used only by hand while tuning the prompt and by unit tests.
 */
export const DEV_PASSAGES: readonly string[] = [
  'The relay server now listens on port 7443 and only accepts loopback connections. I moved the retry logic into client.ts: a failed request is retried five times with exponential backoff and jitter, and the handshake gives up after 8 seconds. The request id travels in the x-trace-id header so the logs on both sides can be joined.',
  'I traced the perigee bug to a degrees-to-radians mix-up in propagate.ts: the inclination was read in degrees and passed straight to Math.sin. The propagator now converts once at load time. Positions are stored in the TEME frame, and the default catalogue loads 1200 satellites, parsed from TLE files with sgp4-lite.',
  'The expression parser is a Pratt parser with nine precedence levels, defined in a table at the top of parse.ts. The lexer drops comments before the parser ever sees them, and a statement ends with a semicolon. AST node definitions live in nodes.ts, and every node carries a source span for error messages.',
  'For the recipe scaler I kept the assumption of strong white flour and 80 percent hydration for the starter. Timers are stored in localStorage so a reload does not lose them. The oven step says to preheat to 245 degrees with the dutch oven inside, and the dough is proofed for eleven hours before the bake.',
  'The melody generator defaults to D dorian at 96 BPM. I rendered the demo at 48 kHz and set the reverb tail to 2.4 seconds, applied after the compressor. The bass line is tuned down to low B and follows the kick drum, which fixed the muddy mix in the second verse.',
  'The watering schedule runs every two days and is stored as a cron string in schedule.json. Soil moisture comes from a capacitive probe, logged every ten minutes. The tomatoes went into bed C this year, and the planner starts seedlings in February under a grow light.',
];
