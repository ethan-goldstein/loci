import { createHash } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { ASSISTANT_CHUNKS, HUMAN_CHUNKS, chunkText } from './chunk.ts';
import { parseChatGptFile, resolveExport } from './chatgpt.ts';
import { parseSessionFile } from './parse.ts';
import type { ParseOptions, ParsedFile } from './parse.ts';
import { mergeCounts, redact } from './redact.ts';
import type { RedactionCounts } from './redact.ts';
import type { Store } from './store.ts';

export interface IngestOptions extends ParseOptions {
  /** Files modified more recently than this are treated as live: their trailing turn is skipped. */
  liveWindowMs?: number;
  onProgress?: (done: number, total: number) => void;
}

export interface IngestReport {
  filesSeen: number;
  filesRead: number;
  filesUnchanged: number;
  lines: number;
  turns: number;
  chunksAdded: number;
  chunksDuplicate: number;
  skipped: Record<string, number>;
  redactions: RedactionCounts;
}

/** Human messages shorter than this ("yes", "continue") carry nothing worth recalling. */
const MIN_HUMAN_CHARS = 16;
const MIN_ASSISTANT_CHARS = 40;

export function* walkJsonl(dir: string): Generator<string> {
  let entries: fs.Dirent[];
  try {
    entries = fs.readdirSync(dir, { withFileTypes: true });
  } catch {
    return;
  }
  for (const e of entries) {
    const p = path.join(dir, e.name);
    if (e.isSymbolicLink()) continue; // never follow links out of the source tree
    if (e.isDirectory()) yield* walkJsonl(p);
    else if (e.isFile() && e.name.endsWith('.jsonl')) yield p;
  }
}

export function sha256(text: string): string {
  return createHash('sha256').update(text).digest('hex');
}

/**
 * Redacts, chunks and stores the turns of one parsed source inside a single transaction.
 * Shared by every importer, so nothing reaches the store without passing through `redact`.
 */
function storeParsed(
  store: Store,
  parsed: ParsedFile,
  where: { projectKey: string; projectName: string; live: boolean },
  report: IngestReport,
  afterStore: () => void,
): RedactionCounts {
  const counts: RedactionCounts = {};
  const bump = (why: string) => {
    report.skipped[why] = (report.skipped[why] ?? 0) + 1;
  };
  store.transaction(() => {
    const sessionPk = new Map<string, number>();
    let projectId = -1;
    const ensureSession = (sessionId: string): number => {
      const hit = sessionPk.get(sessionId);
      if (hit !== undefined) return hit;
      if (projectId < 0) projectId = store.upsertProject(redact(where.projectKey).text, redact(where.projectName).text);
      const info = parsed.sessions.get(sessionId);
      const title = info?.title ? redact(info.title).text : null;
      const pk = store.upsertSession(sessionId, projectId, title, info?.firstTs ?? 0, info?.lastTs ?? 0);
      sessionPk.set(sessionId, pk);
      return pk;
    };

    for (const turn of parsed.turns) {
      // The last assistant turn of a live session may still be growing; pick it up next run.
      if (!turn.closed && where.live) {
        bump('live-turn');
        continue;
      }
      const min = turn.role === 'human' ? MIN_HUMAN_CHARS : MIN_ASSISTANT_CHARS;
      if (turn.text.trim().length < min) {
        bump('too-short');
        continue;
      }
      report.turns++;
      // Redact first: nothing below this line ever sees the original text.
      const clean = redact(turn.text);
      mergeCounts(counts, clean.counts);
      const pieces = chunkText(clean.text, turn.role === 'human' ? HUMAN_CHUNKS : ASSISTANT_CHUNKS);
      const pk = ensureSession(turn.sessionId);
      pieces.forEach((text, part) => {
        const added = store.insertChunk({ uuid: turn.uuid, part, sessionPk: pk, projectId, role: turn.role, ts: turn.ts, text, sha: sha256(`${turn.role}\n${text}`) });
        if (added) report.chunksAdded++;
        else report.chunksDuplicate++;
      });
    }
    store.addRedactions(counts);
    afterStore();
  });
  return counts;
}

function emptyReport(): IngestReport {
  return { filesSeen: 0, filesRead: 0, filesUnchanged: 0, lines: 0, turns: 0, chunksAdded: 0, chunksDuplicate: 0, skipped: {}, redactions: {} };
}

/**
 * Imports a ChatGPT data export: the zip, its unzipped folder, or a conversations JSON file.
 * Conversations land in one project named "ChatGPT". Safe to re-run with a newer export: turns
 * already stored are skipped, new ones are appended.
 */
export async function ingestChatGpt(store: Store, input: string): Promise<IngestReport> {
  const report = emptyReport();
  const resolvedInput = path.resolve(input);
  const st = fs.statSync(resolvedInput, { throwIfNoEntry: false });
  if (!st) throw new Error(`No ChatGPT export at ${input}`);
  const fileKey = sha256(`chatgpt:${resolvedInput}`);
  const stamp = { size: st.isDirectory() ? 0 : st.size, mtime: Math.floor(st.mtimeMs) };
  const prev = store.fileState(fileKey);
  report.filesSeen = 1;
  if (prev && !st.isDirectory() && prev.size === stamp.size && prev.mtime === stamp.mtime) {
    report.filesUnchanged = 1;
    return report;
  }
  const resolved = await resolveExport(resolvedInput);
  try {
    for (const file of resolved.files) {
      const parsed = await parseChatGptFile(file);
      report.filesRead++;
      report.lines += parsed.lines;
      for (const [k, v] of Object.entries(parsed.skipped)) report.skipped[k] = (report.skipped[k] ?? 0) + v;
      const counts = storeParsed(store, parsed, { projectKey: 'chatgpt-export', projectName: 'ChatGPT', live: false }, report, () => undefined);
      mergeCounts(report.redactions, counts);
    }
    store.setFileState(fileKey, stamp.size, stamp.mtime);
  } finally {
    resolved.cleanup();
  }
  return report;
}

export async function ingest(store: Store, sourceRoot: string, opts: IngestOptions = {}): Promise<IngestReport> {
  const liveWindowMs = opts.liveWindowMs ?? 5 * 60_000;
  const report = emptyReport();

  const files: { file: string; size: number; mtime: number }[] = [];
  for (const file of walkJsonl(sourceRoot)) {
    const st = fs.statSync(file);
    files.push({ file, size: st.size, mtime: Math.floor(st.mtimeMs) });
  }
  // Oldest first, so when the same content appears twice the earliest copy is the one kept.
  files.sort((a, b) => a.mtime - b.mtime || a.file.localeCompare(b.file));
  report.filesSeen = files.length;

  const now = Date.now();
  let done = 0;
  for (const f of files) {
    done++;
    opts.onProgress?.(done, files.length);
    const rel = path.relative(sourceRoot, f.file);
    // Bookkeeping is keyed by a hash of the path, so directory names (which encode the home
    // path) are never written to the store.
    const fileKey = sha256(rel);
    const prev = store.fileState(fileKey);
    if (prev && prev.size === f.size && prev.mtime === f.mtime) {
      report.filesUnchanged++;
      continue;
    }
    const parsed = await parseSessionFile(f.file, opts);
    report.filesRead++;
    report.lines += parsed.lines;
    for (const [k, v] of Object.entries(parsed.skipped)) report.skipped[k] = (report.skipped[k] ?? 0) + v;

    const live = now - f.mtime < liveWindowMs;
    const projectKeyRaw = rel.split(path.sep)[0] ?? 'unknown';
    const cwd = [...parsed.sessions.values()].find((s) => s.cwd)?.cwd ?? null;
    const fileCounts = storeParsed(store, parsed, { projectKey: projectKeyRaw, projectName: cwd ? path.basename(cwd) : projectKeyRaw, live }, report, () => {
      // A live file is re-read next time whatever its size, so its trailing turn is not lost.
      store.setFileState(fileKey, live ? -1 : f.size, f.mtime);
    });
    mergeCounts(report.redactions, fileCounts);
  }
  return report;
}
