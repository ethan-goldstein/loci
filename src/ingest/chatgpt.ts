import { execFile } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { promisify } from 'node:util';
import type { ParsedFile, SessionInfo, Turn } from './parse.ts';

/**
 * Importer for a ChatGPT data export (Settings → Data controls → Export data).
 *
 * The export is a zip whose `conversations.json` is one large JSON array. Each conversation is a
 * tree of messages (`mapping`), because editing a prompt or regenerating a reply forks it; the
 * branch the person ended on is found by walking up from `current_node`.
 *
 * As with Claude Code transcripts, only two things are kept: what the person typed and what the
 * assistant said back in prose. System messages, tool calls and their output, hidden context,
 * reasoning traces and images are dropped. Text returned here is NOT yet redacted.
 */

const execFileP = promisify(execFile);

interface ExportMessage {
  id?: string;
  author?: { role?: string };
  create_time?: number | null;
  content?: { content_type?: string; parts?: unknown[] };
  recipient?: string;
  metadata?: { is_visually_hidden_from_conversation?: boolean; is_user_system_message?: boolean };
}

interface ExportNode {
  id?: string;
  message?: ExportMessage | null;
  parent?: string | null;
  children?: string[];
}

export interface ExportConversation {
  id?: string;
  conversation_id?: string;
  title?: string | null;
  create_time?: number | null;
  update_time?: number | null;
  mapping?: Record<string, ExportNode>;
  current_node?: string | null;
}

/**
 * Yields the elements of a top-level JSON array one at a time, without ever holding the whole
 * file in memory. Exports can run to hundreds of megabytes, past what `JSON.parse` can take as a
 * single string. Tracks string and escape state so braces inside text do not confuse the depth.
 */
export async function* streamJsonArray(file: string): AsyncGenerator<unknown> {
  const stream = fs.createReadStream(file, { encoding: 'utf8', highWaterMark: 1 << 20 });
  let depth = 0;
  let inString = false;
  let escaped = false;
  let started = false;
  /** Pieces of the element being read that arrived in earlier chunks. */
  let carry: string[] = [];
  for await (const chunk of stream as AsyncIterable<string>) {
    // Where the current element begins within this chunk: 0 if it started in an earlier one.
    let start = depth >= 2 ? 0 : -1;
    for (let i = 0; i < chunk.length; i++) {
      const c = chunk.charCodeAt(i);
      if (inString) {
        if (escaped) escaped = false;
        else if (c === 92) escaped = true; // backslash
        else if (c === 34) inString = false; // closing quote
        continue;
      }
      if (c === 34) {
        inString = true;
      } else if (c === 91 || c === 123) {
        // [ or {
        if (depth === 0) {
          if (c !== 91) throw new Error('Expected a JSON array at the top level');
          started = true;
        } else if (depth === 1) start = i; // an element begins
        depth++;
      } else if (c === 93 || c === 125) {
        // ] or }
        depth--;
        if (depth < 0) throw new Error('Unbalanced JSON');
        if (depth === 1) {
          carry.push(chunk.slice(start, i + 1));
          yield JSON.parse(carry.join(''));
          carry = [];
          start = -1;
        }
      }
    }
    if (start >= 0) carry.push(chunk.slice(start)); // the element continues into the next chunk
  }
  if (!started) throw new Error('Not a JSON array');
  if (depth !== 0 || inString) throw new Error('JSON ended early');
}

/** The messages on the branch the conversation ended on, oldest first. */
export function activeBranch(conv: ExportConversation): ExportMessage[] {
  const mapping = conv.mapping ?? {};
  let leaf = conv.current_node ?? null;
  if (!leaf || !mapping[leaf]) {
    // No recorded end point: take the most recently created leaf.
    let best = -Infinity;
    for (const [id, node] of Object.entries(mapping)) {
      if ((node.children ?? []).length > 0) continue;
      const t = node.message?.create_time ?? 0;
      if (t >= best) {
        best = t;
        leaf = id;
      }
    }
  }
  const out: ExportMessage[] = [];
  const seen = new Set<string>();
  while (leaf && mapping[leaf] && !seen.has(leaf)) {
    seen.add(leaf); // guards against a malformed export with a parent cycle
    const node = mapping[leaf]!;
    if (node.message) out.push(node.message);
    leaf = node.parent ?? null;
  }
  return out.reverse();
}

function textOf(message: ExportMessage): string {
  const type = message.content?.content_type;
  if (type !== 'text' && type !== 'multimodal_text') return '';
  const parts = message.content?.parts;
  if (!Array.isArray(parts)) return '';
  // Multimodal parts mix strings with image pointers; keep the words only.
  return parts
    .filter((p): p is string => typeof p === 'string')
    .join('\n\n')
    .trim();
}

/** Converts one exported conversation into turns and a session, in the shape the ingest expects. */
export function conversationTurns(conv: ExportConversation): { turns: Turn[]; session: SessionInfo } | null {
  const rawId = conv.conversation_id ?? conv.id;
  if (typeof rawId !== 'string' || rawId.length === 0) return null;
  const sessionId = `chatgpt:${rawId}`;
  const started = typeof conv.create_time === 'number' ? conv.create_time * 1000 : Number.NaN;
  const turns: Turn[] = [];
  let open: { uuid: string; ts: number; parts: string[] } | null = null;
  let lastTs = Number.isFinite(started) ? started : 0;

  const flush = () => {
    if (!open) return;
    const text = open.parts.join('\n\n').trim();
    if (text) turns.push({ uuid: open.uuid, role: 'assistant', sessionId, ts: open.ts, text, closed: true });
    open = null;
  };

  for (const m of activeBranch(conv)) {
    const role = m.author?.role;
    if (typeof m.id !== 'string') continue;
    if (typeof m.create_time === 'number') lastTs = Math.max(lastTs, m.create_time * 1000);
    const ts = typeof m.create_time === 'number' ? m.create_time * 1000 : lastTs;
    if (role === 'user') {
      flush(); // any user-side message ends the assistant's turn
      if (m.metadata?.is_visually_hidden_from_conversation || m.metadata?.is_user_system_message) continue;
      const text = textOf(m);
      if (text) turns.push({ uuid: `chatgpt:${m.id}`, role: 'human', sessionId, ts, text, closed: true });
    } else if (role === 'assistant') {
      // `recipient` other than "all" means the assistant was addressing a tool, not the person.
      if ((m.recipient ?? 'all') !== 'all' || m.metadata?.is_visually_hidden_from_conversation) continue;
      const text = textOf(m);
      if (!text) continue;
      if (open) open.parts.push(text);
      else open = { uuid: `chatgpt:${m.id}`, ts, parts: [text] };
    }
    // system and tool messages are dropped
  }
  flush();

  const times = turns.map((t) => t.ts).filter(Number.isFinite);
  const firstTs = Number.isFinite(started) ? started : times.length ? Math.min(...times) : 0;
  const title = typeof conv.title === 'string' && conv.title.trim() ? conv.title.trim().slice(0, 200) : null;
  return { turns, session: { sessionId, title, cwd: null, entrypoint: 'chatgpt-export', firstTs, lastTs: times.length ? Math.max(firstTs, ...times) : firstTs } };
}

export interface ResolvedExport {
  /** The conversation files to read, in name order. */
  files: string[];
  /** Removes anything extracted to a temporary directory. Always call it. */
  cleanup: () => void;
}

/**
 * Accepts the export zip, the unzipped folder, or a conversations JSON file. A zip is opened with
 * the system `unzip`, extracting ONLY the conversation JSON into a fresh private directory: the
 * rest of the archive (images, HTML) is never written to disk, and `unzip` refuses entries that
 * would escape the target directory.
 */
export async function resolveExport(input: string): Promise<ResolvedExport> {
  const target = path.resolve(input);
  let st: fs.Stats;
  try {
    st = fs.statSync(target);
  } catch {
    throw new Error(`No ChatGPT export at ${input}`);
  }
  const isConversations = (name: string) => /^conversations(-\d+)?\.json$/.test(name);
  if (st.isDirectory()) {
    const files = fs.readdirSync(target).filter(isConversations).sort().map((f) => path.join(target, f));
    if (files.length === 0) throw new Error('That folder has no conversations.json. Point at the unzipped ChatGPT export.');
    return { files, cleanup: () => undefined };
  }
  if (target.toLowerCase().endsWith('.json')) return { files: [target], cleanup: () => undefined };
  if (!target.toLowerCase().endsWith('.zip')) throw new Error('Expected a ChatGPT export .zip, its unzipped folder, or conversations.json');

  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'loci-chatgpt-'));
  fs.chmodSync(dir, 0o700);
  const cleanup = () => fs.rmSync(dir, { recursive: true, force: true });
  try {
    // -j drops any directory part of the stored names, so nothing lands outside `dir`.
    await execFileP('unzip', ['-o', '-qq', '-j', target, 'conversations.json', 'conversations-*.json', '-d', dir], { maxBuffer: 1 << 20 }).catch((err: unknown) => {
      // unzip exits 11 when one of the two patterns matched nothing, which is expected.
      if ((err as { code?: number | string }).code !== 11) throw err;
    });
    const files = fs.readdirSync(dir).filter(isConversations).sort().map((f) => path.join(dir, f));
    if (files.length === 0) throw new Error('That zip has no conversations.json. Is it a ChatGPT data export?');
    return { files, cleanup };
  } catch (err) {
    cleanup();
    if ((err as NodeJS.ErrnoException).code === 'ENOENT') throw new Error('Could not run `unzip`. Unzip the export yourself and pass the folder instead.', { cause: err });
    throw err instanceof Error && err.message.startsWith('That zip') ? err : new Error('Could not read that zip as a ChatGPT export', { cause: err });
  }
}

/** Reads one conversations file into the same shape the Claude Code parser produces. */
export async function parseChatGptFile(file: string): Promise<ParsedFile> {
  const turns: Turn[] = [];
  const sessions = new Map<string, SessionInfo>();
  const skipped: Record<string, number> = {};
  let lines = 0;
  for await (const item of streamJsonArray(file)) {
    lines++;
    if (!item || typeof item !== 'object') {
      skipped['not-a-conversation'] = (skipped['not-a-conversation'] ?? 0) + 1;
      continue;
    }
    const parsed = conversationTurns(item as ExportConversation);
    if (!parsed) {
      skipped['no-id'] = (skipped['no-id'] ?? 0) + 1;
      continue;
    }
    if (parsed.turns.length === 0) {
      skipped['empty-conversation'] = (skipped['empty-conversation'] ?? 0) + 1;
      continue;
    }
    sessions.set(parsed.session.sessionId, parsed.session);
    for (const t of parsed.turns) turns.push(t);
  }
  return { turns, sessions, lines, skipped };
}
