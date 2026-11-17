import fs from 'node:fs';
import readline from 'node:readline';

/**
 * Stream parser for Claude Code session transcripts (one JSON object per line).
 *
 * Keeps two things only: what the person typed, and what the assistant said back in prose.
 * Tool calls, tool results, thinking blocks, attachments, system reminders, compaction summaries
 * and machine-originated prompts are dropped, so the corpus is the conversation rather than the
 * 1.3 GB of file dumps around it.
 */

export type Role = 'human' | 'assistant';

export interface Turn {
  /** uuid of the human line, or of the first assistant line of the turn. Stable across re-reads. */
  uuid: string;
  role: Role;
  sessionId: string;
  /** Milliseconds since the epoch. */
  ts: number;
  /** Raw text. Not yet redacted: callers must redact before storing. */
  text: string;
  /** False for the trailing assistant turn of a file, which may still be growing. */
  closed: boolean;
}

export interface SessionInfo {
  sessionId: string;
  title: string | null;
  cwd: string | null;
  entrypoint: string | null;
  firstTs: number;
  lastTs: number;
}

export interface ParsedFile {
  turns: Turn[];
  sessions: Map<string, SessionInfo>;
  lines: number;
  /** Why lines were dropped, by reason. Counts only. */
  skipped: Record<string, number>;
}

export interface ParseOptions {
  /** Include one-shot automated runs (`entrypoint: "sdk-cli"`). Off by default. */
  includeAutomated?: boolean;
  /** Include subagent side conversations. Off by default. */
  includeSubagents?: boolean;
}

/** Wrapper blocks the harness injects into "user" text. Matched by tag-name prefix. */
const WRAPPER_TAGS = [
  'system-reminder',
  'pasted_content',
  'task-notification',
  'local-command-stdout',
  'local-command-stderr',
  'local-command-caveat',
  'command-name',
  'command-message',
  'command-args',
  'bash-input',
  'bash-stdout',
  'bash-stderr',
  'scheduled-task',
  'user-prompt-submit-hook',
  'fork-boilerplate',
  'ide_opened_file',
  'ide_selection',
  'side-sessions-event',
  'ci-monitor-event',
];

const WRAPPER_RE = new RegExp(
  `<((?:${WRAPPER_TAGS.join('|')})[A-Za-z0-9_-]*)(?:\\s[^>]*)?>[\\s\\S]*?<\\/\\1(?:\\s[^>]*)?>`,
  'g',
);
const WRAPPER_OPEN_RE = new RegExp(`<(?:${WRAPPER_TAGS.join('|')})[A-Za-z0-9_-]*(?:\\s[^>]*)?>`);

/**
 * Removes harness-injected wrapper blocks from a user message, leaving only the person's own
 * words. If an opening wrapper tag survives (unbalanced or truncated block), everything from that
 * tag onward is dropped rather than risk keeping machine text.
 */
export function stripWrappers(text: string): string {
  let out = text.replace(WRAPPER_RE, '\n');
  const open = WRAPPER_OPEN_RE.exec(out);
  if (open) out = out.slice(0, open.index);
  return out.replace(/\n{3,}/g, '\n\n').trim();
}

const INTERRUPT_RE = /^\[Request interrupted/;

interface ContentBlock {
  type?: string;
  text?: unknown;
}

interface RawLine {
  type?: string;
  uuid?: string;
  sessionId?: string;
  timestamp?: string;
  cwd?: string;
  entrypoint?: string;
  isSidechain?: boolean;
  isMeta?: boolean;
  isCompactSummary?: boolean;
  isVisibleInTranscriptOnly?: boolean;
  isApiErrorMessage?: boolean;
  origin?: { kind?: string };
  aiTitle?: string;
  customTitle?: string;
  message?: { model?: string; content?: string | ContentBlock[] };
}

function textOf(content: string | ContentBlock[] | undefined): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const b of content) {
    if (b && b.type === 'text' && typeof b.text === 'string') parts.push(b.text);
  }
  return parts.join('\n\n');
}

/** A cheap pre-filter: only these line types can contribute, so skip JSON.parse on the rest. */
function mightMatter(line: string): boolean {
  return (
    line.includes('"type":"user"') ||
    line.includes('"type":"assistant"') ||
    line.includes('"type":"ai-title"') ||
    line.includes('"type":"custom-title"')
  );
}

export async function parseSessionFile(file: string, opts: ParseOptions = {}): Promise<ParsedFile> {
  const turns: Turn[] = [];
  const sessions = new Map<string, SessionInfo>();
  const skipped: Record<string, number> = {};
  const skip = (why: string) => {
    skipped[why] = (skipped[why] ?? 0) + 1;
  };
  /** Open assistant turn per session, waiting for the next human prompt to close it. */
  const open = new Map<string, { uuid: string; ts: number; parts: string[] }>();
  let lines = 0;

  const touch = (o: RawLine, ts: number): SessionInfo | null => {
    if (!o.sessionId) return null;
    let s = sessions.get(o.sessionId);
    if (!s) {
      s = { sessionId: o.sessionId, title: null, cwd: null, entrypoint: null, firstTs: ts, lastTs: ts };
      sessions.set(o.sessionId, s);
    }
    if (Number.isFinite(ts)) {
      if (!Number.isFinite(s.firstTs) || ts < s.firstTs) s.firstTs = ts;
      if (!Number.isFinite(s.lastTs) || ts > s.lastTs) s.lastTs = ts;
    }
    if (!s.cwd && typeof o.cwd === 'string') s.cwd = o.cwd;
    if (!s.entrypoint && typeof o.entrypoint === 'string') s.entrypoint = o.entrypoint;
    return s;
  };

  const flush = (sessionId: string, closed: boolean) => {
    const acc = open.get(sessionId);
    if (!acc) return;
    open.delete(sessionId);
    const text = acc.parts.join('\n\n').trim();
    if (text.length > 0) turns.push({ uuid: acc.uuid, role: 'assistant', sessionId, ts: acc.ts, text, closed });
  };

  const rl = readline.createInterface({ input: fs.createReadStream(file), crlfDelay: Infinity });
  for await (const line of rl) {
    if (line.length === 0) continue;
    lines++;
    if (!mightMatter(line)) continue;
    // Tool results are ~80% of the bytes on disk and never kept: skip before parsing.
    if (line.includes('"sourceToolAssistantUUID":')) {
      skip('tool-result');
      continue;
    }
    let o: RawLine;
    try {
      o = JSON.parse(line) as RawLine;
    } catch {
      skip('unparseable');
      continue;
    }

    if (o.type === 'ai-title' || o.type === 'custom-title') {
      const title = o.type === 'ai-title' ? o.aiTitle : o.customTitle;
      if (o.sessionId && typeof title === 'string' && title.trim()) {
        const s = touch(o, Number.NaN);
        // A custom title is the person's own choice, so it wins over the generated one.
        if (s && (o.type === 'custom-title' || !s.title)) s.title = title.trim().slice(0, 200);
      }
      continue;
    }
    if (o.type !== 'user' && o.type !== 'assistant') continue;
    if (!o.uuid || !o.sessionId) {
      skip('no-id');
      continue;
    }
    if (o.entrypoint === 'sdk-cli' && !opts.includeAutomated) {
      skip('automated');
      continue;
    }
    if (o.isSidechain && !opts.includeSubagents) {
      skip('subagent');
      continue;
    }
    const ts = Date.parse(o.timestamp ?? '');
    if (!Number.isFinite(ts)) {
      skip('no-timestamp');
      continue;
    }
    touch(o, ts);

    if (o.type === 'assistant') {
      if (o.isApiErrorMessage || o.message?.model === '<synthetic>') {
        skip('api-error');
        continue;
      }
      const text = textOf(o.message?.content).trim();
      if (text.length === 0) continue; // tool_use or thinking only
      const acc = open.get(o.sessionId);
      if (acc) acc.parts.push(text);
      else open.set(o.sessionId, { uuid: o.uuid, ts, parts: [text] });
      continue;
    }

    // A user line that is not a tool result starts a new turn, whoever wrote it.
    flush(o.sessionId, true);

    if (o.isMeta || o.isCompactSummary || o.isVisibleInTranscriptOnly) {
      skip('meta');
      continue;
    }
    if (o.isSidechain) {
      skip('subagent-prompt'); // written by the parent agent, not the person
      continue;
    }
    if (o.origin?.kind !== undefined && o.origin.kind !== 'human') {
      skip('machine-origin');
      continue;
    }
    const text = stripWrappers(textOf(o.message?.content));
    if (text.length === 0) {
      skip('wrapper-only');
      continue;
    }
    if (INTERRUPT_RE.test(text)) {
      skip('interrupt');
      continue;
    }
    turns.push({ uuid: o.uuid, role: 'human', sessionId: o.sessionId, ts, text, closed: true });
  }

  for (const sessionId of [...open.keys()]) flush(sessionId, false);
  return { turns, sessions, lines, skipped };
}
