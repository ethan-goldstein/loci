import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { afterAll, describe, expect, it } from 'vitest';
import { activeBranch, conversationTurns, parseChatGptFile, resolveExport, streamJsonArray } from '../../src/ingest/chatgpt.ts';
import type { ExportConversation } from '../../src/ingest/chatgpt.ts';
import { ingestChatGpt } from '../../src/ingest/run.ts';
import { Store } from '../../src/ingest/store.ts';

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'loci-chatgpt-test-'));
afterAll(() => fs.rmSync(tmp, { recursive: true, force: true }));

// Invented for the test; not a real credential.
const SECRET = 'sk-proj-' + 'Qw7'.repeat(14);

type Msg = { id: string; role: string; text?: string; t?: number; type?: string; recipient?: string; hidden?: boolean; parts?: unknown[] };

/** Builds a conversation in the export's tree shape from a parent -> children outline. */
function conversation(id: string, title: string, created: number, nodes: (Msg & { parent: string | null })[], current: string): ExportConversation {
  const mapping: NonNullable<ExportConversation['mapping']> = {};
  for (const n of nodes) {
    mapping[n.id] = {
      id: n.id,
      parent: n.parent,
      children: nodes.filter((c) => c.parent === n.id).map((c) => c.id),
      message:
        n.role === 'root'
          ? null
          : {
              id: n.id,
              author: { role: n.role },
              create_time: n.t ?? null,
              content: { content_type: n.type ?? 'text', parts: n.parts ?? [n.text ?? ''] },
              recipient: n.recipient ?? 'all',
              metadata: n.hidden ? { is_visually_hidden_from_conversation: true } : {},
            },
    };
  }
  return { conversation_id: id, title, create_time: created, update_time: created + 100, mapping, current_node: current };
}

const T0 = Date.UTC(2025, 3, 2, 10, 0, 0) / 1000;
const convA = conversation(
  'conv-a',
  'Planning a tide logger',
  T0,
  [
    { id: 'root', role: 'root', parent: null },
    { id: 'sys', role: 'system', text: 'You are ChatGPT', parent: 'root', hidden: true, t: T0 },
    { id: 'ctx', role: 'user', type: 'user_editable_context', text: 'profile text that is not a prompt', parent: 'sys', hidden: true, t: T0 },
    { id: 'u1', role: 'user', text: `How should I log tide heights from a pressure sensor? My key is ${SECRET}`, parent: 'ctx', t: T0 + 1 },
    // An abandoned branch: the first answer was regenerated.
    { id: 'a1-old', role: 'assistant', text: 'An earlier answer that was thrown away by regenerating.', parent: 'u1', t: T0 + 2 },
    { id: 'a1-think', role: 'assistant', type: 'thoughts', text: 'private reasoning', parent: 'u1', t: T0 + 3 },
    { id: 'a1-tool', role: 'assistant', type: 'code', text: 'search("tide logger")', recipient: 'web', parent: 'a1-think', t: T0 + 4 },
    { id: 't1', role: 'tool', text: 'raw tool output that must not be kept', parent: 'a1-tool', t: T0 + 5 },
    { id: 'a1', role: 'assistant', text: 'Sample the pressure sensor once a minute and subtract barometric pressure.', parent: 't1', t: T0 + 6 },
    { id: 'a1b', role: 'assistant', text: 'Then convert the residual pressure to water depth with the hydrostatic equation.', parent: 'a1', t: T0 + 7 },
    { id: 'u2', role: 'user', type: 'multimodal_text', parts: [{ content_type: 'image_asset_pointer', asset_pointer: 'file-service://x' }, 'Here is a photo of the casing, will this seal hold underwater?'], parent: 'a1b', t: T0 + 8 },
    { id: 'a2', role: 'assistant', text: 'The gasket looks seated, but add a second O-ring before deploying it.', parent: 'u2', t: T0 + 9 },
  ],
  'a2',
);
const convB = conversation(
  'conv-b',
  'Sourdough "hydration" {with braces} [and brackets]',
  T0 + 5000,
  [
    { id: 'rootb', role: 'root', parent: null },
    { id: 'ub', role: 'user', text: 'What hydration should a beginner sourdough use? Answer with "quotes", a \\ backslash and } braces ]', parent: 'rootb', t: T0 + 5001 },
    { id: 'ab', role: 'assistant', text: 'Start near 70 percent {hydration} and adjust: it is "forgiving" at that level.', parent: 'ub', t: T0 + 5002 },
  ],
  'ab',
);
const convEmpty = conversation('conv-empty', 'Nothing kept', T0 + 9000, [{ id: 'rootc', role: 'root', parent: null }, { id: 'sc', role: 'system', text: 'only a system line', parent: 'rootc' }], 'sc');

const exportDir = path.join(tmp, 'export');
fs.mkdirSync(exportDir, { recursive: true });
const jsonFile = path.join(exportDir, 'conversations.json');
fs.writeFileSync(jsonFile, JSON.stringify([convA, convB, convEmpty, { not: 'a conversation' }], null, 1));
fs.writeFileSync(path.join(exportDir, 'chat.html'), '<html>the export also ships a big HTML file</html>');
fs.writeFileSync(path.join(exportDir, 'user.json'), '{"email":"someone@example-mail.com"}');

describe('streamJsonArray', () => {
  it('yields each element whatever the chunk boundaries', async () => {
    const out: unknown[] = [];
    for await (const item of streamJsonArray(jsonFile)) out.push(item);
    expect(out).toEqual([convA, convB, convEmpty, { not: 'a conversation' }]);
  });

  it('survives elements much larger than the read buffer and tricky strings', async () => {
    const big = [{ a: 'x'.repeat(3_000_000), nested: [{ s: '}]"\\' }, [1, 2, [3]]] }, { b: '\\"]}{[' }, [], {}];
    const file = path.join(tmp, 'big.json');
    fs.writeFileSync(file, `${String.fromCharCode(0xfeff)}  ${JSON.stringify(big)}\n`); // leading byte-order mark
    const out: unknown[] = [];
    for await (const item of streamJsonArray(file)) out.push(item);
    expect(out).toEqual(big);
  });

  it('rejects files that are not a complete JSON array', async () => {
    const drain = async (text: string) => {
      const file = path.join(tmp, `bad-${Math.abs(text.length * 31 + text.charCodeAt(0))}.json`);
      fs.writeFileSync(file, text);
      for await (const _ of streamJsonArray(file)) void _;
    };
    await expect(drain('{"a": 1}')).rejects.toThrow(/array/);
    await expect(drain('[{"a": 1}, {"b": ')).rejects.toThrow(/ended early/);
    await expect(drain('   ')).rejects.toThrow(/Not a JSON array/);
  });
});

describe('conversation parsing', () => {
  it('follows the branch the conversation ended on', () => {
    const ids = activeBranch(convA).map((m) => m.id);
    expect(ids).toEqual(['sys', 'ctx', 'u1', 'a1-think', 'a1-tool', 't1', 'a1', 'a1b', 'u2', 'a2']);
    expect(ids).not.toContain('a1-old');
    // Without current_node, the most recent leaf is used.
    expect(activeBranch({ ...convA, current_node: null }).at(-1)!.id).toBe('a2');
  });

  it('keeps typed prompts and assistant prose, and nothing else', () => {
    const { turns, session } = conversationTurns(convA)!;
    expect(turns.map((t) => [t.role, t.uuid])).toEqual([
      ['human', 'chatgpt:u1'],
      ['assistant', 'chatgpt:a1'],
      ['human', 'chatgpt:u2'],
      ['assistant', 'chatgpt:a2'],
    ]);
    // Two consecutive assistant messages merge into one turn.
    expect(turns[1]!.text).toContain('once a minute');
    expect(turns[1]!.text).toContain('hydrostatic equation');
    // The image pointer is dropped, the words beside it are kept.
    expect(turns[2]!.text).toBe('Here is a photo of the casing, will this seal hold underwater?');
    const all = turns.map((t) => t.text).join('\n');
    for (const gone of ['You are ChatGPT', 'profile text', 'thrown away', 'private reasoning', 'search(', 'raw tool output']) expect(all).not.toContain(gone);
    expect(session).toMatchObject({ sessionId: 'chatgpt:conv-a', title: 'Planning a tide logger', firstTs: T0 * 1000 });
    expect(turns[0]!.ts).toBe((T0 + 1) * 1000);
    expect(turns.every((t) => t.closed)).toBe(true);
  });

  it('tolerates missing ids, missing times and parent cycles', () => {
    expect(conversationTurns({ title: 'no id' })).toBeNull();
    const cyclic: ExportConversation = { conversation_id: 'c', mapping: { a: { id: 'a', parent: 'b', message: { id: 'a', author: { role: 'user' }, content: { content_type: 'text', parts: ['a question with no timestamp at all'] } } }, b: { id: 'b', parent: 'a', message: null } }, current_node: 'a' };
    const out = conversationTurns(cyclic)!;
    expect(out.turns).toHaveLength(1);
    expect(Number.isFinite(out.turns[0]!.ts)).toBe(true);
  });

  it('parses a whole file and reports what it skipped', async () => {
    const parsed = await parseChatGptFile(jsonFile);
    expect(parsed.lines).toBe(4);
    expect(parsed.sessions.size).toBe(2);
    expect(parsed.turns).toHaveLength(6);
    expect(parsed.skipped).toEqual({ 'empty-conversation': 1, 'no-id': 1 });
  });
});

describe('ingestChatGpt', () => {
  const zipAvailable = (() => {
    try {
      execFileSync('zip', ['-v'], { stdio: 'ignore' });
      execFileSync('unzip', ['-v'], { stdio: 'ignore' });
      return true;
    } catch {
      return false;
    }
  })();

  it('stores redacted chunks under one project, from the folder or the JSON file', async () => {
    const store = new Store(path.join(tmp, 'a.db'));
    const report = await ingestChatGpt(store, exportDir);
    expect(report.chunksAdded).toBe(6);
    expect(report.redactions['vendor-key']).toBe(1);
    expect(store.projects().map((p) => p.name)).toEqual(['ChatGPT']);
    expect(store.sessions().map((s) => s.title)).toEqual(['Planning a tide logger', 'Sourdough "hydration" {with braces} [and brackets]']);
    const texts = store.allChunks().map((c) => c.text).join('\n');
    expect(texts).not.toContain(SECRET);
    expect(texts).toContain('[REDACTED:vendor-key]');
    expect(texts).not.toContain('example-mail.com'); // user.json is never read

    // Re-running adds nothing; a single JSON file works too and is recognised as the same content.
    expect((await ingestChatGpt(store, exportDir)).chunksAdded).toBe(0);
    const again = await ingestChatGpt(store, jsonFile);
    expect(again.chunksAdded).toBe(0);
    expect(again.chunksDuplicate).toBe(6);
    expect((await ingestChatGpt(store, jsonFile)).filesUnchanged).toBe(1);
    store.close();
  });

  it.skipIf(!zipAvailable)('reads the export zip, extracting only the conversation JSON', async () => {
    const zip = path.join(tmp, 'export.zip');
    execFileSync('zip', ['-q', '-r', zip, '.'], { cwd: exportDir });
    const resolved = await resolveExport(zip);
    const dir = path.dirname(resolved.files[0]!);
    expect(fs.readdirSync(dir)).toEqual(['conversations.json']); // not chat.html, not user.json
    expect(fs.statSync(dir).mode & 0o777).toBe(0o700);
    resolved.cleanup();
    expect(fs.existsSync(dir)).toBe(false);

    const store = new Store(path.join(tmp, 'z.db'));
    expect((await ingestChatGpt(store, zip)).chunksAdded).toBe(6);
    store.close();
  });

  it('gives a clear error for the wrong kind of input', async () => {
    const store = new Store(path.join(tmp, 'e.db'));
    await expect(ingestChatGpt(store, path.join(tmp, 'missing.zip'))).rejects.toThrow(/No ChatGPT export/);
    fs.writeFileSync(path.join(tmp, 'notes.txt'), 'hello');
    await expect(ingestChatGpt(store, path.join(tmp, 'notes.txt'))).rejects.toThrow(/Expected a ChatGPT export/);
    fs.mkdirSync(path.join(tmp, 'emptydir'));
    await expect(ingestChatGpt(store, path.join(tmp, 'emptydir'))).rejects.toThrow(/no conversations\.json/);
    store.close();
  });
});
