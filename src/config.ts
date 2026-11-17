import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

export interface Config {
  /** Private store directory. Never inside a git repository. */
  home: string;
  /** Where Claude Code keeps session transcripts. */
  source: string;
  ollamaUrl: string;
  embedModel: string;
  port: number;
  /** Optional path to a ChatGPT data export (zip, unzipped folder, or conversations.json). */
  chatgptExport?: string;
}

export function expandHome(p: string): string {
  if (p === '~') return os.homedir();
  if (p.startsWith('~/')) return path.join(os.homedir(), p.slice(2));
  return p;
}

/** True when `dir` or any ancestor contains a `.git` entry. */
export function insideGitRepo(dir: string): boolean {
  let cur = path.resolve(dir);
  for (;;) {
    if (fs.existsSync(path.join(cur, '.git'))) return true;
    const parent = path.dirname(cur);
    if (parent === cur) return false;
    cur = parent;
  }
}

function loopbackUrl(raw: string): string {
  const u = new URL(raw);
  const host = u.hostname.replace(/^\[|\]$/g, '');
  if (host !== '127.0.0.1' && host !== 'localhost' && host !== '::1') {
    // Embedding sends chunk text to this endpoint, so it must stay on this machine.
    throw new Error(`LOCI_OLLAMA_URL must be a loopback address, got host "${u.hostname}"`);
  }
  return u.origin;
}

export function loadConfig(env: NodeJS.ProcessEnv = process.env): Config {
  const home = path.resolve(expandHome(env.LOCI_HOME ?? '~/.loci'));
  const source = path.resolve(expandHome(env.LOCI_SOURCE ?? '~/.claude/projects'));
  const port = Number(env.LOCI_PORT ?? 4319);
  if (!Number.isInteger(port) || port < 1024 || port > 65535) {
    throw new Error(`LOCI_PORT must be an integer between 1024 and 65535, got "${env.LOCI_PORT}"`);
  }
  const embedModel = env.LOCI_EMBED_MODEL ?? 'nomic-embed-text';
  if (!/^[A-Za-z0-9._:/-]{1,80}$/.test(embedModel)) {
    throw new Error('LOCI_EMBED_MODEL contains unexpected characters');
  }
  return {
    home,
    source,
    ollamaUrl: loopbackUrl(env.LOCI_OLLAMA_URL ?? 'http://127.0.0.1:11434'),
    embedModel,
    port,
    chatgptExport: env.LOCI_CHATGPT_EXPORT ? path.resolve(expandHome(env.LOCI_CHATGPT_EXPORT)) : undefined,
  };
}

/** Creates the private store directory (mode 700) and refuses to use one inside a git repo. */
export function ensureHome(home: string): void {
  if (insideGitRepo(home)) {
    throw new Error(`Refusing to keep private data inside a git repository: ${home}`);
  }
  fs.mkdirSync(home, { recursive: true, mode: 0o700 });
  fs.chmodSync(home, 0o700);
}

/** File-name-safe tag for a model name, used in vectors.<tag>.f32 and hnsw.<tag>.bin. */
export function modelTag(model: string): string {
  return model.replace(/[^A-Za-z0-9._-]+/g, '_');
}

export function paths(cfg: Config) {
  const tag = modelTag(cfg.embedModel);
  return {
    db: path.join(cfg.home, 'loci.db'),
    vectors: path.join(cfg.home, `vectors.${tag}.f32`),
    hnsw: path.join(cfg.home, `hnsw.${tag}.bin`),
  };
}
