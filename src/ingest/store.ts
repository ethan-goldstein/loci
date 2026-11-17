import { DatabaseSync } from 'node:sqlite';
import type { StatementSync } from 'node:sqlite';
import type { Role } from './parse.ts';
import type { RedactionCounts } from './redact.ts';

/**
 * The private store. One SQLite file holding redacted chunks and their bookkeeping.
 *
 * Chunks are append-only and their ids are contiguous from 1, so chunk `id` always lives at row
 * `id - 1` of the vectors file and is node `id - 1` of the index. Nothing is ever deleted or
 * renumbered; `loci reset` starts over.
 */

export interface ChunkRow {
  id: number;
  uuid: string;
  part: number;
  sessionPk: number;
  projectId: number;
  role: Role;
  ts: number;
  text: string;
}

export interface NewChunk {
  uuid: string;
  part: number;
  sessionPk: number;
  projectId: number;
  role: Role;
  ts: number;
  text: string;
  sha: string;
}

export interface SessionRow {
  pk: number;
  sessionId: string;
  projectId: number;
  title: string | null;
  firstTs: number;
  lastTs: number;
}

export interface ProjectRow {
  id: number;
  key: string;
  name: string;
}

const SCHEMA = `
CREATE TABLE IF NOT EXISTS projects(
  id INTEGER PRIMARY KEY,
  key TEXT NOT NULL UNIQUE,
  name TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions(
  pk INTEGER PRIMARY KEY,
  session_id TEXT NOT NULL UNIQUE,
  project_id INTEGER NOT NULL REFERENCES projects(id),
  title TEXT,
  first_ts INTEGER NOT NULL,
  last_ts INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks(
  id INTEGER PRIMARY KEY,
  uuid TEXT NOT NULL,
  part INTEGER NOT NULL,
  session_pk INTEGER NOT NULL REFERENCES sessions(pk),
  project_id INTEGER NOT NULL REFERENCES projects(id),
  role TEXT NOT NULL CHECK (role IN ('human','assistant')),
  ts INTEGER NOT NULL,
  text TEXT NOT NULL,
  sha TEXT NOT NULL UNIQUE,
  UNIQUE(uuid, part)
);
CREATE INDEX IF NOT EXISTS chunks_session ON chunks(session_pk, ts);
CREATE INDEX IF NOT EXISTS chunks_project_ts ON chunks(project_id, ts);
CREATE TABLE IF NOT EXISTS ingest_files(
  path TEXT PRIMARY KEY,
  size INTEGER NOT NULL,
  mtime INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS redactions(
  family TEXT PRIMARY KEY,
  count INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS meta(
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, tokenize='unicode61');
`;

export class Store {
  readonly db: DatabaseSync;
  readonly readOnly: boolean;
  private nextId = 1;
  private readonly stmts = new Map<string, StatementSync>();

  constructor(file: string, opts: { readOnly?: boolean } = {}) {
    this.readOnly = opts.readOnly ?? false;
    this.db = new DatabaseSync(file, { readOnly: this.readOnly });
    if (!this.readOnly) {
      this.db.exec('PRAGMA journal_mode = WAL; PRAGMA synchronous = NORMAL; PRAGMA foreign_keys = ON;');
      this.db.exec(SCHEMA);
    }
    this.nextId = this.chunkCount() + 1;
  }

  private q(sql: string): StatementSync {
    let s = this.stmts.get(sql);
    if (!s) {
      s = this.db.prepare(sql);
      this.stmts.set(sql, s);
    }
    return s;
  }

  transaction<T>(fn: () => T): T {
    this.db.exec('BEGIN');
    try {
      const out = fn();
      this.db.exec('COMMIT');
      return out;
    } catch (err) {
      this.db.exec('ROLLBACK');
      throw err;
    }
  }

  chunkCount(): number {
    const row = this.q('SELECT COALESCE(MAX(id), 0) AS n FROM chunks').get() as { n: number };
    return Number(row.n);
  }

  upsertProject(key: string, name: string): number {
    this.q('INSERT OR IGNORE INTO projects(key, name) VALUES (?, ?)').run(key, name);
    const row = this.q('SELECT id FROM projects WHERE key = ?').get(key) as { id: number };
    return Number(row.id);
  }

  upsertSession(sessionId: string, projectId: number, title: string | null, firstTs: number, lastTs: number): number {
    this.q(
      `INSERT INTO sessions(session_id, project_id, title, first_ts, last_ts) VALUES (?, ?, ?, ?, ?)
       ON CONFLICT(session_id) DO UPDATE SET
         title = COALESCE(excluded.title, sessions.title),
         first_ts = MIN(sessions.first_ts, excluded.first_ts),
         last_ts = MAX(sessions.last_ts, excluded.last_ts)`,
    ).run(sessionId, projectId, title, firstTs, lastTs);
    const row = this.q('SELECT pk FROM sessions WHERE session_id = ?').get(sessionId) as { pk: number };
    return Number(row.pk);
  }

  /** Appends a chunk unless the same `(uuid, part)` or the same content is already stored. */
  insertChunk(c: NewChunk): boolean {
    const dup = this.q('SELECT 1 FROM chunks WHERE (uuid = ? AND part = ?) OR sha = ? LIMIT 1').get(c.uuid, c.part, c.sha);
    if (dup) return false;
    const id = this.nextId++;
    this.q(
      'INSERT INTO chunks(id, uuid, part, session_pk, project_id, role, ts, text, sha) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
    ).run(id, c.uuid, c.part, c.sessionPk, c.projectId, c.role, c.ts, c.text, c.sha);
    this.q('INSERT INTO chunks_fts(rowid, text) VALUES (?, ?)').run(id, c.text);
    return true;
  }

  fileState(path: string): { size: number; mtime: number } | null {
    const row = this.q('SELECT size, mtime FROM ingest_files WHERE path = ?').get(path) as
      | { size: number; mtime: number }
      | undefined;
    return row ? { size: Number(row.size), mtime: Number(row.mtime) } : null;
  }

  setFileState(path: string, size: number, mtime: number): void {
    this.q(
      'INSERT INTO ingest_files(path, size, mtime) VALUES (?, ?, ?) ON CONFLICT(path) DO UPDATE SET size = excluded.size, mtime = excluded.mtime',
    ).run(path, size, mtime);
  }

  addRedactions(counts: RedactionCounts): void {
    for (const [family, n] of Object.entries(counts)) {
      this.q(
        'INSERT INTO redactions(family, count) VALUES (?, ?) ON CONFLICT(family) DO UPDATE SET count = count + excluded.count',
      ).run(family, n);
    }
  }

  redactions(): RedactionCounts {
    const out: RedactionCounts = {};
    for (const r of this.q('SELECT family, count FROM redactions ORDER BY count DESC').all() as { family: string; count: number }[]) {
      out[r.family] = Number(r.count);
    }
    return out;
  }

  setMeta(key: string, value: string): void {
    this.q('INSERT INTO meta(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value').run(key, value);
  }

  getMeta(key: string): string | null {
    const row = this.q('SELECT value FROM meta WHERE key = ?').get(key) as { value: string } | undefined;
    return row ? row.value : null;
  }

  private static toChunk(r: Record<string, unknown>): ChunkRow {
    return {
      id: Number(r.id),
      uuid: String(r.uuid),
      part: Number(r.part),
      sessionPk: Number(r.session_pk),
      projectId: Number(r.project_id),
      role: r.role as Role,
      ts: Number(r.ts),
      text: String(r.text),
    };
  }

  getChunk(id: number): ChunkRow | null {
    const r = this.q('SELECT * FROM chunks WHERE id = ?').get(id) as Record<string, unknown> | undefined;
    return r ? Store.toChunk(r) : null;
  }

  /** Chunks with `id > afterId`, in id order. */
  chunksAfter(afterId: number, limit: number): ChunkRow[] {
    return (this.q('SELECT * FROM chunks WHERE id > ? ORDER BY id LIMIT ?').all(afterId, limit) as Record<string, unknown>[]).map(
      Store.toChunk,
    );
  }

  /** Every chunk in id order. The corpus is small enough (tens of thousands) to hold in memory. */
  allChunks(): ChunkRow[] {
    return (this.q('SELECT * FROM chunks ORDER BY id').all() as Record<string, unknown>[]).map(Store.toChunk);
  }

  sessionChunks(sessionPk: number, limit: number): ChunkRow[] {
    return (
      this.q('SELECT * FROM chunks WHERE session_pk = ? ORDER BY ts, id LIMIT ?').all(sessionPk, limit) as Record<string, unknown>[]
    ).map(Store.toChunk);
  }

  projects(): ProjectRow[] {
    return (this.q('SELECT id, key, name FROM projects ORDER BY id').all() as Record<string, unknown>[]).map((r) => ({
      id: Number(r.id),
      key: String(r.key),
      name: String(r.name),
    }));
  }

  sessions(): SessionRow[] {
    return (this.q('SELECT * FROM sessions ORDER BY first_ts').all() as Record<string, unknown>[]).map((r) => ({
      pk: Number(r.pk),
      sessionId: String(r.session_id),
      projectId: Number(r.project_id),
      title: r.title === null ? null : String(r.title),
      firstTs: Number(r.first_ts),
      lastTs: Number(r.last_ts),
    }));
  }

  /** Oracle only: SQLite's own BM25 ranking, used to check the from-scratch implementation. */
  ftsTop(matchExpr: string, k: number): { id: number; score: number }[] {
    return (
      this.q('SELECT rowid AS id, bm25(chunks_fts) AS score FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?').all(
        matchExpr,
        k,
      ) as { id: number; score: number }[]
    ).map((r) => ({ id: Number(r.id), score: -Number(r.score) }));
  }

  close(): void {
    this.stmts.clear();
    this.db.close();
  }
}
