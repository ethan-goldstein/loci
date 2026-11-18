#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import { ensureHome, loadConfig, paths } from './config.ts';
import type { Config } from './config.ts';
import { ollamaEmbedder } from './embed/ollama.ts';
import { embedAll } from './embed/run.ts';
import { Engine } from './engine.ts';
import type { Mode } from './engine.ts';
import { DIVERSIFY_DEFAULT } from './search/core.ts';
import { ingest, ingestChatGpt } from './ingest/run.ts';
import { Store } from './ingest/store.ts';

type Flags = Record<string, string | boolean>;
type Command = (cfg: Config, args: string[], flags: Flags) => Promise<void>;

export function parseArgs(argv: string[]): { args: string[]; flags: Flags } {
  const args: string[] = [];
  const flags: Flags = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]!;
    if (a.startsWith('--')) {
      const eq = a.indexOf('=');
      if (eq > 0) flags[a.slice(2, eq)] = a.slice(eq + 1);
      else if (i + 1 < argv.length && !argv[i + 1]!.startsWith('--')) flags[a.slice(2)] = argv[++i]!;
      else flags[a.slice(2)] = true;
    } else args.push(a);
  }
  return { args, flags };
}

export function num(flags: Flags, name: string, fallback: number, min: number, max: number): number {
  const raw = flags[name];
  if (raw === undefined || raw === true || raw === false) return fallback;
  const n = Number(raw);
  if (!Number.isFinite(n) || n < min || n > max) throw new Error(`--${name} must be a number between ${min} and ${max}`);
  return n;
}

function openStore(cfg: Config, readOnly = false): Store {
  const p = paths(cfg);
  if (readOnly && !fs.existsSync(p.db)) throw new Error('No store yet. Run `loci ingest` first.');
  if (!readOnly) ensureHome(cfg.home);
  const store = new Store(p.db, { readOnly });
  if (!readOnly) fs.chmodSync(p.db, 0o600);
  return store;
}

function progress(stage: string, done: number, total: number): void {
  if (process.stderr.isTTY) process.stderr.write(`\r  ${stage} ${done}/${total}   `);
}
function endProgress(): void {
  if (process.stderr.isTTY) process.stderr.write('\r\x1b[K');
}

function openEngine(cfg: Config, save = true): Engine {
  const engine = Engine.open(cfg, { embedder: ollamaEmbedder(cfg.ollamaUrl, cfg.embedModel), save, onProgress: progress });
  endProgress();
  return engine;
}

/** Writes a private JSON file into the store directory. */
function writePrivate(cfg: Config, name: string, value: unknown): string {
  ensureHome(cfg.home);
  const file = path.join(cfg.home, name);
  fs.writeFileSync(file, JSON.stringify(value, null, 2), { mode: 0o600 });
  return file;
}

function readPrivate<T>(cfg: Config, name: string): T | null {
  const file = path.join(cfg.home, name);
  return fs.existsSync(file) ? (JSON.parse(fs.readFileSync(file, 'utf8')) as T) : null;
}

const commands: Record<string, { help: string; run: Command }> = {
  ingest: {
    help: 'Read session transcripts, redact secrets, store chunks. Safe to re-run. Flags: --chatgpt <export zip or folder>',
    run: async (cfg, _args, flags) => {
      if (flags.chatgpt === true) throw new Error('--chatgpt needs the path to a ChatGPT export zip, its unzipped folder, or conversations.json');
      const store = openStore(cfg);
      try {
        const t0 = performance.now();
        const report = await ingest(store, cfg.source, {
          includeAutomated: flags['include-automated'] === true,
          includeSubagents: flags['include-subagents'] === true,
          onProgress: (done, total) => {
            if (done % 50 === 0 || done === total) progress('reading files', done, total);
          },
        });
        endProgress();
        // A ChatGPT export is optional: from the flag, or from LOCI_CHATGPT_EXPORT so plain `loci ingest` keeps picking it up.
        const exportPath = typeof flags.chatgpt === 'string' ? flags.chatgpt : cfg.chatgptExport;
        const chatgpt = exportPath ? await ingestChatGpt(store, exportPath) : undefined;
        console.log(JSON.stringify({ ...report, chatgpt, seconds: Number(((performance.now() - t0) / 1000).toFixed(1)), totalChunks: store.chunkCount() }, null, 2));
      } finally {
        store.close();
      }
    },
  },
  embed: {
    help: 'Embed new chunks with the local Ollama model. Resumable.',
    run: async (cfg, _args, flags) => {
      const store = openStore(cfg, true);
      try {
        const report = await embedAll(store, paths(cfg).vectors, ollamaEmbedder(cfg.ollamaUrl, cfg.embedModel), {
          batch: num(flags, 'batch', 32, 1, 256),
          onProgress: (done, total) => progress('embedded', done, total),
        });
        endProgress();
        console.log(JSON.stringify(report, null, 2));
      } finally {
        store.close();
      }
    },
  },
  build: {
    help: 'Build or extend the HNSW index and save it.',
    run: async (cfg) => {
      const engine = openEngine(cfg);
      try {
        console.log(JSON.stringify({ ...engine.hnsw.stats(), dim: engine.dim, buildSeconds: Number((engine.indexBuildMs / 1000).toFixed(1)), terms: engine.bm25.vocabulary }, null, 2));
      } finally {
        engine.close();
      }
    },
  },
  search: {
    help: 'Search your history. Flags: --k 10 --mode hybrid|dense|bm25|recent --project <name> --plain (do not fold near-copies) --json',
    run: async (cfg, args, flags) => {
      const query = args.join(' ').trim();
      if (!query) throw new Error('Usage: loci search <words> [--k 10] [--mode hybrid|dense|bm25] [--project name]');
      if (query.length > 2000) throw new Error('Query is too long (2000 characters max)');
      const mode = (flags.mode ?? 'hybrid') as Mode;
      if (!['hybrid', 'dense', 'bm25', 'recent'].includes(mode)) throw new Error('--mode must be hybrid, dense, bm25 or recent');
      const engine = openEngine(cfg);
      try {
        let projectId: number | undefined;
        if (typeof flags.project === 'string') {
          const want = flags.project.toLowerCase();
          const match = engine.projects.find((p) => p.name.toLowerCase() === want) ?? engine.projects.find((p) => p.name.toLowerCase().includes(want));
          if (!match) throw new Error(`No project matching "${flags.project}"`);
          projectId = match.id;
        }
        const t0 = performance.now();
        const res = await engine.search(query, { k: num(flags, 'k', 10, 1, 100), mode, projectId, diversify: flags.plain ? undefined : DIVERSIFY_DEFAULT });
        const ms = performance.now() - t0;
        const hits = res.hits.map((h) => engine.describe(h, 300));
        if (flags.json) {
          console.log(JSON.stringify({ ms: Number(ms.toFixed(1)), dense: res.dense, folded: res.folded, hits }, null, 2));
          return;
        }
        console.log(`${hits.length} results in ${ms.toFixed(0)} ms (${mode}, dense side: ${res.dense})\n`);
        hits.forEach((h, i) => {
          console.log(`${String(i + 1).padStart(2)}. [${h.project}] ${h.date.slice(0, 10)} ${h.role}${h.similarity === null ? '' : `  sim ${h.similarity}`}${h.session ? `  · ${h.session}` : ''}`);
          console.log(`    ${h.text.replace(/\s+/g, ' ').slice(0, 220)}\n`);
        });
      } finally {
        engine.close();
      }
    },
  },
  bench: {
    help: 'Measure the index against exact search. Flags: --synthetic 50000 (0 to skip)',
    run: async (cfg, _args, flags) => {
      const { runBench } = await import('./prove/bench.ts');
      const engine = openEngine(cfg);
      try {
        const report = await runBench(engine, {
          embedder: ollamaEmbedder(cfg.ollamaUrl, cfg.embedModel),
          maxQueries: num(flags, 'queries', 500, 10, 5000),
          syntheticN: num(flags, 'synthetic', 50_000, 0, 1_000_000),
          onProgress: progress,
        });
        endProgress();
        writePrivate(cfg, 'bench.json', report);
        console.log(JSON.stringify(report, null, 2));
      } finally {
        engine.close();
      }
    },
  },
  eval: {
    help: 'Retrieval eval with baselines and intervals. Flags: --mask (hide names and paths)',
    run: async (cfg, _args, flags) => {
      const { runEval } = await import('./prove/eval.ts');
      const engine = openEngine(cfg);
      try {
        const report = await runEval(engine, {
          embedder: ollamaEmbedder(cfg.ollamaUrl, cfg.embedModel),
          mask: flags.mask === true,
          maxQueries: flags.queries === undefined ? undefined : num(flags, 'queries', 1000, 10, 100_000),
          onProgress: progress,
        });
        endProgress();
        writePrivate(cfg, 'eval.json', report);
        console.log(JSON.stringify(report, null, 2));
      } finally {
        engine.close();
      }
    },
  },
  ab: {
    help: 'Agent A/B: answers with and without recalled context, judged blind. Flags: --model llama3.2:3b --sessions N (smoke run, not saved)',
    run: async (cfg, _args, flags) => {
      const { AB, fileCache, fingerprint, runAb } = await import('./prove/ab.ts');
      const { modelDigest, ollamaAbModel } = await import('./prove/abmodel.ts');
      const name = typeof flags.model === 'string' ? flags.model : 'llama3.2:3b';
      if (!/^[A-Za-z0-9._:/-]{1,80}$/.test(name)) throw new Error('--model contains unexpected characters');
      const smoke = flags.sessions === undefined ? undefined : num(flags, 'sessions', 2, 1, 100_000);
      const engine = openEngine(cfg);
      try {
        // Generation goes to the same loopback-only endpoint as embedding: nothing leaves this machine.
        const model = ollamaAbModel(cfg.ollamaUrl, name);
        ensureHome(cfg.home);
        // The cache holds generated questions and answers, so it stays in the private store.
        const cache = fileCache(path.join(cfg.home, 'ab-cache.jsonl'), fingerprint(engine, model, cfg.embedModel, AB));
        const { report, items } = await runAb(engine, {
          embedder: ollamaEmbedder(cfg.ollamaUrl, cfg.embedModel),
          model,
          modelDigest: await modelDigest(cfg.ollamaUrl, name),
          cache,
          maxSessions: smoke,
          onProgress: progress,
        });
        endProgress();
        if (smoke === undefined) {
          writePrivate(cfg, 'ab.json', report);
          writePrivate(cfg, 'ab-items.json', items);
        }
        // Counts and scores only: the questions and answers are never printed.
        console.log(JSON.stringify({ ...report, rows: undefined }, null, 2));
      } finally {
        engine.close();
      }
    },
  },
  repeats: {
    help: 'Find what you keep telling your assistant across sessions. Flags: --min-sessions 3 --z 5 --rules --json',
    run: async (cfg, _args, flags) => {
      const { runRepeats, draftRules } = await import('./insight/repeats.ts');
      const engine = openEngine(cfg);
      try {
        const report = await runRepeats(engine, {
          embedder: ollamaEmbedder(cfg.ollamaUrl, cfg.embedModel),
          minSessions: num(flags, 'min-sessions', 3, 2, 50),
          z: num(flags, 'z', 5, 2, 12),
          onProgress: progress,
        });
        endProgress();
        // The report holds your own sentences, so it goes to the private store, not the repository.
        const file = writePrivate(cfg, 'repeats.json', report);
        if (flags.json) {
          console.log(JSON.stringify(report, null, 2));
          return;
        }
        if (flags.rules) {
          console.log(draftRules(report));
          return;
        }
        const day = (ms: number) => new Date(ms).toISOString().slice(0, 10);
        console.log(`${report.sentences.toLocaleString('en-US')} sentences from ${report.prompts.toLocaleString('en-US')} prompts you typed (${report.templatedPrompts} templated prompts and ${report.sameDayBursts} same-day bursts left out).`);
        console.log(`Same-thing threshold: similarity ${report.threshold.tau.toFixed(3)} (typical pair ${report.threshold.median.toFixed(3)}, ${report.threshold.z.toFixed(1)} robust standard deviations above it).`);
        console.log(`Cross-session pairs above it: ${report.pairs.above.toLocaleString('en-US')} found, about ${report.pairs.expectedByChance.toFixed(1)} expected by chance.`);
        console.log(`${report.groups} things said in ${report.minSessions}+ sessions on 2+ days (${report.byKind.instruction} instructions, ${report.byKind.task} recurring tasks, ${report.byKind.question} questions, ${report.byKind.statement} statements), covering ${(report.share * 100).toFixed(1)}% of sentences.\n`);
        for (const g of report.top.slice(0, 15)) {
          console.log(`${String(g.occurrences).padStart(3)}x in ${String(g.sessions).padStart(2)} sessions, ${g.projects.length} projects, ${day(g.first)} to ${day(g.last)}  [${g.kind}]`);
          console.log(`     ${g.text}`);
        }
        console.log(`\nFull list saved to ${file}. \`loci repeats --rules\` prints draft standing rules.`);
      } finally {
        engine.close();
      }
    },
  },
  report: {
    help: 'Write RESULTS.md from the last bench, eval, ab and repeats. Flags: --out RESULTS.md',
    run: async (cfg, _args, flags) => {
      const { renderResults } = await import('./prove/report.ts');
      const bench = readPrivate<import('./prove/bench.ts').BenchReport>(cfg, 'bench.json');
      const ev = readPrivate<import('./prove/eval.ts').EvalReport>(cfg, 'eval.json');
      const ab = readPrivate<import('./prove/ab.ts').AbReport>(cfg, 'ab.json');
      const repeats = readPrivate<import('./insight/repeats.ts').RepeatsReport>(cfg, 'repeats.json');
      if (!bench && !ev && !ab && !repeats) throw new Error('Nothing to report. Run `loci bench` and `loci eval` first.');
      const out = path.resolve(typeof flags.out === 'string' ? flags.out : 'RESULTS.md');
      fs.writeFileSync(out, renderResults(bench, ev, new Date().toISOString().slice(0, 10), ab, repeats));
      console.log(`wrote ${path.relative(process.cwd(), out) || out}`);
    },
  },
  stats: {
    help: 'Print corpus counts. Never prints conversation text.',
    run: async (cfg) => {
      const store = openStore(cfg, true);
      try {
        const projects = store.projects();
        const rows = store.db
          .prepare('SELECT project_id AS p, role, COUNT(*) AS n, SUM(LENGTH(text)) AS chars FROM chunks GROUP BY project_id, role')
          .all() as { p: number; role: string; n: number; chars: number }[];
        const byProject = projects.map((p) => {
          const mine = rows.filter((r) => Number(r.p) === p.id);
          const get = (role: string) => Number(mine.find((r) => r.role === role)?.n ?? 0);
          return { project: p.name, human: get('human'), assistant: get('assistant'), chars: mine.reduce((a, r) => a + Number(r.chars), 0) };
        });
        const total = store.chunkCount();
        const span = store.db.prepare('SELECT MIN(ts) AS first, MAX(ts) AS last FROM chunks').get() as { first: number; last: number };
        console.log(
          JSON.stringify(
            {
              chunks: total,
              sessions: store.sessions().length,
              projects: projects.length,
              from: total ? new Date(Number(span.first)).toISOString().slice(0, 10) : null,
              to: total ? new Date(Number(span.last)).toISOString().slice(0, 10) : null,
              redactions: store.redactions(),
              byProject: byProject.filter((p) => p.human + p.assistant > 0).sort((a, b) => b.chars - a.chars),
            },
            null,
            2,
          ),
        );
      } finally {
        store.close();
      }
    },
  },
  serve: {
    help: 'Start the local viewer on 127.0.0.1. Flags: --demo (synthetic data, safe to record)',
    run: async (cfg, _args, flags) => {
      const { startServer } = await import('./serve/http.ts');
      await startServer(cfg, { demo: flags.demo === true });
    },
  },
  mcp: {
    help: 'Run the read-only MCP server on stdio (for Claude Code and other agents).',
    run: async (cfg) => {
      const { startMcp } = await import('./mcp/server.ts');
      await startMcp(cfg);
    },
  },
  reset: {
    help: 'Delete the private store and start over. Requires --yes.',
    run: async (cfg, _args, flags) => {
      if (flags.yes !== true) throw new Error(`This deletes everything in ${cfg.home}. Re-run with --yes to confirm.`);
      const p = paths(cfg);
      const targets = [p.db, `${p.db}-wal`, `${p.db}-shm`, p.vectors, p.hnsw, ...['bench.json', 'eval.json', 'ab.json', 'ab-items.json', 'ab-cache.jsonl'].map((f) => path.join(cfg.home, f))];
      let removed = 0;
      for (const t of targets) {
        if (fs.existsSync(t)) {
          fs.rmSync(t);
          removed++;
        }
      }
      console.log(`removed ${removed} files`);
    },
  },
};

async function main(): Promise<void> {
  const [name, ...rest] = process.argv.slice(2);
  if (!name || name === 'help' || name === '--help' || !(name in commands)) {
    console.log('loci <command>\n');
    for (const [k, c] of Object.entries(commands)) console.log(`  ${k.padEnd(8)} ${c.help}`);
    process.exitCode = name && name !== 'help' && name !== '--help' ? 1 : 0;
    return;
  }
  const { args, flags } = parseArgs(rest);
  await commands[name]!.run(loadConfig(), args, flags);
}

// Only run when invoked directly, so tests can import the helpers above.
if (process.argv[1] && fs.realpathSync(process.argv[1]) === fs.realpathSync(new URL(import.meta.url).pathname)) {
  main().catch((err: unknown) => {
    // Message only: no stack traces.
    console.error(`loci: ${err instanceof Error ? err.message : String(err)}`);
    process.exitCode = 1;
  });
}
