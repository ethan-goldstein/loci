import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';

/**
 * Fails when something private is tracked by git:
 *   - a store, vector or index file (which hold conversation text or its embeddings)
 *   - an environment file other than the example
 *   - any tracked text file that mentions this machine's home directory
 *
 * Run by CI and before committing. `.gitignore` is the first line of defence; this is the second.
 */

const tracked = execFileSync('git', ['ls-files', '-z'], { encoding: 'utf8' }).split('\0').filter(Boolean);
const problems: string[] = [];

const FORBIDDEN = [/\.db(-wal|-shm)?$/, /\.f32$/, /\.bin$/, /^data\//, /^drafts\//, /(^|\/)\.env(\..+)?$/, /\.task$/, /\.wasm$/];
for (const file of tracked) {
  if (file.endsWith('.env.example')) continue;
  if (FORBIDDEN.some((re) => re.test(file))) problems.push(`${file}: private or third-party binary file is tracked`);
}

const home = os.homedir();
const user = os.userInfo().username;
for (const file of tracked) {
  if (!/\.(ts|js|mjs|json|md|html|css|yml|yaml|txt)$/.test(file) || file === 'package-lock.json') continue;
  const text = fs.readFileSync(file, 'utf8');
  if (text.includes(home) || (user.length >= 4 && text.includes(`/Users/${user}`))) problems.push(`${file}: contains this machine's home directory path`);
}

if (problems.length > 0) {
  console.error(problems.join('\n'));
  process.exit(1);
}
console.log(`guard: ${tracked.length} tracked files, nothing private`);
