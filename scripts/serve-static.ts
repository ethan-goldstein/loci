import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

/**
 * A tiny static file server for checking the public demo build the way a static host serves it:
 * under a sub-path, with no API behind it. Loopback only, GET and HEAD only, confined to one
 * directory. For tests and local preview; it is not how anything is deployed.
 *
 *   node scripts/serve-static.ts [dir] [port] [prefix]
 *   node scripts/serve-static.ts dist-pages 4331 /loci/
 */

const root = fileURLToPath(new URL('..', import.meta.url));
const dir = path.resolve(root, process.argv[2] ?? 'dist-pages') + path.sep;
const port = Number(process.argv[3] ?? 4331);
const prefix = process.argv[4] ?? '/loci/';

const TYPES: Record<string, string> = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.woff2': 'font/woff2',
};

if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error('port must be a whole number between 1 and 65535');
if (!prefix.startsWith('/') || !prefix.endsWith('/')) throw new Error('prefix must start and end with a slash');
if (!fs.existsSync(path.join(dir, 'index.html'))) throw new Error('Nothing to serve. Run `npm run build:pages` first.');

const server = http.createServer((req, res) => {
  const send = (status: number, type: string, body: Buffer | string) => {
    res.writeHead(status, { 'content-type': type, 'x-content-type-options': 'nosniff', 'cache-control': 'no-store' });
    res.end(req.method === 'HEAD' ? undefined : body);
  };
  if (req.method !== 'GET' && req.method !== 'HEAD') return send(405, 'text/plain', 'Method not allowed');
  let pathname: string;
  try {
    pathname = decodeURIComponent(new URL(req.url ?? '/', `http://127.0.0.1:${port}`).pathname);
  } catch {
    return send(400, 'text/plain', 'Bad path');
  }
  if (pathname === '/' && prefix !== '/') {
    res.writeHead(302, { location: prefix });
    return res.end();
  }
  if (!pathname.startsWith(prefix) || pathname.includes('\0')) return send(404, 'text/plain', 'Not found');
  let rel = pathname.slice(prefix.length);
  if (rel === '' || rel.endsWith('/')) rel += 'index.html';
  const full = path.normalize(path.join(dir, rel));
  // Confinement: the resolved path must stay inside the served directory.
  if (!full.startsWith(dir) || !fs.existsSync(full) || !fs.statSync(full).isFile()) return send(404, 'text/plain', 'Not found');
  send(200, TYPES[path.extname(full).toLowerCase()] ?? 'application/octet-stream', fs.readFileSync(full));
});

server.listen(port, '127.0.0.1', () => console.log(`static: serving ${path.relative(root, dir) || '.'} at http://127.0.0.1:${port}${prefix}`));
