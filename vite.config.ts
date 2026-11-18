import { defineConfig } from 'vitest/config';
import type { Plugin } from 'vite';
import { fileURLToPath } from 'node:url';

const apiPort = Number(process.env.LOCI_PORT ?? 4319);
const at = (rel: string) => fileURLToPath(new URL(rel, import.meta.url));

/**
 * The public demo is served by a static host that cannot set response headers, so its content
 * security policy travels in the page. It is stricter than the local server's: no WebAssembly,
 * no camera, nothing but this site's own files.
 */
const PAGES_CSP = [
  "default-src 'self'",
  "script-src 'self'",
  "style-src 'self' 'unsafe-inline'",
  "img-src 'self' data: blob:",
  "font-src 'self'",
  "connect-src 'self'",
  "object-src 'none'",
  "base-uri 'none'",
  "form-action 'none'",
].join('; ');

const pagesCsp = (): Plugin => ({
  name: 'loci-pages-csp',
  transformIndexHtml: (html) => html.replace('<meta name="referrer"', `<meta http-equiv="Content-Security-Policy" content="${PAGES_CSP}" />\n    <meta name="referrer"`),
});

/**
 * Two builds from one viewer:
 *   vite build               the local viewer, served by `loci serve` (dist/)
 *   vite build --mode pages  the public demo: invented data, engine in the page (dist-pages/)
 */
export default defineConfig(({ mode }) => {
  const pages = mode === 'pages';
  return {
    root: at('./web'),
    // Relative, so the demo works at any path: a project page, a fork, a custom domain.
    base: pages ? './' : '/',
    // The demo ships no third-party runtime assets: no MediaPipe, no gesture model.
    publicDir: at(pages ? './web/public-pages' : './web/public'),
    plugins: pages ? [pagesCsp()] : [],
    server: {
      host: '127.0.0.1',
      port: 4320,
      strictPort: true,
      // The viewer talks to the LOCI server for data; both stay on loopback.
      proxy: { '/api': `http://127.0.0.1:${apiPort}` },
    },
    build: {
      target: 'es2022',
      sourcemap: false,
      // Keep every asset a real file: an inlined data: font would be refused by the content security policy.
      assetsInlineLimit: 0,
      outDir: at(pages ? './dist-pages' : './dist'),
      emptyOutDir: true,
    },
    test: {
      root: at('.'),
      include: ['tests/unit/**/*.test.ts'],
      environment: 'node',
      testTimeout: 30_000,
      // Setup hooks build whole demo stores; give them the same room as the tests on a busy machine.
      hookTimeout: 30_000,
    },
  };
});
