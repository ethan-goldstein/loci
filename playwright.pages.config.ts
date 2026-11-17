import { defineConfig } from '@playwright/test';

/**
 * End-to-end tests for the public demo build (`npm run build:pages`), served the way a static
 * host serves it: from a sub-path, with no server behind it. Everything runs in the page, on
 * invented data. WebGL runs on SwiftShader, the software renderer.
 */
export default defineConfig({
  testDir: 'tests/e2e-pages',
  timeout: 60_000,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [['list'], ['html', { open: 'never', outputFolder: 'playwright-report-pages' }]] : 'list',
  use: {
    baseURL: 'http://127.0.0.1:4331/loci/',
    viewport: { width: 1440, height: 900 },
    launchOptions: { args: ['--use-angle=swiftshader', '--use-gl=angle', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] },
  },
  projects: [{ name: 'chromium', use: { browserName: 'chromium' } }],
  webServer: {
    command: 'node scripts/serve-static.ts dist-pages 4331 /loci/',
    url: 'http://127.0.0.1:4331/loci/',
    reuseExistingServer: false,
    timeout: 30_000,
  },
});
