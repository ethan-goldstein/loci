import { defineConfig } from '@playwright/test';

/**
 * End-to-end tests run against the demo store (invented text, no model needed), so they never
 * touch private data and work in CI. WebGL runs on SwiftShader, the software renderer.
 */
export default defineConfig({
  testDir: 'tests/e2e',
  timeout: 60_000,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [['list'], ['html', { open: 'never' }]] : 'list',
  use: {
    baseURL: 'http://127.0.0.1:4329',
    viewport: { width: 1440, height: 900 },
    launchOptions: { args: ['--use-angle=swiftshader', '--use-gl=angle', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] },
  },
  projects: [{ name: 'chromium', use: { browserName: 'chromium' } }],
  webServer: {
    command: 'node src/cli.ts serve --demo',
    url: 'http://127.0.0.1:4329/api/health',
    env: { LOCI_PORT: '4329' },
    reuseExistingServer: false,
    timeout: 60_000,
  },
});
