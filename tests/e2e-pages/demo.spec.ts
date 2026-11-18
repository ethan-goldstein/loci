import { expect, test } from '@playwright/test';
import type { Page } from '@playwright/test';

/**
 * The public demo: a static build with no server behind it. These tests check that the whole
 * thing really does run in the page: no API calls, no requests to anywhere else, and the same
 * search, detail and repeats behaviour as the local viewer.
 */

const stage = '#stage';
const ORIGIN = 'http://127.0.0.1:4331';

interface Watch {
  errors: string[];
  requests: string[];
}

function watch(page: Page): Watch {
  const w: Watch = { errors: [], requests: [] };
  page.on('console', (m) => m.type() === 'error' && w.errors.push(m.text()));
  page.on('pageerror', (e) => w.errors.push(e.message));
  page.on('request', (r) => w.requests.push(r.url()));
  return w;
}

async function ready(page: Page): Promise<void> {
  await page.goto('./');
  await expect(page.locator(stage)).toHaveAttribute('data-ready', '1', { timeout: 30_000 });
}

/** Opens the demo with the welcome panel already dismissed. */
async function readyPastIntro(page: Page): Promise<void> {
  await page.addInitScript(() => localStorage.setItem('loci.intro.seen', '1'));
  await ready(page);
  await expect(page.locator('#intro')).toBeHidden();
}

test('loads from static files only and introduces itself', async ({ page }) => {
  const w = watch(page);
  // Count any attempt to compile WebAssembly: the public demo is meant to ship none.
  await page.addInitScript(() => {
    const seen = { n: 0 };
    (window as unknown as { __wasm: { n: number } }).__wasm = seen;
    const W = WebAssembly as unknown as Record<string, unknown>;
    for (const name of ['instantiate', 'instantiateStreaming', 'compile', 'compileStreaming', 'validate']) {
      const original = W[name] as (...args: unknown[]) => unknown;
      W[name] = (...args: unknown[]) => {
        seen.n++;
        return original.apply(WebAssembly, args);
      };
    }
    const Module = WebAssembly.Module;
    (WebAssembly as unknown as { Module: unknown }).Module = function (this: unknown, ...args: [BufferSource]) {
      seen.n++;
      return new Module(...args);
    };
  });
  await ready(page);
  await expect(page.locator(stage)).toHaveAttribute('data-particles', '6000');
  await expect(page.locator('#fatal')).toBeHidden();
  await expect(page.locator('#demo-badge')).toBeVisible();

  // First visit: the welcome panel says what this is and that the data is invented.
  await expect(page.locator('#intro')).toBeVisible();
  await expect(page.locator('#intro-title')).toHaveText('A map of how you work with AI');
  await expect(page.locator('#intro')).toContainText('made-up conversations');
  const link = page.locator('#intro a');
  await expect(link).toHaveAttribute('href', 'https://github.com/ethan-goldstein/loci');
  await expect(link).toHaveAttribute('rel', 'noopener noreferrer');

  // The public build has no camera feature and carries a content security policy in the page.
  await expect(page.locator('#gesture')).toBeHidden();
  const csp = await page.locator('meta[http-equiv="Content-Security-Policy"]').getAttribute('content');
  expect(csp).toContain("default-src 'self'");
  expect(csp).toContain("object-src 'none'");
  expect(csp).not.toContain('unsafe-eval');

  // Dismissing is remembered; "about" brings it back.
  await page.locator('#intro-close').click();
  await expect(page.locator('#intro')).toBeHidden();
  await page.reload();
  await expect(page.locator(stage)).toHaveAttribute('data-ready', '1', { timeout: 30_000 });
  await expect(page.locator('#intro')).toBeHidden();
  await page.locator('#about-open').click();
  await expect(page.locator('#intro')).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(page.locator('#intro')).toBeHidden();

  // Nothing was fetched from anywhere but this site, and there is no API to call.
  expect(w.requests.length).toBeGreaterThan(5);
  expect(w.requests.filter((u) => !u.startsWith(`${ORIGIN}/loci/`) && !u.startsWith('data:') && !u.startsWith('blob:'))).toEqual([]);
  expect(w.requests.filter((u) => u.includes('/api/'))).toEqual([]);
  // A search has not run yet in this test, so run one, then check no WebAssembly was ever compiled.
  await page.locator('#q').fill('satellite orbit');
  await page.locator('#q').press('Enter');
  await expect(page.locator('#results-list li')).toHaveCount(12);
  expect(await page.evaluate(() => (window as unknown as { __wasm: { n: number } }).__wasm.n)).toBe(0);
  expect(w.errors).toEqual([]);
});

test('the welcome panel can start a search, which runs in the page', async ({ page }) => {
  const w = watch(page);
  await ready(page);
  await page.locator('#intro button[data-try="search"]').click();
  await expect(page.locator('#intro')).toBeHidden();
  await expect(page.locator('#q')).toHaveValue('sourdough starter hydration');
  await expect(page.locator('#results')).toBeVisible();
  await expect(page.locator('#results-list li')).toHaveCount(12);
  await expect(page.locator('#results-stats')).toContainText('graph search visited');
  const metas = await page.locator('#results-list .hit-meta').allTextContents();
  expect(metas.filter((m) => m.includes('kitchen')).length).toBeGreaterThanOrEqual(11);
  await expect(page.locator(stage)).toHaveAttribute('data-hits', '12');
  expect(Number(await page.locator(stage).getAttribute('data-trace-steps'))).toBeGreaterThan(10);

  // Near-copies are folded: the list is not one reply repeated, and the line above it says so.
  await expect(page.locator('#results-stats')).toContainText('folded');
  const openings = (await page.locator('#results-list .hit-text').allTextContents()).map((t) => t.trim().slice(0, 48));
  expect(new Set(openings).size).toBeGreaterThanOrEqual(7);

  // Every ranking mode works without a server.
  for (const mode of ['dense', 'bm25', 'recent', 'hybrid']) {
    await page.locator(`#search-modes button[data-mode="${mode}"]`).click();
    await expect(page.locator(`#search-modes button[data-mode="${mode}"]`)).toHaveAttribute('aria-checked', 'true');
    await expect(page.locator('#results-list li')).toHaveCount(12);
  }

  // Opening a result shows readable text and its neighbours in the index.
  await page.locator('#results-list li button').first().click();
  await expect(page.locator('#detail')).toBeVisible();
  await expect(page.locator('#detail-meta')).toContainText('kitchen');
  expect(((await page.locator('#detail-text').textContent()) ?? '').length).toBeGreaterThan(40);
  await expect(page.locator('#detail-neighbours li')).not.toHaveCount(0);

  expect(w.requests.filter((u) => u.includes('/api/'))).toEqual([]);
  expect(w.errors).toEqual([]);
});

test('what keeps getting repeated: planted habits lead, of more than one kind', async ({ page }) => {
  const w = watch(page);
  await readyPastIntro(page);
  await page.locator('#open-repeats').click();
  await expect(page.locator('#repeats')).toBeVisible();
  const texts = await page.locator('#repeats-list .hit-text').allTextContents();
  expect(texts.length).toBeGreaterThanOrEqual(8);
  expect(texts.slice(0, 3).every((t) => /unit tests before you commit|answer short|push to the remote/.test(t))).toBe(true);
  expect(texts.some((t) => t.includes('metric units'))).toBe(true);
  const kinds = new Set(await page.locator('#repeats-list .kind').allTextContents());
  expect(kinds.has('instruction') && kinds.has('question') && kinds.has('task')).toBe(true);

  await page.locator('#repeats-list li button').first().click();
  expect(Number(await page.locator(stage).getAttribute('data-repeat'))).toBeGreaterThanOrEqual(3);
  await page.locator('#repeats-close').click();
  await expect(page.locator(stage)).toHaveAttribute('data-repeat', '0');
  expect(w.errors).toEqual([]);
});

test('where the time goes: the profile panel works without a server', async ({ page }) => {
  const w = watch(page);
  await readyPastIntro(page);
  await page.locator('#open-profile').click();
  await expect(page.locator('#profile')).toBeVisible();
  await expect(page.locator('#open-profile')).toHaveAttribute('aria-pressed', 'true');
  await expect(page.locator(stage)).toHaveAttribute('data-profile', '1');
  await expect(page.locator('#profile-stats')).toContainText('counts only, no text');
  await expect(page.locator('#profile .tile')).toHaveCount(4);
  // One cell per hour of the week, and the invented history keeps the small hours empty.
  await expect(page.locator('#profile svg.rhythm rect')).toHaveCount(168);
  const cells = await page.locator('#profile svg.rhythm rect').evaluateAll((els) => els.map((e) => Number(e.getAttribute('data-v'))));
  expect(cells.reduce((a, b) => a + b, 0)).toBeGreaterThan(500);
  for (let d = 0; d < 7; d++) for (let h = 2; h <= 7; h++) expect(cells[d * 24 + h]).toBe(0);
  await expect(page.locator('#profile .bar')).toHaveCount(6);
  await expect(page.locator('#profile svg.trend polyline')).toHaveCount(1);
  // Every chart has a table for screen readers.
  await expect(page.locator('#profile table.sr-only')).toHaveCount(3);
  // The value under the pointer is shown beside the mark.
  await page.locator('#profile .bar').first().hover();
  await expect(page.locator('#profile-tip')).toBeVisible();
  await expect(page.locator('#profile-tip')).toContainText('session');
  // No message text and no project name appears in the panel.
  const text = (await page.locator('#profile').textContent()) ?? '';
  expect(/sourdough|tokenizer|satellite|kitchen|parser|garden|network|music|orbit/i.test(text)).toBe(false);

  // One right-hand panel at a time.
  await page.locator('#open-repeats').click();
  await expect(page.locator('#repeats')).toBeVisible();
  await expect(page.locator('#profile')).toBeHidden();
  await page.locator('#open-profile').click();
  await expect(page.locator('#repeats')).toBeHidden();
  await page.locator('#profile-close').click();
  await expect(page.locator('#profile')).toBeHidden();
  await expect(page.locator('#open-profile')).toHaveAttribute('aria-pressed', 'false');
  expect(w.requests.filter((u) => u.includes('/api/'))).toEqual([]);
  expect(w.errors).toEqual([]);
});

test('project filter and timeline work without a server', async ({ page }) => {
  await readyPastIntro(page);
  // Isolate one project, then search inside it: every hit must belong to it.
  const first = page.locator('#legend button').first();
  const name = ((await first.locator('.name').textContent()) ?? '').trim();
  expect(name.length).toBeGreaterThan(2);
  await first.click();
  await page.locator('#q').fill('add a test for the case');
  await page.locator('#q').press('Enter');
  await expect(page.locator('#results-list li')).toHaveCount(12);
  const metas = await page.locator('#results-list .hit-meta').allTextContents();
  expect(metas.every((m) => m.includes(name))).toBe(true);

  // Scrubbing back in time hides the newer points.
  await page.locator('#scrub').fill('250');
  await expect.poll(async () => Number(await page.locator(stage).getAttribute('data-reveal'))).toBeLessThan(0.3);
});

test('fits a phone screen', async ({ page }) => {
  const w = watch(page);
  await page.setViewportSize({ width: 390, height: 844 });
  await ready(page);
  // The welcome panel fits and can be dismissed.
  await expect(page.locator('#intro')).toBeVisible();
  const card = await page.locator('.intro-card').boundingBox();
  expect(card).not.toBeNull();
  expect(card!.x).toBeGreaterThanOrEqual(0);
  expect(card!.x + card!.width).toBeLessThanOrEqual(390);
  expect(card!.y + card!.height).toBeLessThanOrEqual(844);
  await page.locator('#intro-close').click();

  // Nothing makes the page wider than the screen.
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);

  await page.locator('#q').fill('satellite orbit');
  await page.locator('#q').press('Enter');
  await expect(page.locator('#results')).toBeVisible();
  await expect(page.locator('#results-list li')).toHaveCount(12);
  const results = await page.locator('#results').boundingBox();
  expect(results!.x).toBeGreaterThanOrEqual(0);
  expect(results!.x + results!.width).toBeLessThanOrEqual(390);
  expect(w.errors).toEqual([]);
});
