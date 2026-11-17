import { expect, test } from '@playwright/test';

const stage = '#stage';

async function ready(page: import('@playwright/test').Page): Promise<void> {
  await page.goto('/?debug');
  await expect(page.locator(stage)).toHaveAttribute('data-ready', '1', { timeout: 30_000 });
}

test('draws one particle per chunk and the simulation moves them', async ({ page }) => {
  const errors: string[] = [];
  page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));
  page.on('pageerror', (e) => errors.push(e.message));
  await ready(page);
  await expect(page.locator(stage)).toHaveAttribute('data-particles', '6000');
  await expect(page.locator('#demo-badge')).toBeVisible();
  await expect(page.locator('#summary')).toContainText('6,000 chunks');
  await expect(page.locator('#fatal')).toBeHidden();

  // Positions read back from the GPU must change while the layout settles.
  await expect.poll(async () => Number(await page.locator(stage).getAttribute('data-frame')), { timeout: 20_000 }).toBeGreaterThan(40);
  const first = await page.locator(stage).getAttribute('data-sample');
  await expect.poll(async () => page.locator(stage).getAttribute('data-sample'), { timeout: 20_000 }).not.toBe(first);
  const sample = (await page.locator(stage).getAttribute('data-sample'))!.split(',').map(Number);
  expect(sample.every(Number.isFinite)).toBe(true);
  expect(errors).toEqual([]);
});

test('search returns on-topic results and draws the path the index took', async ({ page }) => {
  await ready(page);
  await page.locator('#q').fill('sourdough starter hydration');
  await page.locator('#q').press('Enter');
  await expect(page.locator('#results')).toBeVisible();
  await expect(page.locator('#results-list li')).toHaveCount(12);
  await expect(page.locator('#results-stats')).toContainText('graph search visited');
  // Every result should come from the topic the query words belong to.
  const metas = await page.locator('#results-list .hit-meta').allTextContents();
  expect(metas.filter((m) => m.includes('kitchen')).length).toBeGreaterThanOrEqual(11);
  await expect(page.locator(stage)).toHaveAttribute('data-hits', '12');
  expect(Number(await page.locator(stage).getAttribute('data-trace-steps'))).toBeGreaterThan(10);

  // Keyword-only mode works without the vector index.
  await page.locator('#search-modes button[data-mode="bm25"]').click();
  await expect(page.locator('#results-stats')).not.toContainText('graph search');
  await expect(page.locator('#results-list li')).toHaveCount(12);

  // Opening a result shows its text and its neighbours in the index.
  await page.locator('#results-list li button').first().click();
  await expect(page.locator('#detail')).toBeVisible();
  await expect(page.locator('#detail-meta')).toContainText('kitchen');
  await expect(page.locator('#detail-neighbours li')).not.toHaveCount(0);

  await page.keyboard.press('Escape');
  await expect(page.locator('#results')).toBeHidden();
  await expect(page.locator('#detail')).toBeHidden();
});

test('the time scrubber hides later chunks and project focus works', async ({ page }) => {
  await ready(page);
  await page.locator('#scrub').fill('250');
  await expect(page.locator(stage)).toHaveAttribute('data-reveal', '0.250');
  await expect(page.locator('#scrub-date')).not.toHaveText('');
  const early = await page.locator('#scrub-date').textContent();
  await page.locator('#scrub').fill('1000');
  await expect(page.locator('#scrub-date')).not.toHaveText(early ?? '');

  const firstProject = page.locator('#legend button').first();
  await firstProject.click();
  await expect(firstProject).toHaveAttribute('aria-pressed', 'true');
  await firstProject.click();
  await expect(firstProject).toHaveAttribute('aria-pressed', 'false');
});

test('view options: labels, colour modes, dust, glow and depth', async ({ page }) => {
  await ready(page);
  const canvas = page.locator(stage);

  // Denser field: more dust points than chunks, and region labels drawn from topic words.
  expect(Number(await canvas.getAttribute('data-dust-count'))).toBeGreaterThan(6000);
  expect(Number(await canvas.getAttribute('data-label-count'))).toBeGreaterThanOrEqual(4);
  await expect.poll(async () => page.locator('.cluster-label').evaluateAll((els) => els.filter((e) => Number(getComputedStyle(e).opacity) > 0.2).length), { timeout: 20_000 }).toBeGreaterThan(0);
  const words = (await page.locator('.cluster-label').allTextContents()).join(' ').toLowerCase();
  expect(/sourdough|tokenizer|satellite|tomato|socket|chord|orbit|grammar|compost|latency|melody|hydration/.test(words)).toBe(true);

  await page.locator('#toggle-labels').click();
  await expect(page.locator('#labels')).toBeHidden();
  await expect(page.locator('#toggle-labels')).toHaveAttribute('aria-pressed', 'false');
  for (const name of ['dust', 'glow', 'depth']) {
    await page.locator(`#toggle-${name}`).click();
    await expect(canvas).toHaveAttribute(`data-${name}`, '0');
  }

  // Colour modes change the field, the key and the chart together.
  await page.locator('#color-modes button[data-color="time"]').click();
  await expect(canvas).toHaveAttribute('data-color', 'time');
  await expect(page.locator('#mode-key .ramp')).toBeVisible();
  await page.locator('#color-modes button[data-color="speaker"]').click();
  await expect(page.locator('#mode-key')).toContainText('assistant');
  await expect(page.locator('#activity-table th[scope="col"]')).toHaveText(['Period', 'You', 'Assistant', 'Total']);

  // The choices are remembered for this viewer.
  await page.reload();
  await expect(canvas).toHaveAttribute('data-ready', '1', { timeout: 30_000 });
  await expect(canvas).toHaveAttribute('data-color', 'speaker');
  await expect(canvas).toHaveAttribute('data-glow', '0');
  await expect(page.locator('#labels')).toBeHidden();
  await expect(page.locator('#color-modes button[data-color="speaker"]')).toHaveAttribute('aria-checked', 'true');
});

test('the activity chart shows weekly counts, a tooltip and moves the time slider', async ({ page }) => {
  await ready(page);
  const chart = page.locator('#activity');
  await expect(chart).toBeVisible();
  // The table behind the picture carries the same numbers and adds up to the corpus.
  const rows = page.locator('#activity-table tr');
  expect(await rows.count()).toBeGreaterThan(10);
  const totals = await page.locator('#activity-table tr td:last-child').allTextContents();
  expect(totals.map(Number).reduce((a, b) => a + b, 0)).toBe(6000);
  await expect(page.locator('#activity-table th[scope="col"]').first()).toHaveText('Period');

  const box = (await chart.boundingBox())!;
  await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.6);
  await expect(page.locator('#activity-tip')).toBeVisible();
  await expect(page.locator('#activity-tip .tip-head')).toContainText('chunks');
  expect(await page.locator('#activity-tip .tip-row').count()).toBeGreaterThan(1);

  await page.mouse.click(box.x + box.width * 0.25, box.y + box.height * 0.6);
  const reveal = Number(await page.locator(stage).getAttribute('data-reveal'));
  expect(reveal).toBeGreaterThan(0.1);
  expect(reveal).toBeLessThan(0.4);
  await expect(page.locator('#scrub')).not.toHaveValue('1000');
});

test('what you keep repeating: lists habits and lights up where they were said', async ({ page }) => {
  await ready(page);
  const canvas = page.locator(stage);
  await page.locator('#open-repeats').click();
  await expect(page.locator('#repeats')).toBeVisible();
  await expect(page.locator('#open-repeats')).toHaveAttribute('aria-pressed', 'true');
  await expect(page.locator('#repeats-stats')).toContainText('sessions on 2+ days');
  const items = page.locator('#repeats-list li button');
  expect(await items.count()).toBeGreaterThanOrEqual(3);
  // The demo plants three standing instructions; they should lead the list.
  const texts = await page.locator('#repeats-list .hit-text').allTextContents();
  expect(texts.slice(0, 3).every((t) => /unit tests before you commit|answer short|push to the remote/.test(t))).toBe(true);
  await expect(page.locator('#repeats-list .kind').first()).toHaveText('instruction');

  await items.first().click();
  await expect(items.first()).toHaveAttribute('aria-current', 'true');
  expect(Number(await canvas.getAttribute('data-repeat'))).toBeGreaterThanOrEqual(3);

  // A search takes over the right-hand panel; closing clears the highlight.
  await page.locator('#q').fill('satellite orbit');
  await page.locator('#q').press('Enter');
  await expect(page.locator('#results')).toBeVisible();
  await expect(page.locator('#repeats')).toBeHidden();
  await page.locator('#open-repeats').click();
  await expect(page.locator('#repeats')).toBeVisible();
  await expect(page.locator('#results')).toBeHidden();
  await page.locator('#repeats-close').click();
  await expect(page.locator('#repeats')).toBeHidden();
  await expect(canvas).toHaveAttribute('data-repeat', '0');
});

test('conversation text is never interpreted as markup', async ({ page }) => {
  // The demo server is the real server code path; intercept one response to inject hostile text.
  await page.route('**/api/search?**', async (route) => {
    const res = await route.fetch();
    const body = await res.json();
    body.hits[0].text = '<img src=x onerror="window.__pwned = 1"><script>window.__pwned = 1</script>';
    body.hits[0].project = '<b>bold</b>';
    await route.fulfill({ response: res, json: body });
  });
  // Region labels and project names also come from conversation-derived data.
  await page.route('**/api/manifest', async (route) => {
    const res = await route.fetch();
    const body = await res.json();
    body.clusters[0].words = ['<img src=x onerror="window.__pwned = 1">', '<b>bold</b>'];
    body.projects[0].name = '<i>italic</i><script>window.__pwned = 1</script>';
    await route.fulfill({ response: res, json: body });
  });
  // So do the sentences in the repeats panel.
  await page.route('**/api/repeats', async (route) => {
    const res = await route.fetch();
    const body = await res.json();
    body.items[0].text = '<img src=x onerror="window.__pwned = 1">said again';
    body.items[0].variants = ['"><script>window.__pwned = 1</script>'];
    await route.fulfill({ response: res, json: body });
  });
  await ready(page);
  await page.locator('#open-repeats').click();
  await expect(page.locator('#repeats-list .hit-text').first()).toContainText('<img src=x');
  expect(await page.locator('#repeats img, #repeats script').count()).toBe(0);
  await page.locator('#repeats-close').click();
  expect(await page.locator('#labels img, #labels b, #legend i, #legend script').count()).toBe(0);
  await expect(page.locator('#legend .name').first()).toContainText('<i>italic</i>');
  await page.locator('#q').fill('satellite orbit');
  await page.locator('#q').press('Enter');
  await expect(page.locator('#results-list li')).toHaveCount(12);
  await expect(page.locator('#results-list .hit-text').first()).toContainText('<img src=x');
  expect(await page.locator('#results-list img, #results-list script, #results-list b').count()).toBe(0);
  expect(await page.evaluate(() => (window as unknown as { __pwned?: number }).__pwned)).toBeUndefined();
});

test('the server refuses foreign hosts, foreign origins, writes and path escapes', async ({ request }) => {
  expect((await request.get('/api/health')).status()).toBe(200);
  expect((await request.get('/api/health', { headers: { host: 'evil.example' } })).status()).toBe(403);
  expect((await request.get('/api/health', { headers: { origin: 'https://evil.example' } })).status()).toBe(403);
  expect((await request.post('/api/search?q=x')).status()).toBe(405);
  expect((await request.get('/api/search')).status()).toBe(400);
  expect((await request.get(`/api/search?q=${'a'.repeat(600)}`)).status()).toBe(400);
  expect((await request.get('/api/search?q=x&k=9999')).status()).toBe(400);
  expect((await request.get('/api/chunk?particle=-1')).status()).toBe(400);
  expect((await request.get('/api/buffers/secrets')).status()).toBe(404);
  expect((await request.get('/..%2f..%2fpackage.json')).status()).not.toBe(200);
  expect((await request.get('/%2e%2e/%2e%2e/src/cli.ts')).status()).not.toBe(200);
  const res = await request.get('/');
  expect(res.headers()['content-security-policy']).toContain("default-src 'self'");
  expect(res.headers()['x-content-type-options']).toBe('nosniff');
});
