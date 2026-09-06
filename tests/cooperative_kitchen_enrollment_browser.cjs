/* Offline browser acceptance of the user-ID entry contract. No cloud calls. */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const crypto = require('node:crypto');
const {chromium} = require(process.env.KITCHEN_PLAYWRIGHT_MODULE || 'playwright');
const root = path.resolve(__dirname, '..');
const output = process.env.KITCHEN_ENROLLMENT_BROWSER_OUTPUT || path.join(root, 'output/cooperative_kitchen/user-id-browser');
const report = {mode: 'offline_frontend_fixture', passed: false, checks: [], errors: [], started_at: new Date().toISOString()};
const record = name => report.checks.push({name, passed: true});
const clone = value => JSON.parse(JSON.stringify(value));
const enrollmentOpen = {mode: 'internal_pilot', enabled: true, formal_ready: false, participant_id_pattern: '^[A-Za-z][A-Za-z0-9_-]{2,31}$', participant_id_example: 'user_01'};
let enrollment = clone(enrollmentOpen), studyReady = false;
const sessions = new Map(), receipts = new Map(), bodies = [], issuedIds = new Set();
function view(session) { return {run: session, state: null, episodes: [], questions: [], can_act: false, can_ask: false, can_auto: false, can_next: session.phase === 'instructions', can_restart: false, can_swap: false, consent: {text: {zh: '仅用于前端协议测试。', en: 'Frontend protocol test only.'}}}; }
function send(res, status, value, extra = {}) { res.writeHead(status, {'Content-Type': 'application/json', 'Cache-Control': 'no-store', ...extra}); res.end(JSON.stringify(value)); }
const server = http.createServer(async (req, res) => {
  try {
    const url = new URL(req.url, 'http://127.0.0.1');
    if (!url.pathname.startsWith('/api/')) {
      const name = {'/': 'index.html', '/app.js': 'app.js', '/style.css': 'style.css', '/renderer.js': 'renderer.js', '/favicon.svg': 'favicon.svg'}[url.pathname];
      if (!name) return send(res, 404, {});
      res.writeHead(200, {'Content-Type': name.endsWith('.js') ? 'text/javascript' : name.endsWith('.css') ? 'text/css' : name.endsWith('.svg') ? 'image/svg+xml' : 'text/html'});
      return res.end(fs.readFileSync(path.join(root, 'ui/cooperative_kitchen_web', name)));
    }
    if (url.pathname === '/api/status') return send(res, 200, {study_ready: studyReady, enrollment, policy_kind: 'neural'});
    let session = sessions.get((req.headers.cookie || '').match(/enrollment_fixture=([^;]+)/)?.[1]);
    if (url.pathname === '/api/view') return send(res, session ? 200 : 401, session ? view(session) : {error: 'No session'});
    let raw = ''; for await (const part of req) raw += part;
    const body = JSON.parse(raw || '{}');
    if (url.pathname === '/api/session') {
      bodies.push(clone(body));
      const prior = receipts.get(body.operation_id);
      if (prior) {
        assert.deepEqual(body, prior.body);
        session = sessions.get(prior.id);
      } else {
        if (body.mode === 'pilot') {
          if (!enrollment?.enabled) return send(res, 503, {error: 'Closed', code: 'enrollment_closed'});
          if (!/^[A-Za-z][A-Za-z0-9_-]{2,31}$/.test(body.participant_id || '')) return send(res, 400, {error: 'Invalid ID', code: 'invalid_participant_id'});
          if (issuedIds.has(body.participant_id.toLowerCase())) return send(res, 409, {error: 'User ID already exists', code: 'participant_id_taken'});
          issuedIds.add(body.participant_id.toLowerCase());
        }
        session = {id: crypto.randomUUID(), participant_id: body.participant_id || 'freeplay_fixture', mode: body.mode, phase: body.mode === 'pilot' ? 'consent' : 'instructions', version: 0, episode_id: null, episode_index: -1, language: body.language};
        sessions.set(session.id, session); receipts.set(body.operation_id, {id: session.id, body: clone(body)});
      }
      return send(res, 200, view(session), {'Set-Cookie': `enrollment_fixture=${session.id}; HttpOnly; SameSite=Strict; Path=/`});
    }
    if (url.pathname === '/api/command' && session && body.command === 'language') { session.language = body.language; session.version++; return send(res, 200, view(session)); }
    return send(res, 404, {error: 'Unimplemented fixture route'});
  } catch (error) { report.errors.push(error.message); send(res, 500, {error: 'Fixture failed'}); }
});
let browser;
async function lobby(url, size = {width: 1280, height: 800}) {
  const context = await browser.newContext({viewport: size, reducedMotion: 'reduce'}), page = await context.newPage();
  page.on('pageerror', error => report.errors.push(error.message));
  page.on('console', message => { if (message.type() === 'error' && /pattern attribute|Invalid regular expression/.test(message.text())) report.errors.push(message.text()); });
  await page.goto(url); await page.locator('#lobby').waitFor({state: 'visible'});
  await page.waitForFunction(() => !document.querySelector('#language').disabled);
  return {context, page};
}
async function noOverflow(page, size) {
  for (const selector of ['#user-id', '#join-study', '#freeplay']) { const b = await page.locator(selector).boundingBox(); assert.ok(b && b.x >= 0 && b.y >= 0 && b.x + b.width <= size.width && b.y + b.height <= size.height, selector); }
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
}
(async () => {
  try {
    fs.mkdirSync(output, {recursive: true});
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve)); const url = `http://127.0.0.1:${server.address().port}`;
    browser = await chromium.launch({headless: true, executablePath: process.env.KITCHEN_CHROME || undefined});
    let {context, page} = await lobby(url);
    assert.equal(await page.locator('#join-study').isEnabled(), true);
    assert.match(await page.locator('#status-label').innerText(), /内部预实验/);
    assert.match(await page.locator('#study-availability').innerText(), /尚未通过正式实验验收/);
    assert.equal(await page.locator('#user-id').getAttribute('name'), 'participant_id');
    assert.equal(await page.locator('#user-id').getAttribute('placeholder'), 'user_01');
    assert.doesNotMatch(await page.locator('body').innerText(), /邀请码|invitation/i);
    record('internal_pilot_enabled_without_claiming_formal_readiness');
    for (const [value, valid] of [['ab', false], ['1ab', false], ['user name', false], ['user@email.com', false], ['用户01', false], ['user-01', true], ['User_01', true], ['a' + '1'.repeat(31), true], ['a' + '1'.repeat(32), false]]) {
      await page.locator('#user-id').evaluate((input, value) => { input.value = value; input.dispatchEvent(new Event('input', {bubbles: true})); }, value);
      assert.equal(await page.locator('#user-id').evaluate(input => input.checkValidity()), valid, value);
    }
    await page.locator('#user-id').fill('bad@id'); await page.locator('#join-study').click(); assert.equal(bodies.length, 0);
    assert.match(await page.locator('#user-id-help').innerText(), /不要填写姓名、邮箱或电话号码/);
    record('id_pattern_native_validation_and_no_personal_details');
    await page.locator('#user-id').fill('user_01');
    for (const size of [{width: 1365, height: 900}, {width: 1280, height: 800}]) {
      await page.setViewportSize(size);
      for (const lang of ['zh', 'en']) {
        if (await page.locator('html').getAttribute('lang') !== lang) await page.locator('#language').click();
        await noOverflow(page, size);
        await page.screenshot({path: path.join(output, `user-id-${size.width}x${size.height}-${lang}.png`), fullPage: true});
      }
    }
    record('bilingual_entry_fits_both_viewports');
    await page.locator('#user-id').fill('user_01'); await page.locator('#join-study').click(); await page.locator('#consent').waitFor({state: 'visible'});
    assert.deepEqual(Object.keys(bodies.at(-1)).sort(), ['language', 'mode', 'operation_id', 'participant_id']);
    assert.equal(bodies.at(-1).participant_id, 'user_01'); assert.equal(bodies.at(-1).mode, 'pilot');
    await page.reload(); await page.locator('#consent').waitFor({state: 'visible'}); assert.equal(issuedIds.size, 1); await context.close();
    record('participant_id_contract_and_cookie_refresh_resume');
    ({context, page} = await lobby(url));
    await page.locator('#user-id').fill('user_01'); await page.locator('#join-study').click(); await page.locator('#user-id-error').waitFor({state: 'visible'});
    assert.equal(await page.locator('#user-id').inputValue(), 'user_01'); assert.match(await page.locator('#user-id-error').innerText(), /加 _1/);
    assert.equal(await page.locator('#user-id').getAttribute('aria-invalid'), 'true'); assert.equal(await page.locator('#notice').isVisible(), false);
    await page.locator('#language').click(); assert.match(await page.locator('#user-id-error').innerText(), /Add _1/); assert.equal(await page.locator('#user-id').inputValue(), 'user_01');
    await page.screenshot({path: path.join(output, 'duplicate-user-id-en.png'), fullPage: true});
    await page.locator('#user-id').fill('user_01_1'); assert.equal(await page.locator('#user-id-error').isVisible(), false);
    await page.locator('#join-study').click(); await page.locator('#consent').waitFor({state: 'visible'}); await context.close();
    record('duplicate_id_keeps_input_localizes_hint_and_recovers_with_suffix');
    ({context, page} = await lobby(url));
    await page.locator('#user-id').fill('  trimmed_01  '); await page.locator('#join-study').click(); await page.locator('#consent').waitFor({state: 'visible'});
    assert.equal(bodies.at(-1).participant_id, 'trimmed_01'); await context.close();
    record('participant_id_trims_surrounding_whitespace_before_validation');
    ({context, page} = await lobby(url));
    let dropped = false;
    await page.route('**/api/session', async route => { if (!dropped) { dropped = true; await route.fetch(); await route.abort('failed'); } else await route.continue(); });
    await page.locator('#user-id').fill('recovery_01'); await page.locator('#join-study').click(); await page.locator('#retry-request').waitFor({state: 'visible'});
    assert.equal(await page.locator('#user-id').inputValue(), 'recovery_01');
    await page.reload(); await page.locator('#consent').waitFor({state: 'visible'});
    const attempts = bodies.filter(body => body.participant_id === 'recovery_01'); assert.equal(attempts.length, 2); assert.deepEqual(attempts[0], attempts[1]); await context.close();
    record('uncertain_enrollment_replays_same_id_and_operation_after_reload');
    enrollment = {mode: 'closed', enabled: false, formal_ready: false}; studyReady = true;
    ({context, page} = await lobby(url)); assert.equal(await page.locator('#join-study').isDisabled(), true); assert.equal(await page.locator('#freeplay').isEnabled(), true);
    await page.locator('#freeplay').click(); await page.locator('#instructions').waitFor({state: 'visible'}); assert.equal(bodies.at(-1).mode, 'freeplay'); assert.ok(!('participant_id' in bodies.at(-1))); await context.close();
    record('closed_enrollment_ignores_legacy_ready_flag_and_keeps_freeplay');
    enrollment = null;
    ({context, page} = await lobby(url)); assert.equal(await page.locator('#join-study').isDisabled(), true); await context.close();
    record('missing_enrollment_contract_fails_closed');
    assert.deepEqual(report.errors, []); report.passed = true;
  } catch (error) { report.failure = error.stack || error.message; process.exitCode = 1; }
  finally { report.finished_at = new Date().toISOString(); fs.mkdirSync(output, {recursive: true}); fs.writeFileSync(path.join(output, 'enrollment_browser_acceptance.json'), JSON.stringify(report, null, 2)); if (browser) await browser.close(); if (server.listening) await new Promise(resolve => server.close(resolve)); console.log(JSON.stringify({passed: report.passed, checks: report.checks.length, output, failure: report.failure || null}, null, 2)); }
})();
