/* Browser acceptance for the research UI.
 * Default: isolated protocol fixture, explicitly NOT a trained-policy test.
 * --real http://127.0.0.1:8003 additionally exercises real server free play.
 * --full <fixture.json> instead uses the isolated Python server with complete
 * 180-step physics, the selected trained Actor, real question bank/scenarios.
 * --auto <local-url> verifies freeplay automatic demonstration against a real
 * server, preserving the neural AI and testing both complete role assignments.
 * --layout runs only request/keyboard layout regressions on the protocol fixture.
 * --app-script <file> with --layout tests a saved frontend revision as a negative
 * control. It changes only the fixture's served app.js, never the source files.
 * KITCHEN_PLAYWRIGHT_MODULE / KITCHEN_CHROME select a local browser runtime.
 */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const crypto = require('node:crypto');
const {chromium} = require(process.env.KITCHEN_PLAYWRIGHT_MODULE || process.env.CANDY_PLAYWRIGHT_MODULE || 'playwright');
const Demo = require('../ui/cooperative_kitchen_demo/engine.js');
const args = process.argv.slice(2), fullFixture = args.includes('--full') ? args[args.indexOf('--full') + 1] : null;
const autoURL = args.includes('--auto') ? args[args.indexOf('--auto') + 1] : null;
const layoutOnly = args.includes('--layout');
const appScript = args.includes('--app-script') ? path.resolve(args[args.indexOf('--app-script') + 1]) : null;
if (appScript) assert.ok(layoutOnly && !autoURL && !fullFixture, '--app-script is restricted to the isolated layout fixture');
const root = path.resolve(__dirname, '..'), output = path.resolve(process.env.KITCHEN_STUDY_BROWSER_OUTPUT || path.join(root, autoURL ? 'output/cooperative_kitchen/v1/browser-auto' : fullFixture ? 'output/cooperative_kitchen/v1/browser-full' : 'output/cooperative_kitchen/v1/browser'));
fs.mkdirSync(output, {recursive: true});
const realURL = args.includes('--real') ? args[args.indexOf('--real') + 1] : null;
const report = {schema: 'cooperative_kitchen_study_browser_v1', started: new Date().toISOString(), fixture: 'Isolated protocol fixture with abbreviated rounds; not training or policy validation.', real_server: realURL, checks: [], screenshots: [], errors: []};
const record = (name, detail = {}) => report.checks.push({name, passed: true, ...detail});
const copy = value => JSON.parse(JSON.stringify(value));
const hash = value => crypto.createHash('sha256').update(JSON.stringify(value)).digest('hex');
const sessions = new Map(); let dropNextAction = false, lateQuestion = false, rejectNextQuestionCode = null, enrollment = 0;

function initial(preset = 'supply') { return {...Demo.reset(preset), map: [...Demo.MAP], maxSteps: 180, score: 0}; }
function startEpisode(session, index, stage) { const id = crypto.randomUUID(), state = initial(); state.episode_id = id; session.episodes.push({id, index, phase: stage, done: false, summary: {}, state, frames: [copy(state)]}); Object.assign(session.run, {phase: stage, episode_index: index, episode_id: id}); }
const active = session => session.episodes.find(e => e.id === session.run.episode_id);
const qa = session => session.run.mode === 'freeplay' || session.run.phase === 'task1' && session.condition === 'A';
function publicView(session) {
  const episode = active(session), playing = ['freeplay', 'practice', 'task1', 'task2'].includes(session.run.phase);
  return {run: copy(session.run), state: session.releaseChanged ? null : episode ? copy(episode.state) : null, requires_restart: Boolean(session.releaseChanged),
    episodes: session.episodes.map(({id, index, phase, done, summary}) => ({id, index, phase, done, summary})),
    can_ask: !session.releaseChanged && qa(session) && playing, can_act: !session.releaseChanged && playing && !episode?.done,
    can_next: ['consent', 'instructions'].includes(session.run.phase) || playing && episode?.done && session.run.mode !== 'freeplay',
    can_restart: session.run.mode === 'freeplay' && playing, can_swap: session.run.mode === 'freeplay' && playing,
    questions: qa(session) ? session.questions.map(q => ({...copy(q), answer: q.status === 'complete' ? copy(q.answer) : null})) : [],
    consent: {version: 'protocol-fixture', text: {zh: '这是一份仅供浏览器验收的隔离协议夹具，不用于研究招募或政策验收。', en: 'This isolated protocol fixture is used only for browser verification, not participant recruitment or policy validation.'}},
    survey: session.run.phase === 'questionnaire' ? {items: [
      {id: 'prediction', type: 'prediction', prompt: {zh: '在这个画面下，队友下一步最可能采取哪个动作？', en: 'What action is the teammate most likely to take next?'}, state: initial(), options: ['UP', 'DOWN', 'WAIT'].map(value => ({value, label: {zh: value, en: value}}))},
      {id: 'confidence', type: 'likert', prompt: {zh: '你对理解队友行为有多大信心？', en: 'How confident are you in understanding the teammate?'}, options: [1, 2, 3, 4, 5].map(value => ({value: String(value), label: {zh: `${value}`, en: `${value}`}}))}
    ], draft: copy(session.draft)} : null, completion_code: session.run.phase === 'complete' ? 'FIXTURE-COMPLETE' : null};
}
function send(res, status, value, headers = {}) { if (res.destroyed) return; res.writeHead(status, {'Content-Type': 'application/json', 'Cache-Control': 'no-store', ...headers}); res.end(JSON.stringify(value)); }
function sessionFor(req) { return sessions.get((req.headers.cookie || '').match(/kitchen_fixture=([^;]+)/)?.[1]); }
const server = http.createServer(async (req, res) => {
  try {
    const url = new URL(req.url, 'http://127.0.0.1'); let body = {};
    if (req.method === 'POST') { let raw = ''; for await (const part of req) raw += part; body = JSON.parse(raw || '{}'); }
    if (!url.pathname.startsWith('/api/')) { const file = {'/': 'index.html', '/app.js': 'app.js', '/renderer.js': 'renderer.js', '/style.css': 'style.css', '/favicon.svg': 'favicon.svg'}[url.pathname]; if (!file) return send(res, 404, {error: 'Missing file'}); const data = fs.readFileSync(file === 'app.js' && appScript ? appScript : path.join(root, 'ui/cooperative_kitchen_web', file)); res.writeHead(200, {'Content-Type': file.endsWith('.js') ? 'text/javascript' : file.endsWith('.css') ? 'text/css' : file.endsWith('.svg') ? 'image/svg+xml' : 'text/html'}); return res.end(data); }
    if (url.pathname === '/api/status') return send(res, 200, {study_ready: false, enrollment: {mode: 'internal_pilot', enabled: true, formal_ready: false, participant_id_pattern: '^[A-Za-z][A-Za-z0-9_-]{2,31}$', participant_id_example: 'user_01'}, policy_kind: 'isolated_protocol_fixture', max_steps: 180, target_orders: 2});
    let session = sessionFor(req);
    if (url.pathname === '/api/session') {
      if (session?.creates.has(body.operation_id)) return send(res, 200, publicView(session));
      if (session?.retrySession && body.mode === 'pilot' && body.participant_id.toLowerCase() === session.run.participant_id.toLowerCase()) {
        session = session.retrySession; session.creates.add(body.operation_id); sessions.set(session.run.id, session);
        return send(res, 200, publicView(session), {'Set-Cookie': `kitchen_fixture=${session.run.id}; HttpOnly; SameSite=Strict; Path=/`});
      }
      if (body.mode === 'pilot' && !/^[A-Za-z][A-Za-z0-9_-]{2,31}$/.test(body.participant_id)) return send(res, 400, {error: 'Invalid user ID', code: 'invalid_participant_id'});
      const id = crypto.randomUUID(); session = {run: {id, participant_id: body.participant_id || `FIXTURE-${++enrollment}`, phase: body.mode === 'freeplay' ? 'instructions' : 'consent', language: body.language, mode: body.mode, version: 0, episode_index: -1, episode_id: null}, condition: body.participant_id === 'fixture_B' ? 'B' : 'A', episodes: [], questions: [], operations: new Map(), creates: new Set([body.operation_id]), draft: {}, exposures: []}; sessions.set(id, session); return send(res, 200, publicView(session), {'Set-Cookie': `kitchen_fixture=${id}; HttpOnly; SameSite=Strict; Path=/`});
    }
    if (url.pathname === '/api/view') return send(res, 200, session ? publicView(session) : {run: null});
    if (!session) return send(res, 404, {error: 'No session'});
    if (url.pathname === '/api/history') { const episode = session.episodes.find(e => e.id === url.searchParams.get('episode_id')); if (!episode) return send(res, 404, {error: 'No episode'}); return send(res, 200, {episode_id: episode.id, frames: copy(episode.frames), version: session.run.version}); }
    if (url.pathname.startsWith('/api/question/')) { if (!qa(session)) return send(res, 403, {error: 'Questions unavailable'}); const job = session.questions.find(q => q.id === url.pathname.split('/').at(-1)); if (!job) return send(res, 404, {error: 'Question missing'}); if (++job.polls >= job.pollTarget) job.status = 'complete'; return send(res, 200, {...job, answer: job.status === 'complete' ? job.answer : null, version: session.run.version}); }
    if (url.pathname === '/api/exposure') { if (!qa(session)) return send(res, 403, {error: 'No exposure permission'}); if (!session.exposures.some(e => e.operation_id === body.operation_id)) session.exposures.push(body); return send(res, 200, {ok: true}); }
    const existing = session.operations.get(body.operation_id);
    if (existing) { if (existing.hash !== hash(body)) return send(res, 409, {error: 'Operation mismatch'}); return send(res, 200, existing.response.run ? publicView(session) : existing.response); }
    if (body.version !== session.run.version) return send(res, 409, {error: 'Stale version'});
    if (url.pathname === '/api/question') {
      if (!qa(session)) return send(res, 403, {error: 'Questions unavailable'});
      if (rejectNextQuestionCode) { const code = rejectNextQuestionCode; rejectNextQuestionCode = null; return send(res, 429, {error: 'UNLOCALIZED BACKEND MESSAGE', code}); }
      const id = crypto.randomUUID(), text = session.run.language === 'zh' ? '这是隔离协议夹具的回答。它只验证回放帧与界面绑定，不证明任何神经策略的解释准确率。\n'.repeat(14) : 'This is an isolated protocol-fixture answer. It verifies frame binding and rendering, not neural explanation accuracy.\n'.repeat(14);
      const job = {id, status: 'queued', ...body, answer: {title: {zh: '所选状态的行为解释', en: 'Explanation for the selected state'}, text, frame: body.frame, kind: body.kind, verified: true, source_summary: {zh: '来源：隔离协议测试夹具。', en: 'Source: isolated protocol fixture.'}}, polls: 0, pollTarget: lateQuestion ? 20 : 1};
      session.questions.push(job); session.run.version++; const response = {id, status: 'queued', version: session.run.version}; session.operations.set(body.operation_id, {hash: hash(body), response}); return send(res, 200, response);
    }
    if (url.pathname !== '/api/command') return send(res, 404, {error: 'Unknown endpoint'});
    const stage = session.run.phase, episode = active(session);
    if (body.command === 'consent') { assert.equal(stage, 'consent'); assert.equal(body.accepted, true); session.run.phase = 'instructions'; }
    else if (body.command === 'next') {
      if (stage === 'instructions') startEpisode(session, 0, session.run.mode === 'freeplay' ? 'freeplay' : 'practice');
      else if (stage === 'practice' && episode.done) startEpisode(session, 1, 'task1');
      else if (['task1', 'task2'].includes(stage) && episode.done) { if (episode.index >= 6) session.run.phase = 'questionnaire'; else startEpisode(session, episode.index + 1, episode.index + 1 <= 3 ? 'task1' : 'task2'); }
      else return send(res, 403, {error: 'Round incomplete'});
    } else if (body.command === 'action') {
      if (!episode || episode.done) return send(res, 403, {error: 'No active round'});
      const result = Demo.step(episode.state, body.action); episode.state = {...result.state, score: result.state.orders * 100 - result.state.turn};
      // Abbreviated protocol fixture only: end every round after four UI commands.
      if (episode.state.turn >= 4) { episode.state.orders = 2; episode.state.done = true; episode.state.reason = 'success'; episode.state.score = 196; }
      episode.frames.push(copy(episode.state)); episode.done = episode.state.done; episode.summary = {orders: episode.state.orders, steps: episode.state.turn, score: episode.state.score};
    } else if (body.command === 'language') session.run.language = body.language;
    else if (['restart', 'swap'].includes(body.command)) { session.releaseChanged = false; if (session.run.mode !== 'freeplay') return send(res, 403, {error: 'No restart permission'}); const preset = body.command === 'swap' && episode.state.preset === 'supply' ? 'cook' : 'supply'; startEpisode(session, session.run.episode_index + 1, 'freeplay'); active(session).state = {...initial(preset), episode_id: session.run.episode_id}; active(session).frames = [copy(active(session).state)]; }
    else if (body.command === 'survey_save') { if (stage !== 'questionnaire') return send(res, 403, {error: 'Survey unavailable'}); session.draft = copy(body.answers); }
    else if (body.command === 'survey_submit') { if (stage !== 'questionnaire') return send(res, 403, {error: 'Survey unavailable'}); assert.ok(body.answers.prediction && body.answers.confidence); session.draft = copy(body.answers); session.run.phase = 'complete'; }
    else return send(res, 400, {error: 'Unknown command'});
    session.run.version++; const response = publicView(session); session.operations.set(body.operation_id, {hash: hash(body), response});
    if (body.command === 'action' && dropNextAction) { dropNextAction = false; req.socket.destroy(); return; }
    send(res, 200, response);
  } catch (error) { send(res, 500, {error: error.message}); }
});

let browser;
const pageView = page => page.evaluate(() => fetch('/api/view').then(r => r.json()));
async function shot(page, name) { await page.evaluate(() => scrollTo(0, 0)); await page.screenshot({path: path.join(output, `${name}.png`), fullPage: true}); report.screenshots.push(`${name}.png`); }
async function layout(page, size) { await page.evaluate(() => scrollTo(0, 0)); for (const selector of ['#board', '[data-action="UP"]', '[data-action="DOWN"]', '[data-action="LEFT"]', '[data-action="RIGHT"]', '[data-action="INTERACT"]', '[data-action="WAIT"]']) { const bounds = await page.locator(selector).boundingBox(); assert.ok(bounds && bounds.y >= -1 && bounds.x >= -1 && bounds.x + bounds.width <= size.width + 1 && bounds.y + bounds.height <= size.height + 1, `${selector} outside ${size.width}x${size.height}`); } assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)); }
async function readyAction(page) { await page.waitForFunction(() => !document.querySelector('[data-action="WAIT"]').disabled); }
async function input(page, action = 'WAIT') { const before = await pageView(page); await readyAction(page); await page.locator(`[data-action="${action}"]`).click(); await page.waitForFunction(turn => document.querySelector('#turn-count').textContent.startsWith(`${turn} /`), before.state.turn + 1); }
async function finishRound(page) { while (!(await pageView(page)).state.done) await input(page); await page.locator('#next').waitFor({state: 'visible'}); }
async function begin(page, target, participantId = null) { await page.goto(target); await page.locator('#lobby').waitFor({state: 'visible'}); if (participantId) { await page.locator('#user-id').fill(['A', 'B'].includes(participantId) ? `fixture_${participantId}` : participantId); await page.locator('#join-study').click(); await page.locator('#consent').waitFor({state: 'visible'}); await page.locator('#consent-check').check(); await page.locator('#accept-consent').click(); } else await page.locator('#freeplay').click(); await page.locator('#start-practice').waitFor({state: 'visible'}); await page.locator('#start-practice').click(); await page.locator('#board').waitFor({state: 'visible'}); }

async function requestLayoutRegression(target) {
  Object.assign(report, {fixture: 'Isolated protocol fixture. Browser layout, native keyboard input and lost-response retry only; no remote requests, training or explanation validation.', app_script_sha256: crypto.createHash('sha256').update(fs.readFileSync(appScript || path.join(root, 'ui/cooperative_kitchen_web/app.js'))).digest('hex')});
  const selectors = ['#board', ...['UP', 'DOWN', 'LEFT', 'RIGHT', 'INTERACT', 'WAIT'].map(action => `[data-action="${action}"]`)];
  async function startSampling(page) {
    await page.evaluate(selectors => {
      window.layoutSamples = [];
      const sample = () => {
        window.layoutSamples.push({scrollX, scrollY, busy: /正在确认|Confirming/.test(document.querySelector('#connection-status').textContent), boxes: selectors.map(selector => { const {x, y, width, height} = document.querySelector(selector).getBoundingClientRect(); return {x, y, width, height}; })});
        window.layoutSampleRAF = requestAnimationFrame(sample);
      };
      sample();
    }, selectors);
  }
  async function stopSampling(page, name, detail) {
    const samples = await page.evaluate(() => { cancelAnimationFrame(window.layoutSampleRAF); return window.layoutSamples; });
    const spread = values => Math.max(...values) - Math.min(...values);
    const boxes = selectors.map((selector, index) => ({selector, ...Object.fromEntries(['x', 'y', 'width', 'height'].map(key => [key, spread(samples.map(sample => sample.boxes[index][key]))]))}));
    const scroll = {x: spread(samples.map(sample => sample.scrollX)), y: spread(samples.map(sample => sample.scrollY))};
    const maximum = Math.max(scroll.x, scroll.y, ...boxes.flatMap(box => ['x', 'y', 'width', 'height'].map(key => box[key])));
    const pendingSamples = samples.filter(sample => sample.busy).length;
    record(name, {...detail, passed: maximum <= 0.5 && pendingSamples >= 5, samples: samples.length, pending_samples: pendingSamples, maximum_movement_px: maximum, scroll_range_px: scroll, box_ranges_px: boxes});
  }
  for (const size of [{width: 1365, height: 900}, {width: 1280, height: 800}]) for (const language of ['zh', 'en']) {
    // Keep real action animation enabled; a reduced-motion fixture would miss
    // frame changes occurring between acknowledgement and re-enabled controls.
    const context = await browser.newContext({viewport: size}), page = await context.newPage();
    page.on('pageerror', error => report.errors.push(error.stack || error.message));
    await begin(page, target); await readyAction(page);
    if (language === 'en') { await page.locator('#language').click(); await page.waitForFunction(() => document.documentElement.lang === 'en'); await readyAction(page); }
    await layout(page, size);
    let actionsReceived = 0;
    await page.route('**/api/command', async route => {
      if (route.request().postDataJSON()?.command !== 'action') return route.continue();
      actionsReceived++;
      const response = await route.fetch();
      // The authoritative commit already exists while the user is waiting.
      await new Promise(resolve => setTimeout(resolve, 420));
      await route.fulfill({response});
    });
    const detail = {viewport: size, language};
    for (const inputMode of ['keyboard', 'pointer']) {
      await page.locator('#board').focus(); await page.evaluate(() => scrollTo(0, 0));
      const before = await pageView(page); await startSampling(page);
      if (inputMode === 'keyboard') await page.keyboard.press('ArrowUp');
      else await page.locator('[data-action="WAIT"]').click();
      await page.waitForFunction(turn => document.querySelector('#turn-count').textContent.startsWith(`${turn} /`), before.state.turn + 1);
      await readyAction(page); await page.waitForTimeout(70);
      await stopSampling(page, 'action_keeps_map_controls_and_scroll_stable', {...detail, input: inputMode});
      assert.equal((await pageView(page)).state.turn, before.state.turn + 1);
    }
    await page.locator('#board').focus(); await page.evaluate(() => {
      scrollTo(0, 0); window.layoutKeyEvents = [];
      document.addEventListener('keydown', event => window.layoutKeyEvents.push({key: event.key, repeat: event.repeat, trusted: event.isTrusted}), {capture: true});
    });
    const holdBefore = await pageView(page), requestsBefore = actionsReceived; await startSampling(page);
    await page.keyboard.down('ArrowDown');
    // Repeated keyboard.down emits trusted native repeat events, unlike a
    // hand-created KeyboardEvent which cannot exercise default page scrolling.
    for (let index = 0; index < 9; index++) { await page.waitForTimeout(85); await page.keyboard.down('ArrowDown'); }
    await page.keyboard.up('ArrowDown'); await readyAction(page); await page.waitForTimeout(120);
    const keyEvents = await page.evaluate(() => window.layoutKeyEvents.filter(event => event.key === 'ArrowDown'));
    assert.equal(keyEvents.length, 10); assert.ok(keyEvents.every(event => event.trusted)); assert.equal(keyEvents.filter(event => event.repeat).length, 9);
    await stopSampling(page, 'held_native_arrow_keeps_scroll_stable', {...detail, repeated_keydown_events: 9});
    assert.equal((await pageView(page)).state.turn, holdBefore.state.turn + 1); assert.equal(actionsReceived, requestsBefore + 1);
    record('held_native_arrow_advances_only_once', detail);

    // Game shortcuts must not steal editable input or the native button
    // activation performed on Space keyup by the browser.
    await page.locator('#question-input').fill('text'); await page.locator('#question-input').focus();
    const inputTurn = (await pageView(page)).state.turn;
    await page.keyboard.press('Space'); await page.keyboard.press('ArrowLeft');
    assert.equal(await page.locator('#question-input').inputValue(), 'text '); assert.equal((await pageView(page)).state.turn, inputTurn);
    await page.locator('[data-action="WAIT"]').focus(); const buttonRequests = actionsReceived;
    await page.keyboard.down('Space'); await page.waitForTimeout(80); assert.equal(actionsReceived, buttonRequests);
    await page.keyboard.up('Space'); await page.waitForFunction(turn => document.querySelector('#turn-count').textContent.startsWith(`${turn} /`), inputTurn + 1);
    assert.equal(actionsReceived, buttonRequests + 1); assert.equal((await pageView(page)).state.turn, inputTurn + 1);
    record('editable_space_and_native_button_space_are_preserved', detail);
    await context.close();
  }
  const context = await browser.newContext({viewport: {width: 1280, height: 800}}), page = await context.newPage();
  page.on('pageerror', error => report.errors.push(error.stack || error.message));
  await begin(page, target); await readyAction(page);
  const before = await pageView(page), captured = []; let dropped = false;
  await page.route('**/api/command', async route => {
    const body = route.request().postDataJSON();
    if (body.command !== 'action') return route.continue();
    captured.push(copy(body));
    if (!dropped) { dropped = true; await route.fetch(); await route.abort('failed'); }
    else await route.continue();
  });
  await page.locator('[data-action="WAIT"]').click(); await page.locator('#retry-request').waitFor({state: 'visible'});
  assert.equal(await page.locator('#notice').isVisible(), true); assert.equal(await page.locator('[data-action="WAIT"]').isDisabled(), true);
  assert.equal((await pageView(page)).state.turn, before.state.turn + 1);
  await page.locator('#retry-request').click(); await page.waitForFunction(() => document.querySelector('#notice').hidden); await readyAction(page);
  const after = await pageView(page); assert.equal(after.state.turn, before.state.turn + 1);
  assert.equal(captured.length, 2); assert.deepEqual(captured[0], captured[1]);
  const session = sessions.get(after.run.id);
  assert.equal([...session.operations.keys()].filter(id => id === captured[0].operation_id).length, 1);
  assert.equal(active(session).frames.length, 2);
  record('lost_committed_response_has_retry_and_one_operation_one_step', {action_requests: captured.length, committed_actions: active(session).frames.length - 1});
  await context.close(); assert.deepEqual(report.errors, []);
  const failed = report.checks.filter(check => !check.passed); assert.equal(failed.length, 0, `${failed.length} layout checks failed; see per-frame measurements in report`);
  report.status = 'passed';
}

async function actualAutoDemonstration() {
  assert.ok(['127.0.0.1', 'localhost'].includes(new URL(autoURL).hostname));
  Object.assign(report, {fixture: 'Actual server freeplay; no physics or policy fixtures. The server program controls only the player; the selected neural Actor controls the AI.', real_server: autoURL, cloud_model_validation: false});
  browser = await chromium.launch({headless: true, executablePath: process.env.KITCHEN_CHROME || process.env.CANDY_CHROME || undefined});
  const context = await browser.newContext({viewport: {width:1280,height:800}, reducedMotion:'reduce'}), page = await context.newPage();
  page.on('pageerror', error => report.errors.push(error.stack || error.message));
  await begin(page,autoURL); const status = await page.evaluate(() => fetch('/api/status').then(r=>r.json()));
  assert.equal(status.policy_kind,'neural'); report.versions=status.versions; report.test_mode=status.test_mode;
  assert.equal((await pageView(page)).can_auto,true); await layout(page,{width:1280,height:800});
  assert.match(await page.locator('#auto-description').innerText(),/程序玩家.*神经策略/);
  const auto=page.locator('#auto-demo');
  const waitTurn=async turn=>page.waitForFunction(turn=>Number(document.querySelector('#turn-count').textContent.split('/')[0])>=turn,turn);
  const idle=async()=>page.waitForFunction(()=>!/正在确认|Confirming/.test(document.querySelector('#connection-status').textContent));
  const stable=async()=>{await idle();const turn=(await pageView(page)).state.turn;await page.waitForTimeout(700);assert.equal((await pageView(page)).state.turn,turn);assert.equal(await auto.getAttribute('aria-pressed'),'false');return turn;};
  await auto.click();await waitTurn(2);await auto.click();await stable();record('actual_auto_explicit_pause');
  let turn=(await pageView(page)).state.turn;await auto.click();await waitTurn(turn+1);await page.locator('[data-explain="why"]').click();await page.locator('#explanation').waitFor({state:'visible'});await stable();assert.equal(await auto.isDisabled(),true);await shot(page,'auto-paused-explanation-1280x800-zh');await page.locator('#close-explanation').click();record('actual_auto_pauses_for_question_and_answer_reading');
  turn=(await pageView(page)).state.turn;await auto.click();await waitTurn(turn+1);await page.locator('#previous-frame').click();await stable();assert.equal(await auto.isDisabled(),true);await page.locator('#live').click();record('actual_auto_pauses_for_replay');
  turn=(await pageView(page)).state.turn;await auto.click();await waitTurn(turn+1);await page.reload();await page.locator('#board').waitFor({state:'visible'});await stable();record('actual_auto_does_not_resume_after_refresh');
  turn=(await pageView(page)).state.turn;let dropped=false;
  await page.route('**/api/command',async route=>{if(!dropped&&route.request().postDataJSON()?.command==='auto_step'){dropped=true;await route.fetch();await route.abort('failed');}else await route.continue();});
  await auto.click();await page.locator('#retry-request').waitFor({state:'visible'});assert.equal(await auto.getAttribute('aria-pressed'),'false');await page.locator('#retry-request').click();await page.waitForFunction(()=>document.querySelector('#notice').hidden);await stable();assert.equal((await pageView(page)).state.turn,turn+1);await page.unroute('**/api/command');record('actual_auto_lost_response_retries_once_and_remains_paused');
  turn=(await pageView(page)).state.turn;await auto.click();await waitTurn(turn+1);await page.locator('#board').focus();await page.keyboard.press('ArrowUp');await stable();record('actual_auto_manual_keyboard_input_stops_scheduling');
  await page.locator('#restart').click();await waitTurn(0);await page.waitForFunction(()=>document.querySelector('#turn-count').textContent.startsWith('0 /'));
  for(const preset of ['supply','cook']) {
    if(preset==='cook'){await page.locator('#swap-role').click();await page.waitForFunction(()=>document.querySelector('#turn-count').textContent.startsWith('0 /'));await page.locator('#language').click();await page.waitForFunction(()=>document.documentElement.lang==='en');}
    const before=await pageView(page);assert.equal(before.state.actors[0].side,preset==='supply'?'left':'right');
    await auto.click();await page.waitForFunction(()=>!document.querySelector('#result').hidden,{},{timeout:150000});await stable();
    const done=await pageView(page);assert.equal(done.state.orders,2);assert.equal(done.state.done,true);assert.ok(done.state.turn<=180);assert.equal(done.can_auto,false);
    await shot(page,`auto-complete-${preset}-1280x800-${preset==='supply'?'zh':'en'}`);record('actual_auto_completes_two_soups_with_neural_teammate',{preset,turns:done.state.turn,orders:done.state.orders});
    process.stdout.write(`Actual automatic demonstration ${preset}: ${done.state.orders} soups in ${done.state.turn} turns.\n`);
  }
  assert.deepEqual(report.errors,[]);record('actual_auto_no_javascript_errors');report.status='passed';await context.close();
}

async function fullActualStudy() {
  const fixture = JSON.parse(fs.readFileSync(fullFixture, 'utf8'));
  assert.equal(fixture.test_only, true); assert.equal(fixture.dynamics_patched, false);
  assert.ok(['127.0.0.1', 'localhost'].includes(new URL(fixture.url).hostname));
  Object.assign(report, {fixture: 'Isolated PostgreSQL test schema; actual KitchenStudy, trained Actor, evidence engine and frozen eight-question/six-scenario artifacts. Full physics and 180-step horizon. Only release gates bypassed in explicit test_mode, never a recruitment or model-quality acceptance.', real_server: fixture.url, versions: fixture.versions, test_only: true, release_gate_bypass: true, cloud_model_validation: false, dynamics_patched: false, complete_episode_count: 0, committed_actions: 0});
  browser = await chromium.launch({headless: true, executablePath: process.env.KITCHEN_CHROME || process.env.CANDY_CHROME || undefined});
  for (const condition of ['A', 'B']) {
    const size = condition === 'A' ? {width: 1365, height: 900} : {width: 1280, height: 800};
    const context = await browser.newContext({viewport: size, reducedMotion: 'reduce'}), page = await context.newPage();
    page.on('pageerror', error => report.errors.push(error.stack || error.message));
    await page.goto(fixture.url);
    const status = await page.evaluate(() => fetch('/api/status').then(r => r.json()));
    assert.equal(status.namespace, 'test'); assert.equal(status.test_mode, true); assert.equal(status.storage, 'postgresql'); assert.equal(status.policy_kind, 'neural');
    await begin(page, fixture.url, fixture.participant_id_by_condition[condition]);
    if (condition === 'B') { await page.locator('#language').click(); await page.waitForFunction(() => document.documentElement.lang === 'en'); }
    await layout(page, size); await shot(page, `full-practice-${condition.toLowerCase()}-${size.width}x${size.height}`);
    assert.equal(await page.locator('#auto-controls').isVisible(),false);assert.equal((await pageView(page)).can_auto,false);
    assert.equal(await page.locator('#restart').isVisible(), false); assert.equal(await page.locator('#swap-role').isVisible(), false);
    for (let round = 0; round <= 6; round++) {
      let current = await pageView(page);
      assert.equal(current.run.episode_index, round); assert.equal(current.state.maxSteps, 180); assert.equal(current.state.targetOrders, 2); assert.equal(current.state.actors[0].side, 'left');
      const expectedPhase = round === 0 ? 'practice' : round <= 3 ? 'task1' : 'task2';
      assert.equal(current.run.phase, expectedPhase);
      const questionEnabled = condition === 'A' && expectedPhase === 'task1';
      assert.equal(current.can_ask, questionEnabled); assert.equal(await page.locator('#question-controls').isVisible(), questionEnabled);
      if (condition === 'A' && round === 1) {
        const frozen = hash(current.state);
        for (const kind of ['why', 'waiting', 'counterfactual']) {
          if (await page.locator('#explanation').isVisible()) await page.locator('#close-explanation').click();
          await page.locator(`[data-explain="${kind}"]`).click(); await page.locator('#explanation').waitFor({state: 'visible'});
          assert.equal(hash((await pageView(page)).state), frozen);
          assert.ok((await page.locator('#explanation-text').innerText()).length > 20);
        }
        await shot(page, 'full-task1-a-actual-evidence-1365x900-zh');
        await page.locator('#close-explanation').click();
        await page.locator('#question-input').fill('如果我连续等待三步，队友会怎么做？'); await page.locator('#ask-question').click();
        await page.locator('#explanation').waitFor({state: 'visible'}); assert.equal(hash((await pageView(page)).state), frozen);
        await page.locator('#close-explanation').click();
        record('full_actual_task1_a_shortcuts_free_question_and_state_isolation', {cloud_model_validation: false});
      }
      if (expectedPhase === 'task2') {
        assert.equal(await page.locator('#explanation').isVisible(), false); assert.equal(await page.locator('#past-answers').innerText(), '');
        if (round === 4) await shot(page, `full-task2-${condition.toLowerCase()}-${size.width}x${size.height}`);
      }
      current = await pageView(page);
      while (!current.state.done) {
        const turn = current.state.turn;
        const response = page.waitForResponse(r => r.url().endsWith('/api/command') && r.request().method() === 'POST' && r.request().postDataJSON()?.command === 'action');
        await page.locator('[data-action="WAIT"]').click();
        const acknowledged = await response; assert.equal(acknowledged.status(), 200); current = await acknowledged.json();
        assert.equal(current.state.turn, turn + 1); report.committed_actions++;
        if (turn + 1 === 90 && (round === 0 || round === 4)) {
          await page.reload(); await page.locator('#board').waitFor({state: 'visible'});
          current = await pageView(page); assert.equal(current.state.turn, 90);
          assert.equal(current.can_ask, questionEnabled); record('full_actual_mid_round_refresh_restores_committed_state', {condition, round, turn: 90});
        }
      }
      assert.equal(current.state.turn, 180); assert.equal(current.state.done, true);
      assert.equal(current.state.score, current.state.orders * 100 - 180);
      report.complete_episode_count++;
      record('full_actual_180_step_episode_saved', {condition, round, phase: expectedPhase, score: current.state.score, orders: current.state.orders, episode_id: current.run.episode_id});
      process.stdout.write(`Completed ${condition} ${expectedPhase} episode ${round}: 180 acknowledged UI actions.\n`);
      await page.locator('#next').click();
      if (round < 6) await page.waitForFunction(() => document.querySelector('#turn-count').textContent.startsWith('0 /'));
    }
    await page.locator('#survey').waitFor({state: 'visible'});
    const survey = (await pageView(page)).survey;
    assert.equal(survey.items.filter(i => i.type === 'prediction').length, 4); assert.equal(survey.items.filter(i => i.type === 'counterfactual').length, 4); assert.equal(survey.items.filter(i => i.type === 'likert').length, 3);
    assert.equal(await page.locator('.prediction-board').count(), 8);
    assert.ok(survey.items.every(i => !('correct_answer' in i)));
    for (const item of survey.items.filter(i => i.state)) {
      const canvas = page.locator(`[data-item="${item.id}"] .prediction-board`);
      assert.equal(await canvas.getAttribute('data-render-turn'), String(item.state.turn));
    }
    const first = survey.items[0]; await page.locator(`input[name="${first.id}"]`).first().check();
    await page.waitForFunction(() => /已保存|Draft saved/.test(document.querySelector('#survey-save-status').textContent));
    await page.reload(); await page.locator('#survey').waitFor({state: 'visible'}); assert.equal(await page.locator(`input[name="${first.id}"]`).first().isChecked(), true);
    for (const item of survey.items) await page.locator(`input[name="${item.id}"]`).first().check();
    await shot(page, `full-survey-${condition.toLowerCase()}-${size.width}x${size.height}`);
    await page.locator('#submit-survey').click(); await page.locator('#complete').waitFor({state: 'visible'});
    await page.reload(); await page.locator('#complete').waitFor({state: 'visible'});
    assert.ok((await page.locator('#completion-code').innerText()).length > 0);
    record('full_actual_eight_question_three_scale_survey_and_completion_recovery', {condition, items: 11, shared_renderer_maps: 8});
    await context.close();
  }
  assert.equal(report.complete_episode_count, 14); assert.equal(report.committed_actions, 2520); assert.deepEqual(report.errors, []);
  record('full_actual_no_javascript_errors'); report.status = 'passed';
}

(async () => {
  try {
    if (autoURL) { await actualAutoDemonstration(); return; }
    if (fullFixture) { await fullActualStudy(); return; }
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve)); const target = `http://127.0.0.1:${server.address().port}`;
    browser = await chromium.launch({headless: true, executablePath: process.env.KITCHEN_CHROME || process.env.CANDY_CHROME || undefined});
    if (layoutOnly) { await requestLayoutRegression(target); return; }
    const context = await browser.newContext({viewport: {width: 1365, height: 900}, reducedMotion: 'reduce'}), page = await context.newPage(); page.on('pageerror', error => report.errors.push(error.stack || error.message));
    await begin(page, target); assert.equal((await pageView(page)).state.maxSteps, 180); assert.equal(await page.evaluate(() => typeof window.KitchenEngine), 'undefined'); assert.equal(await page.locator('#ai-status').count(), 0); record('server_authority_no_client_policy_or_next_action_preview');
    rejectNextQuestionCode = 'question_rate_limit'; await page.locator('[data-explain="why"]').click(); await page.waitForFunction(() => document.querySelector('#notice-text').textContent === '请等待 2 秒后再提问。'); assert.equal(await page.locator('#notice-text').innerText(), '请等待 2 秒后再提问。');
    await page.locator('#language').click(); await page.waitForFunction(() => document.documentElement.lang === 'en'); rejectNextQuestionCode = 'question_budget_exhausted'; await page.locator('[data-explain="why"]').click(); await page.waitForFunction(() => document.querySelector('#notice-text').textContent.includes('internal pilot')); assert.doesNotMatch(await page.locator('#notice-text').innerText(), /UNLOCALIZED/); await page.locator('#language').click(); await page.waitForFunction(() => document.documentElement.lang === 'zh'); record('qa_limits_have_bilingual_participant_messages');
    for (const size of [{width: 1365, height: 900}, {width: 1280, height: 800}]) { await page.setViewportSize(size); for (const lang of ['zh', 'en']) { if (await page.locator('html').getAttribute('lang') !== lang) await page.locator('#language').click(); await page.waitForFunction(lang => document.documentElement.lang === lang, lang); await layout(page, size); await shot(page, `freeplay-${size.width}x${size.height}-${lang}`); record('viewport_and_language', {size, lang}); } }
    const old = (await pageView(page)).state; await page.locator('#board').focus(); await page.keyboard.press('ArrowUp'); await page.waitForFunction(turn => document.querySelector('#turn-count').textContent.startsWith(`${turn} /`), old.turn + 1); assert.notDeepEqual((await pageView(page)).state.actors[0].position, old.actors[0].position); record('keyboard_moves_once');
    const buttonTurn = (await pageView(page)).state.turn; await page.locator('[data-action="INTERACT"]').focus(); await page.keyboard.press('Space'); await page.waitForFunction(turn => document.querySelector('#turn-count').textContent.startsWith(`${turn} /`), buttonTurn + 1); assert.equal((await pageView(page)).state.actors[0].holding, 'onion'); record('focused_button_space_native_interact_not_wait');
    await page.locator('#previous-frame').click(); const frozen = hash((await pageView(page)).state); await page.locator('[data-explain="counterfactual"]').click(); await page.locator('#explanation').waitFor({state: 'visible'}); assert.match(await page.locator('#explanation-frame').innerText(), /1/); assert.equal(hash((await pageView(page)).state), frozen); await shot(page, 'historical-long-answer-1280x800-en'); const answerBounds = await page.locator('#explanation-text').boundingBox(); assert.ok(answerBounds.height > 300, 'Long answer remains readable without clipping'); record('historical_question_binding_and_isolation');
    await page.locator('#live').click(); await readyAction(page); const beforeDrop = (await pageView(page)).state.turn; let disconnectOnce = true; await page.route('**/api/command', async route => { if (disconnectOnce && route.request().postDataJSON()?.command === 'action') { disconnectOnce = false; await route.fetch(); await route.abort('failed'); } else await route.continue(); }); await page.locator('[data-action="WAIT"]').click(); await page.locator('#retry-request').waitFor({state: 'visible'}); await page.reload(); await page.waitForFunction(turn => document.querySelector('#turn-count').textContent.startsWith(`${turn} /`) && document.querySelector('#notice').hidden, beforeDrop + 1); assert.equal((await pageView(page)).state.turn, beforeDrop + 1); record('lost_response_reload_retries_same_operation_once'); await page.unroute('**/api/command');
    await page.locator('#swap-role').click(); await page.waitForFunction(() => document.querySelector('#turn-count').textContent.startsWith('0 /')); assert.equal((await pageView(page)).state.preset, 'cook'); await shot(page, 'freeplay-swapped-1280x800-en'); await page.locator('#restart').click(); await page.waitForFunction(() => document.querySelector('#board').dataset.renderPreset === 'supply'); record('freeplay_restart_and_swap'); const cookie = (await context.cookies()).find(c => c.name === 'kitchen_fixture'); sessions.get(cookie.value).releaseChanged = true; await page.reload(); await page.locator('#release-recovery').waitFor({state: 'visible'}); assert.equal(await page.locator('#play').isVisible(), false); await page.locator('#recover-restart').click(); await page.locator('#board').waitFor({state: 'visible'}); assert.equal((await pageView(page)).state.turn, 0); record('changed_release_requires_explicit_freeplay_restart'); await context.close();

    const retryContext = await browser.newContext({viewport: {width: 1280, height: 800}, reducedMotion: 'reduce'}), retryPage = await retryContext.newPage();
    retryPage.on('pageerror', error => report.errors.push(error.stack || error.message));
    await begin(retryPage, target, 'RetryUser01');
    const retryCookie = (await retryContext.cookies()).find(c => c.name === 'kitchen_fixture');
    const previousRun = sessions.get(retryCookie.value), retryId = crypto.randomUUID();
    const nextRun = {run: {...previousRun.run, id: retryId, phase: 'consent', version: 0, episode_index: -1, episode_id: null}, condition: previousRun.condition,
      episodes: [], questions: [], operations: new Map(), creates: new Set(), draft: {}, exposures: []};
    previousRun.run.phase = 'technical_retry_closed'; previousRun.run.version++; previousRun.retrySession = nextRun; sessions.set(retryId, nextRun);
    await retryPage.reload(); await retryPage.locator('#technical-retry').waitFor({state: 'visible'});
    assert.equal(await retryPage.locator('#play').isVisible(), false); await retryPage.locator('#resume-technical-retry').click();
    await retryPage.locator('#consent').waitFor({state: 'visible'}); assert.equal((await pageView(retryPage)).run.id, retryId);
    record('technical_retry_has_browser_recovery_path'); await retryContext.close();

    const syncContext = await browser.newContext({viewport: {width: 1365, height: 900}, reducedMotion: 'reduce'}), firstTab = await syncContext.newPage();
    firstTab.on('pageerror', error => report.errors.push(error.stack || error.message));
    await begin(firstTab, target, 'SyncUser01'); await finishRound(firstTab); await firstTab.locator('#next').click();
    await firstTab.waitForFunction(() => document.querySelector('#phase-label').textContent === 'Task 1');
    await firstTab.locator('[data-explain="why"]').click(); await firstTab.locator('#explanation').waitFor({state: 'visible'});
    const lagTab = await syncContext.newPage(); lagTab.on('pageerror', error => report.errors.push(error.stack || error.message));
    let releaseLag, capturedLag; const lagReleased = new Promise(resolve => { releaseLag = resolve; }); const lagCaptured = new Promise(resolve => { capturedLag = resolve; }); let delayInitialView = true;
    await lagTab.route('**/api/view', async route => { if (!delayInitialView) return route.continue(); delayInitialView = false; const response = await route.fetch(); capturedLag(); await lagReleased; await route.fulfill({response}); });
    const lagNavigation = lagTab.goto(target); await lagCaptured; await lagNavigation;
    const secondTab = await syncContext.newPage(); secondTab.on('pageerror', error => report.errors.push(error.stack || error.message));
    await secondTab.goto(target); await secondTab.locator('#board').waitFor({state: 'visible'});
    let releaseAuthority, capturedAuthority; const authorityReleased = new Promise(resolve => { releaseAuthority = resolve; }); const authorityCaptured = new Promise(resolve => { capturedAuthority = resolve; }); let delayAuthorityView = true;
    await firstTab.route('**/api/view', async route => { if (!delayAuthorityView) return route.continue(); delayAuthorityView = false; const response = await route.fetch(); capturedAuthority(); await authorityReleased; await route.fulfill({response}); });
    await input(secondTab); await authorityCaptured;
    for (let round = 0; round < 3; round++) {
      await finishRound(secondTab); await secondTab.locator('#next').click();
      if (round < 2) await secondTab.waitForFunction(() => document.querySelector('#turn-count').textContent.startsWith('0 /'));
      else await secondTab.waitForFunction(() => document.querySelector('#phase-label').textContent === 'Task 2');
    }
    releaseAuthority(); releaseLag(); await lagTab.waitForFunction(() => document.querySelector('#phase-label').textContent === 'Task 2');
    await firstTab.waitForFunction(() => document.querySelector('#phase-label').textContent === 'Task 2');
    assert.equal(await firstTab.locator('#explanation').isVisible(), false); assert.equal(await firstTab.locator('#past-answers').innerText(), '');
    assert.equal(await lagTab.locator('#question-controls').isVisible(), false); assert.equal(await lagTab.locator('#past-answers').innerText(), '');
    assert.equal((await pageView(firstTab)).can_ask, false); record('cross_tab_task2_immediately_hides_task1_answers');
    record('queued_authority_refresh_converges_after_delayed_old_response');
    record('delayed_task1_boot_response_cannot_cross_task2_sync_watermark'); await syncContext.close();

    const rawContext = await browser.newContext({viewport: {width: 1280, height: 800}, reducedMotion: 'reduce'}), rawPage = await rawContext.newPage();
    rawPage.on('pageerror', error => report.errors.push(error.stack || error.message));
    await begin(rawPage, target, 'RawSync01'); await finishRound(rawPage); await rawPage.locator('#next').click(); await rawPage.waitForFunction(() => document.querySelector('#phase-label').textContent === 'Task 1');
    await rawPage.locator('[data-explain="why"]').click(); await rawPage.locator('#explanation').waitFor({state: 'visible'});
    await rawPage.evaluate(async () => {
      let state = await fetch('/api/view', {cache: 'no-store'}).then(response => response.json());
      const post = body => fetch('/api/command', {method: 'POST', credentials: 'same-origin', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)}).then(response => response.json());
      while (state.run.phase === 'task1') {
        while (!state.state.done) state = await post({operation_id: crypto.randomUUID(), version: state.run.version, command: 'action', action: 'WAIT'});
        state = await post({operation_id: crypto.randomUUID(), version: state.run.version, command: 'next'});
      }
    });
    assert.equal((await pageView(rawPage)).run.phase, 'task2');
    await rawPage.waitForFunction(() => document.querySelector('#phase-label').textContent === 'Task 2' && document.querySelector('#explanation').hidden && document.querySelector('#past-answers').textContent === '', {}, {timeout: 8000});
    assert.equal((await pageView(rawPage)).can_ask, false); record('sparse_authority_check_scrubs_direct_api_phase_change'); await rawContext.close();

    for (const condition of ['A', 'B']) {
      const studyContext = await browser.newContext({viewport: {width: 1365, height: 900}, reducedMotion: 'reduce'}), studyPage = await studyContext.newPage(); studyPage.on('pageerror', error => report.errors.push(error.stack || error.message));
      await begin(studyPage, target, condition); assert.equal(await studyPage.locator('#restart').isVisible(), false); assert.equal(await studyPage.locator('#swap-role').isVisible(), false); assert.equal(await studyPage.locator('#auto-controls').isVisible(),false); assert.equal(await studyPage.locator('#question-controls').isVisible(), false); await finishRound(studyPage); await studyPage.locator('#next').click(); await studyPage.waitForFunction(() => document.querySelector('#phase-label').textContent === 'Task 1');
      if (condition === 'A') { await studyPage.locator('#question-input').fill('为什么你会停在这里？'); await studyPage.locator('#ask-question').click(); await studyPage.locator('#explanation').waitFor({state: 'visible'}); await shot(studyPage, 'task1-a-free-question-1365x900-zh'); record('task1_a_free_question'); } else assert.equal(await studyPage.locator('#question-controls').isVisible(), false);
      for (let round = 1; round <= 6; round++) {
        const snapshot = await pageView(studyPage); assert.equal(snapshot.run.episode_index, round);
        if (condition === 'A' && round === 3) { let delayed = false; await studyPage.route('**/api/question/*', async route => { if (!delayed) { delayed = true; const response = await route.fetch(); await new Promise(resolve => setTimeout(resolve, 1800)); await route.fulfill({response}); } else await route.continue(); }); await studyPage.locator('[data-explain="why"]').click(); await studyPage.locator('#question-pending').waitFor({state: 'visible'}); for (let wait = 0; !delayed && wait < 80; wait++) await studyPage.waitForTimeout(20); assert.ok(delayed, 'Start an in-flight answer before the phase transition'); }
        if (round >= 4) { assert.equal(await studyPage.locator('#question-controls').isVisible(), false); assert.equal(await studyPage.locator('#explanation').isVisible(), false); assert.equal(await studyPage.locator('#past-answers').innerText(), ''); }
        await finishRound(studyPage); await studyPage.locator('#next').click();
        if (round === 3) { await studyPage.waitForFunction(() => document.querySelector('#phase-label').textContent === 'Task 2'); await studyPage.waitForTimeout(2000); assert.equal(await studyPage.locator('#explanation').isVisible(), false); await shot(studyPage, `task2-${condition.toLowerCase()}-1365x900-zh`); }
        if (round < 6) await studyPage.waitForFunction(index => document.querySelector('#round-label').textContent.includes(String(((index - 1) % 3) + 1)) && document.querySelector('#turn-count').textContent.startsWith('0 /'), round + 1);
      }
      await studyPage.locator('#survey').waitFor({state: 'visible'}); assert.equal((await pageView(studyPage)).episodes.filter(e => ['task1', 'task2'].includes(e.phase) && e.done).length, 6); record('full_six_round_flow_permissions', {condition});
      await studyPage.locator('input[name="prediction"][value="WAIT"]').check(); await studyPage.waitForFunction(() => document.querySelector('#survey-save-status').textContent.includes('已保存')); await studyPage.reload(); await studyPage.locator('#survey').waitFor({state: 'visible'}); assert.equal(await studyPage.locator('input[name="prediction"][value="WAIT"]').isChecked(), true); assert.equal(await studyPage.locator('.prediction-board').getAttribute('data-render-turn'), '0'); await studyPage.locator('input[name="confidence"][value="4"]').check(); await studyPage.waitForFunction(() => !document.querySelector('#submit-survey').disabled); await shot(studyPage, `survey-${condition.toLowerCase()}-1365x900-zh`); await studyPage.locator('#submit-survey').click(); await studyPage.locator('#complete').waitFor({state: 'visible'}); await studyPage.reload(); await studyPage.locator('#complete').waitFor({state: 'visible'}); record('survey_draft_common_renderer_submit_and_reload', {condition}); await studyContext.close();
    }
    if (realURL) {
      assert.ok(['127.0.0.1', 'localhost', '[::1]'].includes(new URL(realURL).hostname), 'Real UI verification is limited to a local debug server'); const realContext = await browser.newContext({viewport: {width: 1280, height: 800}, reducedMotion: 'reduce'}), realPage = await realContext.newPage(); realPage.on('pageerror', error => report.errors.push(error.stack || error.message)); await realPage.goto(realURL); await realPage.locator('#lobby').waitFor({state: 'visible'}); const liveStatus = await realPage.evaluate(() => fetch('/api/status').then(r => r.json())); await shot(realPage, 'real-server-lobby-1280x800-zh'); await begin(realPage, realURL); await layout(realPage, {width: 1280, height: 800}); await input(realPage, 'UP'); await input(realPage, 'INTERACT'); await input(realPage, 'WAIT'); const realTurn = (await pageView(realPage)).state.turn; await realPage.reload(); await realPage.locator('#board').waitFor({state: 'visible'}); assert.equal((await pageView(realPage)).state.turn, realTurn); await shot(realPage, 'real-server-freeplay-1280x800-zh'); record('real_server_freeplay_commands_and_recovery', {policy_kind: liveStatus.policy_kind, study_ready: liveStatus.study_ready, turns: realTurn}); const factualBefore = hash((await pageView(realPage)).state); for (const kind of ['why', 'waiting', 'counterfactual']) { if (await realPage.locator('#explanation').isVisible()) await realPage.locator('#close-explanation').click(); await realPage.locator(`[data-explain="${kind}"]`).click(); await realPage.locator('#explanation').waitFor({state:'visible'}); assert.match(await realPage.locator('#explanation-frame').innerText(), new RegExp(String(realTurn))); assert.equal(hash((await pageView(realPage)).state), factualBefore); } await shot(realPage, 'real-server-explanation-1280x800-zh'); record('real_server_frame_bound_shortcuts_do_not_advance', {kinds:['why','waiting','counterfactual'],cloud_model_validation:false}); await realContext.close();
    }
    assert.deepEqual(report.errors, []); report.status = 'passed'; record('no_javascript_errors');
  } catch (error) { report.status = 'failed'; report.failure = error.stack || error.message; process.exitCode = 1; }
  finally { report.finished = new Date().toISOString(); fs.writeFileSync(path.join(output, layoutOnly ? 'layout_browser_acceptance.json' : autoURL ? 'auto_demonstration_acceptance.json' : fullFixture ? 'full_study_browser_acceptance.json' : 'browser_acceptance.json'), JSON.stringify(report, null, 2) + '\n'); if (browser) await browser.close(); if (server.listening) await new Promise(resolve => server.close(resolve)); process.stdout.write(JSON.stringify({status: report.status, checks: report.checks.length, output, failure: report.failure || null}, null, 2) + '\n'); }
})();
