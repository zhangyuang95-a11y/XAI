/* Local cooperative-kitchen demo acceptance. No research services or records.
 * node tests/cooperative_kitchen_browser.cjs [http://127.0.0.1:8002] [output/cooperative_kitchen_demo/v1]
 * KITCHEN_PLAYWRIGHT_MODULE and KITCHEN_CHROME may select the desktop runtime.
 * The test starts no servers and only operates normal UI controls. The exposed
 * KitchenDemo accessors are read-only; engine calls below use detached snapshots.
 */
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const {chromium} = require(process.env.KITCHEN_PLAYWRIGHT_MODULE || process.env.CANDY_PLAYWRIGHT_MODULE || 'playwright');
const [target = 'http://127.0.0.1:8002', output = 'output/cooperative_kitchen_demo/v1'] = process.argv.slice(2);
const keyboardOnly = process.argv.includes('--keyboard-only');
assert.ok(['127.0.0.1', 'localhost', '[::1]'].includes(new URL(target).hostname), 'Use a local demo server only');
const out = path.resolve(output);
fs.mkdirSync(out, {recursive: true});
const viewports = [{width: 1365, height: 900}, {width: 1280, height: 800}];
const report = {
  schema: 'cooperative_kitchen_demo_browser_acceptance_v1', status: 'running',
  started: new Date().toISOString(), target,
  policy: 'deterministic_program_teammate', research_data: false,
  browser: 'Chromium', cases: [], screenshots: [], javascript_errors: [],
};
const save = () => fs.writeFileSync(path.join(out, 'browser_acceptance.json'), JSON.stringify(report, null, 2) + '\n');
const hash = value => crypto.createHash('sha256').update(JSON.stringify(value)).digest('hex');
const record = (name, details = {}) => {report.cases.push({name, passed: true, ...details}); save();};
const state = page => page.evaluate(() => window.KitchenDemo.getState());
const history = page => page.evaluate(() => window.KitchenDemo.getHistory());
const ui = page => page.evaluate(() => window.KitchenDemo.getUI());

async function ready(page) {
  await page.waitForFunction(() => window.KitchenDemo && window.KitchenEngine);
  await page.waitForTimeout(200);
}
async function fresh(browser, viewport = viewports[0]) {
  const context = await browser.newContext({viewport});
  const page = await context.newPage();
  page.on('pageerror', error => report.javascript_errors.push(error.stack || error.message));
  await page.goto(target);
  await ready(page);
  assert.equal((await state(page)).turn, 0);
  assert.equal((await ui(page)).language, 'zh', 'Fresh demo should default to Chinese');
  return {context, page};
}
async function begin(page) {
  if (await page.locator('#start').isVisible()) await page.locator('#start').click();
  await page.locator('#board').waitFor({state: 'visible'});
  await page.waitForTimeout(180);
  assert.equal((await ui(page)).started, true);
}
async function locale(page, language) {
  const before = await state(page);
  if ((await ui(page)).language !== language) await page.locator('#language').click();
  await page.waitForFunction(language => window.KitchenDemo.getUI().language === language, language);
  assert.deepEqual(await state(page), before, 'Changing language advances no game time');
  assert.equal(await page.locator('html').getAttribute('lang'), language);
}
async function screenshot(page, name) {
  await page.waitForTimeout(180);
  await page.evaluate(() => window.scrollTo(0, 0));
  const file = path.join(out, `${name}.png`);
  await page.screenshot({path: file, fullPage: true});
  report.screenshots.push(path.basename(file));
}
async function layout(page, viewport) {
  await page.evaluate(() => window.scrollTo(0, 0));
  for (const selector of ['#board', '#auto-demo', '[data-action="UP"]', '[data-action="DOWN"]',
    '[data-action="LEFT"]', '[data-action="RIGHT"]', '[data-action="INTERACT"]', '[data-action="WAIT"]']) {
    const bounds = await page.locator(selector).boundingBox();
    assert.ok(bounds, `${selector} is not visible`);
    assert.ok(bounds.x >= -1 && bounds.y >= -1 && bounds.x + bounds.width <= viewport.width + 1 &&
      bounds.y + bounds.height <= viewport.height + 1, `${selector} does not fit the initial ${viewport.width} × ${viewport.height} viewport`);
    if (selector === '#board') assert.ok(bounds.width >= 450 && bounds.height >= 300, 'Kitchen must remain legible');
  }
  const bounds = await page.locator('#board').boundingBox();
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - innerWidth);
  assert.ok(overflow <= 1, 'Horizontal page overflow');
  for (const selector of ['#orders-count', '#turn-count', '#pot-status', '#role-label', '#previous-frame', '#next-frame', '#timeline', '#live']) {
    assert.ok(await page.locator(selector).isVisible(), `${selector} missing`);
  }
  assert.match(await page.locator('body').innerText(), /PolicyLens/);
  const description = await page.locator('body').innerText();
  assert.match(description, (await ui(page)).language === 'zh' ? /程序队友/ : /program(?:med)? teammate|program policy|scripted teammate/i,
    'The interface must identify the program teammate');
  return {viewport, board: bounds, horizontal_overflow: overflow};
}
async function stepKey(page, key, expectedDelta = 1) {
  const before = await state(page);
  await page.locator('#board').click();
  await page.keyboard.press(key);
  if (expectedDelta) await page.waitForFunction(turn => window.KitchenDemo.getState().turn === turn, before.turn + expectedDelta);
  await page.waitForTimeout(190);
  const after = await state(page);
  assert.equal(after.turn, before.turn + expectedDelta, `Key ${key} should advance ${expectedDelta} joint steps`);
  return {before, after};
}
async function stepButton(page, action) {
  const before = await state(page);
  await page.locator(`[data-action="${action}"]`).click();
  await page.waitForFunction(turn => window.KitchenDemo.getState().turn === turn, before.turn + 1);
  await page.waitForTimeout(190);
  return {before, after: await state(page)};
}
async function frame(page, index) {
  await page.locator('#timeline').evaluate((node, index) => {
    node.value = String(index);
    node.dispatchEvent(new Event('input', {bubbles: true}));
  }, index);
  await page.waitForFunction(index => window.KitchenDemo.getUI().replayIndex === index, index);
  await page.waitForTimeout(180);
}
async function closeExplanation(page) {
  if (await page.locator('#close-explanation').isVisible()) await page.locator('#close-explanation').click();
}
async function keyboardActivation(page) {
  const before = await state(page);
  await page.locator('[data-explain="why"]').focus();
  await page.keyboard.press('Space');
  await page.locator('#explanation').waitFor({state: 'visible'});
  assert.deepEqual(await state(page), before, 'Space on a focused explanation button must activate that button without making a WAIT move');
  await closeExplanation(page);
}
async function instructionRoles(page) {
  const preset = (await state(page)).preset;
  const tokens = await page.locator('.cooperation-diagram .chef-token').evaluateAll(nodes => nodes.map(node => ({text: node.textContent, human: node.classList.contains('human')})));
  assert.deepEqual(tokens, preset === 'supply' ? [{text: '1', human: true}, {text: '2', human: false}] : [{text: '2', human: false}, {text: '1', human: true}],
    'The instruction diagram must retain player/teammate identities after swapping jobs');
}
async function explain(page, kind, expectedFrame) {
  const before = await state(page), beforeHistory = await history(page), language = (await ui(page)).language;
  const expected = await page.evaluate(({kind, expectedFrame, language}) => {
    const source = window.KitchenDemo.getHistory()[expectedFrame];
    return window.KitchenEngine.explain(source, kind, language);
  }, {kind, expectedFrame, language});
  await page.locator(`[data-explain="${kind}"]`).click();
  await page.locator('#explanation').waitFor({state: 'visible'});
  const answer = await page.locator('#explanation-text').innerText();
  assert.ok(answer.includes(expected.text), 'Visible explanation must contain the actual engine explanation for the selected frame');
  assert.match(await page.locator('#explanation-frame').innerText(), new RegExp(`(?:^|\\D)${expectedFrame}(?:\\D|$)`));
  assert.equal(expected.frame, expectedFrame);
  await page.waitForTimeout(500);
  assert.deepEqual(await state(page), before, 'Explanation/forecast mutated the real round');
  assert.deepEqual(await history(page), beforeHistory, 'Explanation/forecast appended or changed history');
  assert.equal((await ui(page)).autoRunning, false, 'Opening an explanation pauses automatic play');
  const textBounds = await page.locator('#explanation-text').evaluate(node => ({client: node.clientWidth, scroll: node.scrollWidth,
    overflow: getComputedStyle(node).overflowY, clientHeight: node.clientHeight, scrollHeight: node.scrollHeight}));
  assert.ok(textBounds.scroll <= textBounds.client + 1, 'Explanation overflows horizontally');
  assert.ok(textBounds.clientHeight >= textBounds.scrollHeight - 1 || !['hidden', 'clip'].includes(textBounds.overflow), 'Explanation is clipped without scrolling');
  return {kind, frame: expectedFrame, language, answer, source_state_sha256: hash(before), forecast_steps: expected.forecast?.steps ?? null};
}
async function reset(page) {
  await closeExplanation(page);
  if ((await ui(page)).replayIndex !== null) await page.locator('#live').click();
  const preset = (await state(page)).preset;
  await page.locator('#restart').click();
  await page.waitForFunction(() => window.KitchenDemo.getState().turn === 0);
  await begin(page);
  const after = await state(page);
  assert.equal(after.preset, preset); assert.equal(after.orders, 0);
  assert.equal((await history(page)).length, 1); assert.equal((await ui(page)).autoRunning, false);
  return after;
}
async function swap(page) {
  await closeExplanation(page);
  const before = await state(page);
  await page.locator('#swap-role').click();
  await page.waitForFunction(preset => window.KitchenDemo.getState().preset !== preset, before.preset);
  await begin(page);
  const after = await state(page);
  assert.equal(after.turn, 0); assert.equal(after.orders, 0); assert.equal((await history(page)).length, 1);
  const human = after.actors.find(actor => actor.id === 'human'), ai = after.actors.find(actor => actor.id === 'ai');
  assert.equal(human.side, after.preset === 'supply' ? 'left' : 'right');
  assert.equal(ai.side, after.preset === 'supply' ? 'right' : 'left');
  return after;
}
async function autoplay(page) {
  await closeExplanation(page);
  const initial = await state(page);
  assert.equal(initial.turn, 0);
  await page.locator('#auto-demo').click();
  await page.waitForFunction(() => window.KitchenDemo.getState().done, null, {timeout: 65000});
  const final = await state(page);
  assert.equal(final.orders, 2, `Autoplay (${initial.preset}) must deliver both soups`);
  assert.ok(final.turn <= 120, 'Automatic demonstration exceeded the maximum steps');
  assert.equal((await ui(page)).autoRunning, false);
  assert.equal((await history(page)).length, final.turn + 1);
  assert.ok(await page.locator('#result').isVisible());
  const before = hash(final);
  await page.keyboard.press('Space'); await page.waitForTimeout(200);
  assert.equal(hash(await state(page)), before, 'A finished round cannot advance');
  return {preset: initial.preset, steps: final.turn, orders: final.orders, reason: final.reason, real_timer_ui_playback: true};
}

let browser;
const watchdog = setTimeout(() => {report.status = 'failed'; report.error = 'Browser acceptance exceeded six minutes'; save(); process.exit(1);}, 6 * 60 * 1000);
(async () => {
  try {
    browser = await chromium.launch({headless: true,
      ...((process.env.KITCHEN_CHROME || process.env.CANDY_CHROME) ? {executablePath: process.env.KITCHEN_CHROME || process.env.CANDY_CHROME} : {})});
    if (keyboardOnly) {
      const prior = JSON.parse(fs.readFileSync(path.join(out, 'browser_acceptance.json'), 'utf8'));
      assert.equal(prior.status, 'passed', 'Supplemental keyboard checks require a prior complete passing report');
      const {context, page} = await fresh(browser);
      await begin(page); await keyboardActivation(page); await stepKey(page, 'Space');
      await swap(page); await page.locator('#show-instructions').click(); await instructionRoles(page);
      await screenshot(page, 'instructions-swapped-1365x900-zh');
      await context.close(); assert.deepEqual(report.javascript_errors, []);
      const extras = [
        {name: 'space_activates_focused_explanation_button_without_advancing_time', passed: true, freshly_reloaded_final_app: true},
        {name: 'board_space_still_advances_exactly_one_wait', passed: true, freshly_reloaded_final_app: true},
        {name: 'instruction_diagram_identity_follows_swapped_roles', passed: true, freshly_reloaded_final_app: true},
      ];
      const names = new Set(extras.map(item => item.name));
      prior.cases = [...prior.cases.filter(item => !names.has(item.name)), ...extras];
      prior.screenshots = [...new Set([...prior.screenshots, ...report.screenshots])];
      prior.supplemental_keyboard_verified = new Date().toISOString();
      fs.writeFileSync(path.join(out, 'browser_acceptance.json'), JSON.stringify(prior, null, 2) + '\n');
      console.log(JSON.stringify({status: prior.status, cases: prior.cases.length, supplemental: extras.map(item => item.name)}));
      return;
    }
    for (const viewport of viewports) {
      const {context, page} = await fresh(browser, viewport);
      assert.ok(await page.locator('#instructions').isVisible());
      await screenshot(page, `instructions-${viewport.width}x${viewport.height}-zh`);
      await begin(page);
      for (const language of ['zh', 'en']) {
        await locale(page, language);
        record('default_layout', {language, ...await layout(page, viewport)});
        await screenshot(page, `default-${viewport.width}x${viewport.height}-${language}`);
      }
      const changed = await swap(page);
      for (const language of ['zh', 'en']) {
        await locale(page, language);
        record('swapped_layout', {preset: changed.preset, language, ...await layout(page, viewport)});
        await screenshot(page, `swapped-${viewport.width}x${viewport.height}-${language}`);
        const evidence = await explain(page, 'counterfactual', 0);
        record('bilingual_explanation_layout', {viewport, ...evidence});
        await screenshot(page, `explanation-${viewport.width}x${viewport.height}-${language}`);
        await closeExplanation(page);
      }
      await context.close();
    }

    const {context, page} = await fresh(browser);
    await begin(page);
    await keyboardActivation(page);
    record('space_activates_focused_explanation_button_without_advancing_time');
    const moved = await stepKey(page, 'ArrowUp');
    assert.deepEqual(moved.after.actors.find(a => a.id === 'human').position, [2, 1]);
    assert.equal(moved.after.actors.find(a => a.id === 'human').facing, 'UP');
    const picked = await stepKey(page, 'e');
    assert.ok(picked.after.actors.find(a => a.id === 'human').holding, 'Facing the ingredient station and pressing E picks up one onion');
    const bumped = await stepKey(page, 'ArrowLeft');
    assert.deepEqual(bumped.after.actors.find(a => a.id === 'human').position, bumped.before.actors.find(a => a.id === 'human').position);
    assert.equal(bumped.after.actors.find(a => a.id === 'human').facing, 'LEFT', 'Blocked movement still turns the chef toward facilities');
    await stepKey(page, 'Space');
    await stepKey(page, 'q', 0);
    await stepButton(page, 'INTERACT');
    record('keyboard_movement_interaction_wait_wall_and_unbound_key', {turn: (await state(page)).turn});

    for (const kind of ['why', 'waiting', 'counterfactual']) {
      record('live_explanation_is_grounded_and_nonmutating', await explain(page, kind, (await state(page)).turn));
      await closeExplanation(page);
    }
    await frame(page, 1);
    const liveBeforeReplay = await state(page);
    record('historical_frame_explanation', await explain(page, 'why', 1));
    await closeExplanation(page);
    record('historical_frame_forecast', await explain(page, 'counterfactual', 1));
    assert.equal(await page.locator('[data-action="WAIT"]').isDisabled(), true, 'Historical replay cannot receive game actions');
    await page.keyboard.press('Space'); await page.waitForTimeout(200);
    assert.deepEqual(await state(page), liveBeforeReplay);
    await closeExplanation(page); await page.locator('#live').click();
    await page.waitForFunction(() => window.KitchenDemo.getUI().replayIndex === null);

    const saveBefore = await state(page), historyBefore = await history(page);
    await page.reload(); await ready(page); await begin(page);
    assert.deepEqual(await state(page), saveBefore); assert.deepEqual(await history(page), historyBefore);
    assert.equal((await ui(page)).autoRunning, false);
    record('refresh_restores_confirmed_state_and_history', {turn: saveBefore.turn});

    await reset(page);
    await page.locator('#auto-demo').click();
    await page.waitForFunction(() => window.KitchenDemo.getState().turn >= 2);
    await page.locator('[data-explain="waiting"]').click();
    const paused = await state(page); await page.waitForTimeout(700);
    assert.deepEqual(await state(page), paused); assert.equal((await ui(page)).autoRunning, false);
    record('opening_explanation_pauses_automatic_demo', {turn: paused.turn});
    await closeExplanation(page);
    await page.locator('#auto-demo').click();
    await page.waitForFunction(turn => window.KitchenDemo.getState().turn > turn, paused.turn);
    await frame(page, 0);
    const replayPaused = await state(page); await page.waitForTimeout(700);
    assert.deepEqual(await state(page), replayPaused); assert.equal((await ui(page)).autoRunning, false);
    record('opening_replay_pauses_automatic_demo', {live_turn: replayPaused.turn, selected_frame: 0});
    await page.locator('#live').click();
    await page.locator('#auto-demo').click();
    await page.waitForFunction(turn => window.KitchenDemo.getState().turn > turn, replayPaused.turn);
    await page.reload(); await ready(page); await begin(page);
    assert.equal((await ui(page)).autoRunning, false);
    const recovered = await state(page); await page.waitForTimeout(700);
    assert.deepEqual(await state(page), recovered);
    record('refresh_during_automatic_demo_restores_paused', {turn: recovered.turn});

    await reset(page);
    record('automatic_demo_completes_default_role', await autoplay(page));
    await screenshot(page, 'completed-default-1365x900-zh');
    const swapped = await swap(page);
    record('role_swap_restarts_fresh_round', {preset: swapped.preset});
    record('automatic_demo_completes_swapped_role', await autoplay(page));
    await screenshot(page, 'completed-swapped-1365x900-zh');

    await reset(page);
    await page.locator('#show-instructions').click();
    assert.ok(await page.locator('#instructions').isVisible());
    await instructionRoles(page);
    record('instruction_diagram_identity_follows_swapped_roles');
    const instructionState = await state(page); await page.keyboard.press('Space'); await page.waitForTimeout(200);
    assert.deepEqual(await state(page), instructionState, 'Instruction reading cannot consume a step');
    record('instructions_pause_game_controls');
    await context.close();
    assert.deepEqual(report.javascript_errors, []);
    report.status = 'passed'; report.finished = new Date().toISOString(); save();
    console.log(JSON.stringify({status: report.status, cases: report.cases.length, report: path.join(out, 'browser_acceptance.json')}));
  } catch (error) {
    report.status = 'failed'; report.error = error.stack; save(); throw error;
  } finally {
    clearTimeout(watchdog);
    if (browser) await browser.close();
  }
})();
