#!/usr/bin/env node
// Short local browser flow. It advances the last frame directly; no full game is played.
const { chromium } = require('playwright-core');
const assert = require('node:assert/strict');
const url = process.env.PONG_SMOKE_URL || 'http://127.0.0.1:18768/pong/';

(async () => {
  const browser = await chromium.launch({ headless: true,
    executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' });
  try {
    const decisions = [];
    for (const group of ['A', 'B']) {
      const page = await browser.newPage({ acceptDownloads: true });
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      await page.goto(url);
      await page.selectOption('#group', group);
      await page.click('#start');
      await page.waitForFunction(() => document.querySelector('#game').hidden === false);
      const initial = await page.evaluate(() => {
        game.paused = true; render();
        return { source: game.controllerSource, decision: game.latestDecision,
          bubbleHidden: document.querySelector('#intentBubble').hidden,
          reviewHidden: document.querySelector('#review').hidden,
          frame: game.frame };
      });
      assert.equal(initial.source, 'coordinated');
      assert.equal(initial.decision.actorHash, '90f86aa11ef66cba65e87d24f3eab05bb2f3ae05cd4e3e4b0258574573a07667');
      decisions.push({ action: initial.decision.action, proposal: initial.decision.nnProposedAction,
        target: initial.decision.planEvidence.target_ball_id });
      assert.equal(initial.bubbleHidden, group !== 'A');
      assert.equal(initial.reviewHidden, group !== 'A');
      if (group === 'B') assert.doesNotMatch(await page.textContent('#status'), /NN|规则协调/);
      if (group === 'B') {
        const beforeTime = await page.evaluate(() => { game.paused = false; return game.timeSeconds; });
        await page.waitForTimeout(1100);
        const advanced = await page.evaluate(() => { game.paused = true; return game.timeSeconds; });
        assert.ok(advanced - beforeTime >= .8 && advanced - beforeTime <= 1.5,
          `fixed-step browser clock advanced ${advanced - beforeTime}s in 1.1s`);
      }
      if (group === 'A') {
        await page.evaluate(() => {
          running = false;
          game.paused = false;
          for (let frame = 0; frame < 20; frame += 1) game.step('stay');
          game.paused = true;
          render();
        });
        await page.locator('#timeline').fill('10');
        assert.equal(await page.textContent('#replayFrame'), '10');
        assert.equal(await page.evaluate(() => replayIndex), 10);
        assert.equal(await page.evaluate(() => document.querySelector('#rangeStart').value), '10');
        await page.fill('#question', '为什么接这颗球？');
        await page.click('#ask');
        const why = await page.textContent('#answer');
        assert.doesNotMatch(why, /我负责|我去接|我实际/);
        await page.fill('#question', '这个动作是NN建议还是规则改选？');
        await page.click('#ask');
        const source = await page.textContent('#answer');
        assert.notEqual(why, source);
        assert.match(source, /NN建议/);
        assert.match(await page.textContent('#technicalEvidenceText'), /没有匹配的抽取程序/);
        const before = await page.evaluate(() => ({ frame: game.frame, x: game.aiX,
          ball: game.balls[0].x, history: game.history.length }));
        await page.fill('#question', '如果我当时不移动，会发生什么？');
        await page.click('#ask');
        assert.match(await page.textContent('#answer'), /假设|快照/);
        const after = await page.evaluate(() => ({ frame: game.frame, x: game.aiX,
          ball: game.balls[0].x, history: game.history.length }));
        assert.deepEqual(after, before);
        assert.equal(await page.evaluate(() => game.questionHistory.length), 3);
        const [download] = await Promise.all([
          page.waitForEvent('download'), page.click('#downloadReview'),
        ]);
        assert.match(download.suggestedFilename(), /pong-task1-review/);
      }
      await page.evaluate(() => {
        game.frame = Math.round(SPEC.durationSeconds / SPEC.fixedDt) - 1;
        game.paused = false; running = true; replayLocked = false; simulationTick();
      });
      if (group === 'A') await page.click('#task2');
      await page.waitForFunction(() => game.task === 2);
      assert.equal(await page.locator('#intentBubble').isVisible(), false);
      assert.equal(await page.locator('#review').isVisible(), false);
      assert.equal(await page.textContent('#technicalEvidenceText'), '');
      await page.evaluate(() => { game.paused = true; $('ask').click(); });
      assert.equal(await page.textContent('#answer'), '当前阶段不提供回放问答。');
      await page.evaluate(() => {
        game.frame = Math.round(SPEC.durationSeconds / SPEC.fixedDt) - 1;
        game.paused = false; simulationTick();
      });
      assert.equal(await page.locator('#questionnaire').isVisible(), true);
      assert.ok(await page.locator('#surveyItems fieldset').count() >= 5);
      assert.deepEqual(errors, []);
      await page.close();
    }
    assert.deepEqual(decisions[0], decisions[1]);
    const missing = await browser.newPage();
    await missing.route('**/nn_model.json', route => route.fulfill({ status: 404, body: 'missing' }));
    await missing.goto(url);
    await missing.click('#start');
    await missing.waitForFunction(() => document.querySelector('#setupError').textContent.length > 0);
    assert.match(await missing.textContent('#setupError'), /冻结模型|模型文件/);
    assert.equal(await missing.locator('#game').isVisible(), false);
    await missing.close();
    const mismatched = await browser.newPage();
    await mismatched.route('**/controller_config.json', async route => {
      const response = await route.fetch();
      const config = await response.json();
      await route.fulfill({ response, json: { ...config, model_sha256: 'wrong-model' } });
    });
    await mismatched.goto(url);
    await mismatched.click('#start');
    await mismatched.waitForFunction(() => document.querySelector('#setupError').textContent.length > 0);
    assert.match(await mismatched.textContent('#setupError'), /哈希.*不一致/);
    assert.equal(await mismatched.locator('#game').isVisible(), false);
    await mismatched.close();
    console.log('v2.3 browser A/B parity, two questions, counterfactual, log, Task 2 and survey: passed');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
