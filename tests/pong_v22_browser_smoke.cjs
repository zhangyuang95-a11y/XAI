#!/usr/bin/env node
// Fast UI flow check against a locally served candidate bundle. No 90s game is played.
const { chromium } = require('playwright');
const assert = require('node:assert/strict');
const url = process.env.PONG_SMOKE_URL || 'http://127.0.0.1:18767/pong/';

(async () => {
  const browser = await chromium.launch({ headless: true,
    executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' });
  try {
    const initialDecisions = [];
    for (const group of ['A', 'B']) {
      const page = await browser.newPage();
      await page.goto(url);
      await page.selectOption('#group', group);
      await page.click('#start');
      await page.waitForFunction(() => document.querySelector('#game').hidden === false);
      const started = await page.evaluate(() => {
        game.paused = true;
        render();
        return { source: game.controllerSource, decision: game.latestDecision,
          bubble: document.querySelector('#intentBubble').hidden,
          review: document.querySelector('#review').hidden };
      });
      assert.equal(started.source, 'hybrid');
      assert.equal(started.decision.nnProposedAction !== undefined, true);
      initialDecisions.push({ action: started.decision.action,
        proposal: started.decision.nnProposedAction,
        probabilities: started.decision.probabilities });
      assert.equal(started.bubble, group !== 'A');
      assert.equal(started.review, group !== 'A');
      if (group === 'A') {
        await page.fill('#question', '为什么这样移动？');
        await page.click('#ask');
        assert.match(await page.textContent('#answer'), /冻结NN|实际提交/);
      }
      await page.evaluate(() => {
        game.frame = Math.round(SPEC.durationSeconds / SPEC.fixedDt) - 1;
        game.paused = false;
        simulationTick();
      });
      if (group === 'A') {
        assert.equal(await page.locator('#task2').isVisible(), true);
        await page.click('#task2');
      }
      await page.waitForFunction(() => game.task === 2);
      assert.equal(await page.locator('#intentBubble').isVisible(), false);
      assert.equal(await page.locator('#review').isVisible(), false);
      await page.evaluate(() => { game.paused = true; $('ask').click(); });
      assert.equal(await page.textContent('#answer'), '当前阶段不提供回放问答。');
      await page.evaluate(() => {
        game.frame = Math.round(SPEC.durationSeconds / SPEC.fixedDt) - 1;
        game.paused = false;
        simulationTick();
      });
      assert.equal(await page.locator('#questionnaire').isVisible(), true);
      assert.ok(await page.locator('#surveyItems fieldset').count() >= 5);
      await page.close();
    }
    assert.deepEqual(initialDecisions[0], initialDecisions[1]);
    console.log('A/B bubble, question, Task 2 isolation, transition, survey: passed');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
