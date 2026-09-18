#!/usr/bin/env node
const { chromium } = require('playwright-core');
const assert = require('node:assert/strict');

const url = process.env.PONG_SMOKE_URL || 'http://127.0.0.1:18765/pong/';

(async () => {
  const browser = await chromium.launch({ headless: true,
    executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' });
  try {
    for (const group of ['A', 'B']) {
      const page = await browser.newPage();
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      await page.goto(url);
      await page.selectOption('#group', group);
      await page.fill('#participant', `three-task-${group}`);
      await page.click('#start');
      await page.waitForFunction(() => !document.querySelector('#game').hidden);
      assert.equal(await page.evaluate(() => game.task), 1);
      assert.equal(await page.evaluate(() => game.seed), 260920);
      assert.equal(await page.evaluate(() => game.history[0].balls.find(ball => ball.ball_id === 'A1').x), 2.65);
      assert.equal(await page.locator('#intentBubble').isVisible(), false);
      await page.keyboard.press('s');
      assert.equal(await page.locator('#review').isVisible(), false);
      await page.keyboard.press('s');

      const finish = () => page.evaluate(() => {
        running = true; game.paused = false; game.phase = 'active';
        game.frame = Math.round(SPEC.durationSeconds / SPEC.fixedDt) - 1;
        simulationTick();
      });
      await finish();
      assert.equal(await page.evaluate(() => game.terminal), true);
      assert.equal(await page.locator('#nextTask').isVisible(), true);
      await page.click('#nextTask');
      assert.deepEqual(await page.evaluate(() => [game.task, game.seed, completedRuns.length]), [2, 260918, 1]);
      await page.reload();
      await page.waitForFunction(() => document.querySelector('#start').textContent.includes('Task 2'));
      assert.equal(await page.locator('#group').inputValue(), group);
      assert.equal(await page.locator('#group').isDisabled(), true);
      await page.click('#start');
      await page.waitForFunction(() => !document.querySelector('#game').hidden);
      assert.deepEqual(await page.evaluate(() => [game.task, game.seed, completedRuns.length]), [2, 260918, 1]);
      assert.deepEqual(await page.evaluate(() => {
        const ball = game.history[0].balls.find(item => item.ball_id === 'A1');
        return [ball.x, ball.vx];
      }), [3, 2]);
      assert.equal(await page.locator('#intentBubble').isVisible(), group === 'A');
      await page.evaluate(() => { for (let index = 0; index < 24; index += 1) simulationTick(); });
      await page.keyboard.press('s');
      assert.equal(await page.locator('#review').isVisible(), group === 'A');
      if (group === 'A') {
        await page.fill('#question', '此时机器人2准备接哪个球？');
        await page.click('#ask');
        assert.equal(await page.evaluate(() => game.questionHistory.length), 1);
        await page.evaluate(() => {
          const start = document.querySelector('#rangeStart');
          const end = document.querySelector('#timeline');
          start.value = '0'; start.dispatchEvent(new Event('input', { bubbles: true }));
          end.value = String(game.history.length - 1);
          end.dispatchEvent(new Event('input', { bubbles: true }));
        });
        await page.fill('#question', '这段时间机器人2在接哪个球？');
        await page.click('#ask');
        assert.equal(await page.evaluate(() => game.questionHistory.length), 2);
        assert.equal(await page.evaluate(() => game.questionHistory[1].start_frame < game.questionHistory[1].end_frame), true);
      }
      await page.keyboard.press('s');
      await finish();
      assert.equal(await page.locator('#review').isVisible(), group === 'A');
      await page.click('#nextTask');
      assert.deepEqual(await page.evaluate(() => [game.task, game.seed, completedRuns.length]), [3, 260919, 2]);
      assert.deepEqual(await page.evaluate(() => {
        const ball = game.history[0].balls.find(item => item.ball_id === 'A1');
        return [ball.x, ball.vx];
      }), [3, -2]);
      assert.equal(await page.locator('#intentBubble').isVisible(), false);
      assert.equal(await page.locator('#review').isVisible(), false);
      assert.equal(await page.textContent('#answer'), '');
      await page.keyboard.press('s');
      assert.equal(await page.locator('#review').isVisible(), false);
      await page.keyboard.press('s');
      await finish();
      assert.equal(await page.locator('#questionnaire').isVisible(), true);
      assert.equal(await page.evaluate(() => completedRuns.length), 3);
      const saved = await page.evaluate(() => completedRuns.map(run =>
        [run.task_id, run.seed, run.explanation_allowed, run.protocol_version]));
      assert.deepEqual(saved.map(row => row[0]), [1, 2, 3]);
      assert.deepEqual(saved.map(row => row[1]), [260920, 260918, 260919]);
      assert.deepEqual(saved.map(row => row[2]), [false, group === 'A', false]);
      assert.ok(saved.every(row => row[3] === 'three-task-explanation.v1'));
      assert.deepEqual(await page.evaluate(() => completedRuns.map(run => run.assignment_source)),
        ['manual_self_select', 'manual_self_select', 'manual_self_select']);
      assert.deepEqual(await page.evaluate(() => completedRuns.map(run => run.bubble_display_count > 0)),
        [false, group === 'A', false]);
      const count = await page.locator('#surveyItems fieldset').count();
      for (let index = 0; index < count; index += 1) {
        await page.locator(`#surveyItems input[name="q${index}"][value="4"]`).check();
      }
      await page.click('#surveyForm button[type="submit"]');
      assert.equal(await page.locator('#completed').isVisible(), true);
      const stored = await page.evaluate(() => {
        const key = Object.keys(localStorage).find(item => item.startsWith('pong.questionnaire.three-task-'));
        return JSON.parse(localStorage.getItem(key));
      });
      assert.equal(stored.version, 'three-task-explanation.v1');
      assert.equal(stored.runs.length, 3);
      assert.deepEqual(errors, []);
      await page.close();
    }
    console.log('Pong three-task browser flow passed for A and B');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
