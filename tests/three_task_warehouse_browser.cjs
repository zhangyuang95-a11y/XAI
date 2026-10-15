#!/usr/bin/env node
const { chromium } = require('playwright-core');
const assert = require('node:assert/strict');

const url = process.env.WAREHOUSE_SMOKE_URL || 'http://127.0.0.1:18765/warehouse/';

(async () => {
  const browser = await chromium.launch({ headless: true,
    executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' });
  try {
    for (const group of ['A', 'B']) {
      const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      await page.goto(url);
      await page.fill('#participantInput', `three-task-${group}`);
      await page.selectOption('#testConditionSelector', group === 'A' ? 'explanation' : 'control');
      await page.check('#rulesAgreement');
      await page.click('#startButton');
      await page.waitForFunction(() => state.view?.study?.stage === 'instructions');
      await page.click('#beginTask1Button');
      await page.waitForFunction(() => state.view?.study?.stage === 'task1');
      assert.equal(await page.locator('#aiActionBubble').isVisible(), false);
      assert.equal(await page.locator('#liveExplanationPanel').isVisible(), false);

      const finish = stage => page.evaluate(async activeStage => {
        let view = state.view;
        for (let step = 0; step < 120 && view.study.stage === activeStage; step += 1) {
          const payload = {
            operation_id: crypto.randomUUID(), run_id: view.study.run_id,
            expected_stage: view.study.stage, expected_state_version: view.study.state_version,
            command: 'human_action', payload: { action: 'WAIT' },
          };
          const response = await fetch('/warehouse/api/study/command', {
            method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Warehouse-Page': PAGE_ID },
            body: JSON.stringify(payload),
          });
          if (!response.ok) throw new Error(`Warehouse step failed: ${response.status}`);
          view = (await response.json()).view;
        }
        await render(view, { skipAnimation: true });
        return view.study.stage;
      }, stage);

      assert.equal(await finish('task1'), 'task1_complete');
      await page.click('#beginTask2Button');
      await page.waitForFunction(() => state.view?.study?.stage === 'task2');
      assert.equal(await page.evaluate(() => state.view.study.task_seed), 51000);
      await page.locator('#actionPad button[data-action="WAIT"]').click();
      await page.waitForFunction(() => Number(state.view?.study?.progress) === 1);
      assert.equal(await page.locator('#aiActionBubble').isVisible(), group === 'A');
      assert.equal(await page.locator('#liveExplanationPanel').isVisible(), group === 'A');
      if (group === 'A') {
        assert.equal(await page.locator('#answerPanel').isVisible(), false);
        await page.locator('#aiActionBubble button').click();
        await page.waitForFunction(() => Number(state.view?.last_explanation?.anchor_frame) === 1);
        assert.equal(await page.locator('#answerPanel').isVisible(), true);
      }
      assert.equal(await finish('task2'), 'task2_complete');
      await page.click('#beginTask3Button');
      await page.waitForFunction(() => state.view?.study?.stage === 'task3');
      await page.reload();
      await page.waitForFunction(() => state.view?.study?.stage === 'task3');
      assert.equal(await page.evaluate(() => state.view.study.task_seed), 51500);
      assert.equal(await page.locator('#aiActionBubble').isVisible(), false);
      assert.equal(await page.locator('#answerPanel').isVisible(), false);
      assert.equal(await page.locator('#liveExplanationPanel').isVisible(), false);
      assert.equal(await finish('task3'), 'survey');
      assert.equal(await page.locator('#surveyPanel').isVisible(), true);
      for (const name of ['coordination_understanding', 'ai_predictability', 'interface_clarity']) {
        await page.locator(`#surveyQuestions input[name="${name}"][value="4"]`).check();
      }
      if (group === 'A') {
        for (const name of ['explanation_clarity', 'explanation_usefulness', 'question_helpfulness']) {
          assert.match(await page.locator(`#surveyQuestions input[name="${name}"]`).first().locator('xpath=../../..').textContent(), /Task 2/);
          await page.locator(`#surveyQuestions input[name="${name}"][value="na"]`).check();
        }
      } else {
        assert.equal(await page.locator('#surveyQuestions input[name="explanation_clarity"]').count(), 0);
      }
      await page.click('#submitSurveyButton');
      await page.waitForFunction(() => state.view?.study?.stage === 'completed');
      assert.equal(await page.locator('#completePanel').isVisible(), true);
      assert.equal(await page.evaluate(() => state.view.study.record_saved), true);
      assert.equal(await page.evaluate(() => Object.keys(state.view.study.round_summaries).length), 3);
      assert.deepEqual(errors, []);
      await page.close();
    }
    console.log('Warehouse three-task browser flow passed for A and B');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
