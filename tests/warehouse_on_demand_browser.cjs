#!/usr/bin/env node
const { chromium } = require('playwright-core');
const assert = require('node:assert/strict');
const base = process.env.WAREHOUSE_SMOKE_URL || 'http://127.0.0.1:18968/warehouse/';

(async () => {
  const browser = await chromium.launch({ headless: true,
    executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' });
  try {
    for (const condition of ['explanation', 'control']) {
      const page = await browser.newPage();
      const errors = [];
      page.on('pageerror', (error) => errors.push(error.message));
      await page.goto(base);
      assert.equal(await page.getAttribute('html', 'lang'), 'zh-CN');
      assert.equal(await page.textContent('#languageButtonLabel'), 'English');
      assert.equal(await page.locator('#sceneTitle').textContent(), '协作配送现场');
      await page.fill('#participantInput', `browser-${condition}`);
      await page.check('#rulesAgreement');
      await page.selectOption('#testConditionSelector', condition);
      await page.click('#startButton');
      await page.waitForFunction(() => document.body.dataset.studyStage === 'instructions');
      await page.click('#beginTask1Button');
      await page.waitForFunction(() => document.body.dataset.studyStage === 'task1');
      await page.locator('#actionPad button[data-action="WAIT"]').click();
      await page.waitForFunction(() => document.querySelector('#stepValue').textContent.startsWith('1 /'));
      const before = await page.evaluate(() => ({
        frame: state.view.study.progress,
        score: state.view.state.user_score,
        battery: state.view.state.agents.map(agent => agent.battery),
        run: state.view.study.run_id,
      }));
      if (condition === 'explanation') {
        await page.waitForFunction(() => !document.querySelector('#aiActionBubble').classList.contains('hidden'));
        assert.equal(await page.locator('#aiActionBubble .bubble-toggle').textContent(), '为什么？');
        assert.equal(await page.locator('#aiActionBubble p').count(), 0);
        await page.locator('#aiActionBubble .bubble-toggle').click();
        await page.waitForFunction(() => document.querySelector('#aiActionBubble p'));
        assert.ok((await page.locator('#aiActionBubble p').textContent()).length > 4);
        assert.match(await page.locator('#answerFrame').textContent(), /1/);
        assert.deepEqual(await page.evaluate(() => ({
          frame: state.view.study.progress, score: state.view.state.user_score,
          battery: state.view.state.agents.map(agent => agent.battery), run: state.view.study.run_id,
        })), before);
        await page.click('#languageButton');
        await page.waitForFunction(() => document.querySelector('#languageButtonLabel').textContent === '中文');
        assert.equal(await page.locator('#aiActionBubble .bubble-toggle').textContent(), 'Hide');
        assert.ok((await page.locator('#aiActionBubble p').textContent()).startsWith('Robot 2'));
        await page.click('#languageButton');
        await page.waitForFunction(() => document.querySelector('#languageButtonLabel').textContent === 'English');
        assert.equal(await page.locator('#aiActionBubble .bubble-toggle').textContent(), '收起');
        await page.locator('#aiActionBubble .bubble-toggle').click();
        assert.equal(await page.locator('#aiActionBubble p').count(), 0);
        await page.locator('#aiActionBubble .bubble-toggle').click();
        assert.equal(await page.locator('#aiActionBubble p').count(), 1);
        await page.locator('#actionPad button[data-action="WAIT"]').click();
        await page.waitForFunction(() => document.querySelector('#stepValue').textContent.startsWith('2 /'));
        assert.equal(await page.locator('#aiActionBubble .bubble-toggle').textContent(), '为什么？');
        assert.equal(await page.locator('#aiActionBubble p').count(), 0);
        await page.evaluate(() => {
          document.querySelector('#surveyQuestions input[name="coordination_understanding"][value="4"]').checked = true;
          document.querySelector('#surveyComment').value = '保留我的输入';
          document.querySelector('#questionInput').value = '自定义问题';
        });
        await page.click('#languageButton');
        await page.waitForFunction(() => document.querySelector('#languageButtonLabel').textContent === '中文');
        assert.equal(await page.locator('#surveyQuestions input[name="coordination_understanding"][value="4"]').isChecked(), true);
        assert.equal(await page.locator('#surveyComment').inputValue(), '保留我的输入');
        assert.equal(await page.locator('#questionInput').inputValue(), '自定义问题');
      } else {
        assert.equal(await page.locator('#aiActionBubble').isVisible(), false);
        const status = await page.evaluate(async () => {
          const response = await fetch('/warehouse/api/study/command', {
            method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Warehouse-Page': PAGE_ID },
            body: JSON.stringify({ operation_id: 'forbidden', run_id: state.view.study.run_id,
              command: 'ask_explanation', payload: { question: 'Why?', question_kind: 'action',
                target_agent: 'robot_2', action_run_id: state.view.study.run_id,
                action_stage: 'task1', action_frame: 1 } }),
          });
          return response.status;
        });
        assert.equal(status, 400);
      }
      assert.deepEqual(errors, []);
      await page.close();
    }
    console.log('Warehouse Chinese default, frame-bound on-demand explanation, language state and B permissions: passed');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
