#!/usr/bin/env node
const { chromium } = require('playwright-core');
const assert = require('node:assert/strict');

const base = process.env.STUDY_SMOKE_URL || 'http://127.0.0.1:18765';

(async () => {
  const browser = await chromium.launch({ headless: true,
    executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' });
  try {
    const warehouse = await browser.newPage({ viewport: { width: 1440, height: 900 } });
    await warehouse.goto(`${base}/warehouse/`);
    await warehouse.fill('#participantInput', 'presentation-demo');
    await warehouse.check('#rulesAgreement');
    await warehouse.click('#startButton');
    await warehouse.waitForFunction(() => state.demoTimeline?.frames?.length === 121);
    assert.equal(await warehouse.evaluate(() => state.demoTimeline.trajectory_seed), 40786);
    await warehouse.click('#demoPlayButton');
    await warehouse.waitForTimeout(850);
    const step = await warehouse.evaluate(() => state.demoIndex);
    assert.ok(step >= 1 && step <= 2, `850ms of playback reached step ${step}`);
    await warehouse.click('#demoPlayButton');
    const paused = await warehouse.evaluate(() => [state.demoIndex, state.demoElapsedMs]);
    await warehouse.waitForTimeout(400);
    assert.deepEqual(await warehouse.evaluate(() => [state.demoIndex, state.demoElapsedMs]), paused);
    await warehouse.click('#languageButton');
    assert.deepEqual(await warehouse.evaluate(() => [state.demoIndex, state.demoElapsedMs]), paused);
    await warehouse.reload();
    await warehouse.waitForFunction(() => state.demoTimeline?.frames?.length === 121);
    assert.equal(await warehouse.evaluate(() => state.demoIndex), paused[0]);
    assert.equal(await warehouse.evaluate(() => state.demoElapsedMs), 0);
    const intermediate = await warehouse.evaluate(() => {
      const index = state.demoTimeline.frames.findIndex((frame, i) => {
        const motion = state.demoTimeline.frames[i + 1]?.transition?.agents?.find(item => item.id === 'robot_1');
        return motion && String(motion.from_position) !== String(motion.to_position);
      });
      if (index < 0) throw new Error('No moving tutorial transition');
      state.demoIndex = index;
      state.demoElapsedMs = 300;
      paintDemoFrame(); paintDemoMotion();
      const motion = state.demoTimeline.frames[index + 1].transition.agents.find(item => item.id === 'robot_1');
      const canvas = document.querySelector('#warehouseCanvas');
      return { progress: Number(canvas.dataset.animationProgress),
        row: Number(canvas.dataset.robot1Row), col: Number(canvas.dataset.robot1Col),
        from: motion.from_position, to: motion.to_position };
    });
    assert.ok(intermediate.progress > 0 && intermediate.progress < 1);
    assert.ok((intermediate.row !== intermediate.from[0] && intermediate.row !== intermediate.to[0])
      || (intermediate.col !== intermediate.from[1] && intermediate.col !== intermediate.to[1]));
    const finished = await warehouse.evaluate(() => {
      const schedule = window.requestAnimationFrame;
      stopDemo();
      state.demoIndex = 0;
      state.demoElapsedMs = 0;
      state.demoPlaying = true;
      state.demoLastTimestamp = 0;
      const generation = state.demoGeneration;
      window.requestAnimationFrame = () => 0;
      try {
        for (let elapsed = 100; elapsed <= 72000 && state.demoPlaying; elapsed += 100) {
          demoTick(elapsed, generation);
        }
      } finally { window.requestAnimationFrame = schedule; }
      return { step: state.demoIndex, playing: state.demoPlaying,
        score: Number(document.querySelector('#scoreValue').textContent),
        finalScore: state.demoTimeline.frames[120].state.user_score };
    });
    assert.equal(finished.step, 120);
    assert.equal(finished.playing, false);
    assert.equal(finished.score, Math.round(finished.finalScore));
    await warehouse.click('#demoPlayButton');
    assert.equal(await warehouse.evaluate(() => state.demoIndex), 0);
    await warehouse.click('#demoPlayButton');
    await warehouse.click('#beginTask1Button');
    await warehouse.waitForFunction(() => state.view?.study?.stage === 'task1');
    assert.equal(await warehouse.evaluate(() => state.demoTimeline), null);
    const metrics = await warehouse.evaluate(async () => (await fetch('/warehouse/api/fixture-metrics')).json());
    assert.ok(!metrics.command_requests.includes('tutorial_advance'));

    const pong = await browser.newPage();
    await pong.goto(`${base}/pong/`);
    await pong.selectOption('#group', 'B');
    await pong.click('#start');
    await pong.waitForFunction(() => game?.task === 1);
    for (const taskId of [1, 2, 3]) {
      await pong.evaluate(() => {
        game.frame = Math.round(SPEC.durationSeconds / SPEC.fixedDt) - 1;
        simulationTick();
      });
      assert.equal(await pong.locator('#roundResult').isVisible(), true);
      if (taskId < 3) await pong.click('#nextTask');
    }
    await pong.evaluate(() => {
      // Task 1 uses the archived weighted-by-type format; Tasks 2–3 use raw event counts.
      Object.assign(completedRuns[0], { score_format: undefined,
        missed_by_type: { small: 2, large: 9 }, missed_balls: 11 });
      Object.assign(completedRuns[1], { score_format: 'pong-misses.v2',
        small_misses: 2, large_misses: 3, weighted_misses: 11 });
      Object.assign(completedRuns[2], { score_format: 'pong-misses.v2',
        small_misses: 2, large_misses: 3, weighted_misses: 11 });
      showQuestionnaire();
    });
    assert.deepEqual(await pong.locator('#scoreSummary tbody tr').allTextContents(),
      ['Task 12311', 'Task 22311', 'Task 32311']);
    const fields = await pong.locator('#surveyItems fieldset').count();
    for (let i = 0; i < fields; i += 1) await pong.locator(`#surveyItems input[name="q${i}"][value="4"]`).check();
    await pong.click('#surveyForm button[type="submit"]');
    assert.equal(await pong.locator('#completedScores tbody tr').count(), 3);
    await pong.reload();
    await pong.waitForFunction(() => !document.querySelector('#completed').hidden);
    assert.deepEqual(await pong.locator('#completedScores tbody tr').allTextContents(),
      ['Task 12311', 'Task 22311', 'Task 32311']);
    assert.equal(await pong.locator('#review').isVisible(), false);
    await pong.click('#newParticipant');
    assert.equal(await pong.locator('#setup').isVisible(), true);
    assert.equal(await pong.evaluate(() => sessionStorage.getItem(SESSION_KEY)), null);
    console.log('Warehouse demo and Pong score presentation passed');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
