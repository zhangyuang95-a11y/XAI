#!/usr/bin/env node
// Execute the real browser controller in a DOM-free V8 context for parity tests.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const appPath = path.join(__dirname, '..', 'domains', 'pong', 'web', 'app.js');
const source = fs.readFileSync(appPath, 'utf8').split("$('start').onclick")[0];
const payload = JSON.parse(fs.readFileSync(0, 'utf8'));
const context = vm.createContext({ console, cases: payload.cases || [], model: payload.model || null,
  program: payload.program || null,
  controller: payload.controller || { controller_mode: 'hybrid', rule_version: 'pong-limited-assist.v2.2' } });
vm.runInContext(fs.readFileSync(path.join(__dirname, '..', 'domains', 'pong', 'web', 'coordinated.js'), 'utf8'), context);
vm.runInContext(source, context, { filename: appPath });
const result = vm.runInContext(`(() => {
  nnController = controller;
  if (model) nnModel = model;
  if (program) nnProgram = program;
  return cases.map(item => {
    const game = new OfflinePong({ group: 'B', task: 1, seed: 260920 });
    game.frame = item.frame_index;
    game.playerX = item.player_x;
    game.aiX = item.ai_x;
    game.missedBalls = item.missed_balls || 0;
    game.totalOpportunities = item.total_opportunities || 0;
    game.successfulOpportunities = item.successful_opportunities || 0;
    game.missedByType = item.missed_by_type || { small: 0, large: 0 };
    game.balls = item.balls.map(ball => ({
      ball_id: ball.ball_id, kind: ball.kind, x: ball.x, y: ball.y, vx: ball.vx, vy: ball.vy,
      width_cells: ball.width_cells, height_cells: ball.height_cells,
      active: ball.active, descendingEncounter: ball.descending_encounter,
      encounterIndex: ball.encounter_index, pendingMiss: ball.pending_miss,
    }));
    const decision = game.chooseHybrid(item.proposal);
    const probabilities = item.features && model ? nnForward(item.features) : null;
    const programTrace = item.features && program ? executeProgram(item.features)?.trace : null;
    let physics = null;
    if (item.simulate) {
      game.latestDecision = { action: item.simulate.ai_action, controllerSource: 'hybrid' };
      game.lastDecisionFrame = game.frame;
      game.step(item.simulate.player_action);
      physics = { frame: game.frame, player_x: game.playerX, ai_x: game.aiX,
        balls: game.balls.map(ball => ({ ball_id: ball.ball_id, x: ball.x, y: ball.y,
          vx: ball.vx, vy: ball.vy, descending_encounter: ball.descendingEncounter,
          encounter_index: ball.encounterIndex, pending_miss: ball.pendingMiss })),
        events: game.history.at(-1).events, missed_balls: game.missedBalls };
    }
    return { action: decision.action, reason: decision.reason, evidence: decision.evidence,
      probabilities, programTrace, commitment: game.hybridCommitment, physics };
  });
})()`, context);
process.stdout.write(JSON.stringify(result));
