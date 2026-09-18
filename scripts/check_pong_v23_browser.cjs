#!/usr/bin/env node
// Read-only Python/browser controller parity fixture; input and output are JSON.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const web = path.join(__dirname, '..', 'domains', 'pong', 'web');
const payload = JSON.parse(fs.readFileSync(0, 'utf8'));
const protocol = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'configs', 'three_task_study.json'), 'utf8'));
const context = vm.createContext({ console, cases: payload.cases || [], crypto: require('node:crypto'),
  location: { protocol: 'http:' } });
for (const name of ['coordinated.js', 'explain.js']) {
  vm.runInContext(fs.readFileSync(path.join(web, name), 'utf8'), context, { filename: name });
}
const source = fs.readFileSync(path.join(web, 'app.js'), 'utf8').split("$('start').onclick")[0];
vm.runInContext(source, context, { filename: 'app.js' });
vm.runInContext(`studyProtocol = ${JSON.stringify(protocol)}`, context);
const result = vm.runInContext(`(() => {
  nnController = { controller_mode: 'coordinated', rule_version: 'pong-coordinated.v2.3' };
  return cases.map(item => {
    const game = new OfflinePong({ group: 'B', task: 1, seed: item.seed || 260920 });
    game.frame = item.frame_index;
    game.playerX = item.player_x;
    game.aiX = item.ai_x;
    game.commitment = null;
    game.coordinator.smallCommitment = null;
    game.balls = item.balls.map(ball => ({ ball_id: ball.ball_id, kind: ball.kind,
      x: ball.x, y: ball.y, vx: ball.vx, vy: ball.vy, width_cells: ball.width_cells,
      height_cells: ball.height_cells, active: ball.active,
      descendingEncounter: ball.descending_encounter, encounterIndex: ball.encounter_index,
      pendingMiss: ball.pending_miss, pendingMissId: ball.pending_miss_id }));
    const decision = game.coordinator.choose(game, item.proposal);
    return { action: decision.action, reason: decision.reason, evidence: decision.evidence,
      bubble: PongExplanations.bubble({ action: decision.action, planEvidence: decision.evidence }) };
  });
})()`, context);
process.stdout.write(JSON.stringify(result));
