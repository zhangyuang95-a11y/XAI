const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const file = path.join(__dirname, '..', 'domains', 'pong', 'web', 'explain.js');
const context = vm.createContext({});
vm.runInContext(fs.readFileSync(file, 'utf8'), context);
const explain = context.PongExplanations;
const evidence = { target_ball_id: 'B1', target_kind: 'large', contact_side: 'left',
  self_distance: 2, partner_distance: 3, partner_status: 'reachable', requires_partner: true,
  holding: false, reason: 'prepare_large_side', planned_sequence: ['B1', 'A2'],
  alternatives: [
    { ball_id: 'B1', viable: true, weight: 3 },
    { ball_id: 'A1', viable: false, weight: 1 },
    { ball_id: 'A2', viable: true, weight: 1 },
  ] };
const decision = { action: 'left', controllerSource: 'coordinated', planEvidence: evidence,
  nnProposedAction: 'right', intervened: true };
const frame = { frame: 12, decision, events: [{ event: 'encounter', encounter_id: 'A2:1', ball_id: 'A2',
  kind: 'small', outcome: 'missed', player_coverage: [false], ai_coverage: [false] }] };
const history = [frame];
const why = explain.answer('为什么接B1？', frame, history, 0, null, null);
const avoid = explain.answer('为什么不接A1？', frame, history, 0, null, null);
const help = explain.answer('我该去哪里？', frame, history, 0, null, null);
const targetAnswer = explain.answer('此时机器人2准备接哪个球？', frame, history, 0, null, null);
const missed = explain.answer('为什么漏接A2？', frame, history, 0, null, null);
const source = explain.answer('这是NN还是规则改选？', frame, history, 0, null, null);
assert.match(why, /B1/);
assert.match(avoid, /A1/);
assert.match(avoid, /不可达|不满足|不能及时覆盖/);
assert.match(help, /右侧/);
assert.match(targetAnswer, /^机器人2负责/);
assert.doesNotMatch(targetAnswer, /我负责|我去/);
const nearSmall = explain.answer('此时机器人2准备接哪个球？',
  { ...frame, decision: { ...decision, planEvidence: { ...evidence,
    target_ball_id: 'A1', target_kind: 'small', requires_partner: false, self_distance: .4 } } },
  history, 0, null, null);
assert.match(nearSmall, /机器人2负责小球A1.*不足1格/);
assert.doesNotMatch(nearSmall, /约0格/);
assert.match(missed, /小球A2.*计1次/);
assert.match(source, /NN建议.*规则/);
assert.notEqual(why, avoid);
const bubble = explain.bubble(decision);
assert.match(bubble.text + bubble.detail, /B1/);
assert.doesNotMatch(bubble.text + bubble.detail, /概率|threshold|秒/);
const range = explain.answerRange('这段漏了什么？', [frame, { ...frame, frame: 13,
  events: [{ event: 'miss_scored', encounter_id: 'A2:1', ball_id: 'A2', kind: 'small' }] }]);
assert.match(range, /漏接1次/);
const sourceRange = explain.answerRange('这段规则改选了几次？', [frame]);
assert.match(sourceRange, /规则改选/);
assert.notEqual(sourceRange, range);
const combined = explain.answer('为什么不接A1，我该去哪里？', frame, history, 0, null, null);
assert.match(combined, /A1/);
assert.match(combined, /右侧/);
const noOldResult = explain.answer('为什么没接住B1？', { ...frame, frame: 13, events: [] },
  [{ ...frame, events: [{ ...frame.events[0], ball_id: 'B1', encounter_id: 'B1:0' }] },
    { ...frame, frame: 13, events: [] }], 1, null, null);
assert.match(noOldResult, /第13帧没有/);
const otherSide = explain.bubble({ ...decision, planEvidence: { ...evidence,
  target_ball_id: 'B2', contact_side: 'right', partner_status: 'covered' } });
assert.match(otherSide.text + otherSide.detail, /B2.*右侧.*另一侧/);
console.log('question-aware Pong explanations: passed');
