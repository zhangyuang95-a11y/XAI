const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {test} = require('node:test');

const app = require(path.resolve(
  __dirname, '../ui/warehouse_family_feedback_research/app.js'));

function view({version=4, frame=8, run='run-1', positions=[[5, 4], [7, 6]]}={}) {
  return {
    session_id: 'session-1', run_id: run, version,
    state: {frame, agents: [
      {id: 'robot_1', position: positions[0]},
      {id: 'robot_2', position: positions[1]},
    ]},
  };
}

test('confirmed consecutive action interpolates both actual robot moves', () => {
  const before = view();
  const after = view({version:5, frame:9, positions:[[5, 5], [6, 6]]});
  const motion = app.confirmedMotion(before, after, 'action', null);
  assert.ok(motion);
  assert.equal(motion.duration, 380);
  assert.deepEqual(app.interpolateMotion(motion, 0), {
    robot_1:[5, 4], robot_2:[7, 6],
  });
  assert.deepEqual(app.interpolateMotion(motion, 0.5), {
    robot_1:[5, 4.5], robot_2:[6.5, 6],
  });
  assert.deepEqual(app.interpolateMotion(motion, 1), {
    robot_1:[5, 5], robot_2:[6, 6],
  });
});

test('stationary robot remains fixed while its teammate moves', () => {
  const motion = app.confirmedMotion(
    view(), view({version:5, frame:9, positions:[[4, 4], [7, 6]]}),
    'action', null);
  assert.deepEqual(app.interpolateMotion(motion, 0.37).robot_2, [7, 6]);
});

test('wait, blocked moves, replay, refresh and stale responses never animate', () => {
  const before = view(), unchanged = view({version:5, frame:9});
  assert.equal(app.confirmedMotion(before, unchanged, 'action', null), null);
  assert.equal(app.confirmedMotion(before,
    view({version:5, frame:9, positions:[[4, 4], [7, 6]]}), 'action', 8), null);
  assert.equal(app.confirmedMotion(before,
    view({version:5, frame:9, positions:[[4, 4], [7, 6]]}), 'question', null), null);
  assert.equal(app.confirmedMotion(before,
    view({version:6, frame:10, positions:[[4, 4], [7, 6]]}), 'action', null), null);
  assert.equal(app.confirmedMotion(before,
    view({version:5, frame:9, run:'run-2',positions:[[4, 4], [7, 6]]}), 'action', null), null);
});

test('easing is continuous, monotone and clamps endpoints', () => {
  const values=[-1,0,.1,.25,.5,.75,.9,1,2].map(app.easeMotion);
  assert.equal(values[0],0);assert.equal(values[1],0);
  assert.equal(values.at(-1),1);assert.equal(values.at(-2),1);
  for(let i=1;i<values.length;i++)assert.ok(values[i]>=values[i-1]);
  assert.ok(values[3]>0 && values[3]<values[4]);
});

test('participant answer and collapsed evidence remain separate', () => {
  const answers=app.visibleAnswers({
    run_id:'run-1', explain_allowed:true,
    release:{explanation_ready:true},
    flow:{stage:'task1'}, allowed_kinds:['question'],
    answers:[{id:'q-1',run_id:'run-1',status:'complete',frame:12,
      question:'为什么？',text:'机器人2向右，距离缩短了一格。',
      evidence_detail:'动作概率：向右 80%',sources:['frozen NN']}],
  });
  assert.equal(answers.length,1);
  assert.equal(answers[0].text,'机器人2向右，距离缩短了一格。');
  assert.equal(answers[0].evidence_detail,'动作概率：向右 80%');
  assert.deepEqual(answers[0].sources,['frozen NN']);
});

test('long translated subtitle is clipped before the centered workflow', () => {
  const css=fs.readFileSync(path.resolve(
    __dirname, '../ui/warehouse_family_feedback_research/styles.css'),'utf8');
  const subtitleRule=css.match(/\.subtitle\s*\{([^}]*)\}/)?.[1] || '';
  assert.match(subtitleRule,/min-width:\s*0/);
  assert.match(subtitleRule,/overflow:\s*hidden/);
  assert.match(subtitleRule,/text-overflow:\s*ellipsis/);
  assert.match(subtitleRule,/white-space:\s*nowrap/);
});
