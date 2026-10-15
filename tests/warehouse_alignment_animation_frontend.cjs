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

test('wait, unconfirmed blocked moves, replay, refresh and stale responses never animate', () => {
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

for (const kind of ['same_target','swap','occupied_stationary','none']) {
  test(`confirmed ${kind} rejection approaches and rebounds without changing final state`, () => {
    const before=view({positions:[[3,2],[3,4]]});
    const after=view({version:5,frame:9,positions:[[3,2],[3,4]]});
    after.state.public_feedback={valid:true,collision_kind:kind,
      submitted_actions:{robot_1:'RIGHT',robot_2:kind==='occupied_stationary'?'WAIT':'LEFT'},
      move_canceled:{robot_1:true,robot_2:kind!=='occupied_stationary'}};
    const saved=JSON.stringify({before,after});
    const motion=app.confirmedMotion(before,after,'action',null);
    assert.equal(motion.duration,380);
    assert.deepEqual(app.interpolateMotion(motion,0),{robot_1:[3,2],robot_2:[3,4]});
    const contact=app.interpolateMotion(motion,.45);
    assert.ok(contact.robot_1[1]>2 && contact.robot_1[1]<3);
    if(kind==='occupied_stationary')assert.deepEqual(contact.robot_2,[3,4]);
    else assert.ok(contact.robot_2[1]<4 && contact.robot_2[1]>3);
    assert.deepEqual(app.interpolateMotion(motion,1),{robot_1:[3,2],robot_2:[3,4]});
    assert.equal(JSON.stringify({before,after}),saved);
    assert.equal(!!app.collisionNotice(after.state.public_feedback,'en'),kind!=='none');
    assert.equal(app.confirmedMotion(before,after,'question',null),null);
    assert.equal(app.confirmedMotion(before,after,'action',0),null);
  });
}

test('WAIT does not become a movement even if the collision flag is present',()=>{
  const before=view(),after=view({version:5,frame:9});
  after.state.public_feedback={valid:true,collision_kind:'occupied_stationary',
    submitted_actions:{robot_1:'WAIT',robot_2:'WAIT'},move_canceled:{robot_1:true,robot_2:true}};
  assert.equal(app.confirmedMotion(before,after,'action',null),null);
});

test('only the latest answer is presented',()=>{
  const current=view();
  current.explain_allowed=true;
  current.allowed_kinds=['question'];
  current.release={model_ready:true,explanation_ready:true};
  current.flow={mode:'study',stage:'task1'};
  current.answers=[
    {id:'new',run_id:current.run_id,question:'new',answer:'new answer',frame:8},
    {id:'old',run_id:current.run_id,question:'old',answer:'old answer',frame:3},
  ];
  assert.deepEqual(app.visibleAnswers(current).map(answer=>answer.id),['new']);
});

test('each confirmed AI-AI tutorial step reserves the full 380ms even when stationary', () => {
  const before={...view({run:null}),flow:{mode:'study',stage:'instructions'},
    tutorial:{frame_index:0,total_frames:4}};
  const after={...view({version:5,frame:9,run:null}),flow:{mode:'study',stage:'instructions'},
    tutorial:{frame_index:1,total_frames:4}};
  const motion=app.confirmedMotion(before,after,'tutorial_advance',null);
  assert.ok(motion);assert.equal(motion.duration,380);assert.equal(motion.tutorial,true);
  assert.equal(app.confirmedMotion(before,{...after,tutorial:{...after.tutorial,frame_index:2}},'tutorial_advance',null),null);
  assert.equal(app.confirmedMotion(before,after,'tutorial_select',null),null);
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
  assert.equal(Object.hasOwn(answers[0],'sources'),false);
});

test('scorecard shows the five measures used by the native runtime', () => {
  const html=fs.readFileSync(path.resolve(
    __dirname, '../ui/warehouse_family_feedback_research/index.html'),'utf8');
  const scoreStrip=html.slice(html.indexOf('<div class="score-strip metrics">'),
    html.indexOf('<div class="canvas-wrap">'));
  const metricIds=[...scoreStrip.matchAll(/<strong id="([^"]+Value)">/g)].map(match=>match[1]);
  assert.deepEqual(metricIds,[
    'scoreValue','deliveriesValue','stepsValue','collisionsValue','shutdownsValue',
  ]);
  assert.deepEqual(app.frameMetrics({
    state:{frame:7,total_deliveries:2,robot_collision_events:1,
      shutdown_count:0},
    metrics:{score:-14},
  }),{deliveries:2,score:-14,legacy_score:null,steps:7,collisions:1,
    shutdowns:0});
  const source=fs.readFileSync(path.resolve(
    __dirname, '../ui/warehouse_family_feedback_research/app.js'),'utf8');
  assert.match(source,/机器人碰撞 −200，断电 −50，每步 −1。/);
  assert.match(source,/−200 per robot collision, −50 per shutdown, −1 per turn\./);
  assert.doesNotMatch(source,/参与者绕路每单位 −2/);
  assert.doesNotMatch(source,/−2 per human detour unit/);
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
