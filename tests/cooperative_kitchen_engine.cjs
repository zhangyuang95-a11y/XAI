'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const E = require('../ui/cooperative_kitchen_demo/engine.js');
const actor = (state, id = 'human') => state.actors.find(a => a.id === id);
const place = (state, id, position, facing, holding = null) => Object.assign(actor(state, id), { position, facing, holding });
const advance = (state, action = 'WAIT', auto = false) => E.step(state, action, { auto }).state;

function automatic(preset, initial = E.reset(preset)) {
  let state = initial;
  const history = [E.snapshot(state)];
  while (!state.done) {
    const result = E.step(state, 'WAIT', { auto: true });
    for (const event of result.events) {
      assert.equal(typeof E.eventText(event, 'zh'), 'string');
      assert.equal(typeof E.eventText(event, 'en'), 'string');
    }
    state = result.state;
    assert.deepEqual(E.restore(E.snapshot(state)), state);
    history.push(E.snapshot(state));
  }
  return { state, history };
}

test('fixed layout and both assignments have the same abilities and initial heading', () => {
  assert.deepEqual(E.MAP, ['#########', '#I.X#X.D#', '#...C...#', '#H..#..A#', '#...C...#', '#S..#..P#', '#########']);
  const left = E.reset('supply');
  const right = E.reset('cook');
  assert.deepEqual(actor(left).position, [3, 1]);
  assert.deepEqual(actor(right).position, [3, 7]);
  assert.deepEqual(actor(left, 'ai').position, actor(right).position);
  for (const state of [left, right]) {
    assert.equal(state.maxSteps, 120); assert.equal(state.targetOrders, 2);
    for (const chef of state.actors) assert.equal(chef.facing, 'UP');
    assert.deepEqual(E.restore(state), state);
  }
  assert.throws(() => E.reset('unknown'));
});

test('WAIT and invalid interaction each consume one step without moving the human', () => {
  const initial = E.reset();
  const waiting = advance(initial);
  assert.equal(waiting.turn, 1);
  assert.deepEqual(actor(waiting), actor(initial));
  const invalid = E.step(waiting, 'INTERACT');
  assert.equal(invalid.state.turn, 2);
  assert.deepEqual(actor(invalid.state), actor(initial));
  assert.equal(invalid.events.find(e => e.actor === 'human').reason, 'no_station');
  assert.throws(() => E.step(initial, 'JUMP'));
});

test('movement changes heading even against a wall, and stations prevent crossing sides', () => {
  const initial = E.reset();
  place(initial, 'human', [2, 3], 'UP');
  let state = advance(initial, 'RIGHT');
  assert.deepEqual(actor(state).position, [2, 3]);
  assert.equal(actor(state).facing, 'RIGHT');
  state = advance(state, 'DOWN');
  assert.deepEqual(actor(state).position, [3, 3]);
  assert.equal(actor(state).facing, 'DOWN');
  state = advance(state, 'RIGHT');
  assert.deepEqual(actor(state).position, [3, 3]);
  assert.equal(actor(state).facing, 'RIGHT');
  assert.equal(state.turn, 3);
});

test('ingredient and plate sources are infinite but require an empty hand', () => {
  for (const [preset, location, facing, item] of [['supply', [2, 1], 'UP', 'onion'], ['cook', [2, 7], 'UP', 'plate']]) {
    const initial = E.reset(preset); place(initial, 'human', location, facing);
    const taken = advance(initial, 'INTERACT');
    assert.equal(actor(taken).holding, item);
    const repeated = E.step(taken, 'INTERACT');
    assert.equal(actor(repeated.state).holding, item);
    assert.equal(repeated.events.find(e => e.actor === 'human').reason, 'hands_full');
  }
});

test('a counter holds one object, permits explicit pickup, and never swaps held items', () => {
  const initial = E.reset(); place(initial, 'human', [4, 3], 'RIGHT', 'plate');
  const dropped = advance(initial, 'INTERACT');
  assert.equal(dropped.counters['4,4'], 'plate'); assert.equal(actor(dropped).holding, null);
  actor(dropped).holding = 'onion';
  const blocked = E.step(dropped, 'INTERACT');
  assert.equal(blocked.state.counters['4,4'], 'plate'); assert.equal(actor(blocked.state).holding, 'onion');
  assert.equal(blocked.events.find(e => e.actor === 'human').reason, 'counter_occupied');
  actor(blocked.state).holding = null;
  const picked = advance(blocked.state, 'INTERACT');
  assert.equal(picked.counters['4,4'], null); assert.equal(actor(picked).holding, 'plate');
});

test('same-turn drop and pickup cannot bypass the shared counter', () => {
  const initial = E.reset();
  place(initial, 'human', [2, 3], 'RIGHT');
  place(initial, 'ai', [2, 5], 'LEFT', 'soup');
  initial.counters['4,4'] = 'onion';
  const result = E.step(initial, 'INTERACT');
  assert.equal(result.actions.ai, 'INTERACT');
  assert.equal(result.state.counters['2,4'], 'soup');
  assert.equal(actor(result.state).holding, null);
  assert.equal(actor(result.state, 'ai').holding, null);
  assert.equal(result.events.find(e => e.actor === 'human').reason, 'counter_empty');
});

test('simultaneous pickup uses alternating turn priority without duplication', () => {
  for (const turn of [0, 1]) {
    const initial = E.reset(); initial.turn = turn;
    place(initial, 'human', [2, 3], 'RIGHT'); place(initial, 'ai', [2, 5], 'LEFT');
    initial.counters['2,4'] = 'onion';
    const result = E.step(initial, 'INTERACT');
    assert.deepEqual(result.actions, { human: 'INTERACT', ai: 'INTERACT' });
    assert.equal(result.state.counters['2,4'], null);
    assert.equal(actor(result.state, turn % 2 === 0 ? 'human' : 'ai').holding, 'onion');
    assert.equal(actor(result.state, turn % 2 === 0 ? 'ai' : 'human').holding, null);
    assert.equal(result.events.filter(e => e.type === 'conflict').length, 1);
  }
});

test('simultaneous placement does not overwrite or destroy the losing object', () => {
  for (const turn of [0, 1]) {
    const initial = E.reset(); initial.turn = turn;
    place(initial, 'human', [2, 3], 'RIGHT', 'onion'); place(initial, 'ai', [2, 5], 'LEFT', 'soup');
    initial.counters['4,4'] = 'plate';
    const result = E.step(initial, 'INTERACT');
    const winner = turn % 2 === 0 ? 'human' : 'ai';
    const loser = winner === 'human' ? 'ai' : 'human';
    assert.equal(result.state.counters['2,4'], winner === 'human' ? 'onion' : 'soup');
    assert.equal(actor(result.state, winner).holding, null);
    assert.equal(actor(result.state, loser).holding, loser === 'human' ? 'onion' : 'soup');
  }
});

test('third onion starts four subsequent cooking steps and cannot plate on the readiness transition', () => {
  let state = E.reset('cook');
  place(state, 'human', [5, 6], 'RIGHT', 'onion'); state.pot.ingredients = 2;
  state = advance(state, 'INTERACT');
  assert.deepEqual(state.pot, { ingredients: 3, remaining: 4, ready: false });
  actor(state).holding = 'plate';
  for (const remaining of [3, 2, 1]) {
    state = advance(state);
    assert.equal(state.pot.remaining, remaining); assert.equal(state.pot.ready, false);
  }
  const earlyPlate = E.step(state, 'INTERACT');
  assert.equal(earlyPlate.events.find(e => e.actor === 'human').reason, 'pot_cooking');
  assert.equal(actor(earlyPlate.state).holding, 'plate');
  assert.deepEqual(earlyPlate.state.pot, { ingredients: 3, remaining: 0, ready: true });
  state = advance(earlyPlate.state, 'INTERACT');
  assert.equal(actor(state).holding, 'soup');
  assert.deepEqual(state.pot, { ingredients: 0, remaining: 0, ready: false });
});

test('a full pot rejects a fourth onion; plating needs a held plate', () => {
  const initial = E.reset('cook');
  initial.pot = { ingredients: 3, remaining: 0, ready: true };
  place(initial, 'human', [5, 6], 'RIGHT', 'onion');
  let result = E.step(initial, 'INTERACT');
  assert.equal(result.state.pot.ingredients, 3); assert.equal(actor(result.state).holding, 'onion');
  actor(result.state).holding = null;
  result = E.step(result.state, 'INTERACT');
  assert.equal(actor(result.state).holding, null); assert.equal(result.state.pot.ready, true);
  assert.equal(result.events.find(e => e.actor === 'human').reason, 'plate_needed');
});

test('serving only accepts soup and success has priority at the step limit', () => {
  let state = E.reset();
  place(state, 'human', [5, 2], 'LEFT', 'onion');
  let result = E.step(state, 'INTERACT');
  assert.equal(result.state.orders, 0); assert.equal(actor(result.state).holding, 'onion');
  assert.equal(result.events.find(e => e.actor === 'human').reason, 'soup_needed');
  state = result.state; state.turn = 119; state.orders = 1; actor(state).holding = 'soup';
  result = E.step(state, 'INTERACT');
  assert.equal(result.state.turn, 120); assert.equal(result.state.orders, 2);
  assert.equal(result.state.reason, 'success'); assert.equal(actor(result.state).holding, null);
  assert.deepEqual(advance(result.state), result.state);
});

test('120 steps without two servings produces a stable timeout', () => {
  const initial = E.reset(); initial.turn = 119;
  const result = advance(initial);
  assert.equal(result.done, true); assert.equal(result.reason, 'timeout'); assert.equal(result.turn, 120);
  assert.deepEqual(advance(result, 'UP'), result);
});

test('trash releases every held item and an empty-handed discard does nothing', () => {
  for (const item of ['onion', 'plate', 'soup']) {
    const initial = E.reset(); place(initial, 'human', [2, 3], 'UP', item);
    const result = E.step(initial, 'INTERACT');
    assert.equal(actor(result.state).holding, null); assert.equal(result.state.orders, 0);
    assert.equal(result.events.find(e => e.actor === 'human').type, 'discard');
  }
  const initial = E.reset(); place(initial, 'human', [2, 3], 'UP');
  assert.equal(E.step(initial, 'INTERACT').events.find(e => e.actor === 'human').reason, 'hands_empty');
});

test('supply demand accounts for pot, both counters, and right-held ingredients', () => {
  const state = E.reset('cook'); state.pot.ingredients = 1;
  state.counters['2,4'] = 'onion'; actor(state).holding = 'onion';
  const decision = E.decide(state, 'ai');
  assert.equal(decision.facts.neededOnions, 0);
  assert.equal(decision.facts.stagedOnions, 1);
  assert.equal(decision.facts.rightHeldOnions, 1);
  assert.equal(decision.rule, 'wait_soup');
});

test('a soup-holding teammate waits at blocked counters, then resumes after a real pickup', () => {
  const initial = E.reset();
  place(initial, 'human', [4, 3], 'RIGHT'); place(initial, 'ai', [4, 5], 'LEFT', 'soup');
  initial.counters = { '2,4': 'plate', '4,4': 'onion' };
  assert.equal(E.decide(initial, 'ai').rule, 'wait_space');
  const result = E.step(initial, 'INTERACT');
  assert.equal(result.actions.ai, 'WAIT'); assert.equal(actor(result.state, 'ai').holding, 'soup');
  assert.equal(result.state.counters['4,4'], null);
  const resumed = E.step(result.state, 'WAIT');
  assert.equal(resumed.actions.ai, 'INTERACT');
  assert.equal(resumed.state.counters['4,4'], 'soup');
  assert.equal(actor(resumed.state, 'ai').holding, null);
});

test('left teammate actively clears congestion when the human is holding soup', () => {
  const initial = E.reset('cook');
  place(initial, 'human', [4, 5], 'LEFT', 'soup'); place(initial, 'ai', [4, 3], 'RIGHT');
  initial.counters = { '2,4': 'onion', '4,4': 'onion' };
  assert.equal(E.decide(initial, 'ai').rule, 'clear_for_handoff');
  const result = E.step(initial, 'WAIT');
  assert.equal(result.state.counters['4,4'], null);
  assert.equal(actor(result.state, 'ai').holding, 'onion');
  assert.equal(E.decide(result.state, 'ai').rule, 'discard_for_handoff');
  const handedBack = E.step(result.state, 'INTERACT');
  assert.equal(handedBack.state.counters['4,4'], 'soup');
});

test('right teammate clears wrong plates and can recover to finish the demo', () => {
  const initial = E.reset(); initial.counters = { '2,4': 'plate', '4,4': 'plate' };
  assert.equal(E.decide(initial, 'ai').rule, 'clear_plate');
  const run = automatic('supply', initial);
  assert.equal(run.state.reason, 'success'); assert.equal(run.state.orders, 2);
});

test('the teammate only fetches a plate after the soup is ready', () => {
  const state = E.reset();
  state.pot = { ingredients: 3, remaining: 2, ready: false };
  place(state, 'ai', [2, 7], 'UP');
  assert.equal(E.decide(state, 'ai').action, 'WAIT');
  let result = advance(state);
  assert.equal(actor(result, 'ai').holding, null);
  result = advance(result);
  assert.equal(result.pot.ready, true); assert.equal(actor(result, 'ai').holding, null);
  result = advance(result);
  assert.equal(actor(result, 'ai').holding, 'plate');
});

test('both decisions read the same pre-action state and never inspect the new human command', () => {
  const initial = E.reset();
  const expected = E.decide(initial, 'ai');
  for (const action of E.ACTIONS) {
    assert.deepEqual(E.step(initial, action).decisions.ai, expected);
    assert.equal(E.step(initial, action).actions.ai, expected.action);
  }
  const automated = E.step(initial, 'LEFT', { auto: true });
  assert.deepEqual(automated.decisions.human, E.decide(initial, 'human'));
  assert.deepEqual(automated.decisions.ai, expected);
});

test('snapshots are independent, validate corruption, and reproduce the exact continuation', () => {
  let state = E.reset();
  for (let i = 0; i < 18; i++) state = advance(state, 'WAIT', true);
  const saved = E.snapshot(state);
  const restored = E.restore(saved);
  assert.deepEqual(E.step(state, 'UP'), E.step(restored, 'UP'));
  restored.actors[0].holding = 'soup'; assert.notDeepEqual(restored, saved);
  const invalids = [
    { ...saved, schema: 2 }, { ...saved, maxSteps: 240 }, { ...saved, turn: -1 },
    { ...saved, pot: { ingredients: 3, remaining: 0, ready: false } },
    { ...saved, counters: { '2,4': 'tomato', '4,4': null } }, { ...saved, done: true, reason: null },
  ];
  for (const bad of invalids) assert.throws(() => E.restore(bad));
  const badActor = E.snapshot(saved); badActor.actors[0].position = [0, 0];
  assert.throws(() => E.restore(badActor));
});

test('all explanations bind the selected frame and preserve both current and historical state', () => {
  let state = E.reset();
  for (let i = 0; i < 31; i++) state = advance(state, 'WAIT', true);
  const original = JSON.stringify(state);
  for (const language of ['zh', 'en']) for (const kind of ['why', 'waiting', 'counterfactual']) {
    const explanation = E.explain(state, kind, language);
    assert.equal(explanation.frame, 31); assert.equal(explanation.kind, kind);
    assert.deepEqual(explanation.decision, E.decide(state, 'ai'));
    assert.ok(explanation.title.length > 3 && explanation.text.length > 30);
    assert.equal(JSON.stringify(state), original);
  }
  assert.match(E.explain(state, 'waiting', 'zh').text, /4 个联合步/);
  assert.match(E.explain(state, 'waiting', 'en').text, /4 joint steps/);
  assert.throws(() => E.explain(state, 'unsupported'));
});

test('counterfactual is exactly three isolated human WAIT steps and stops at terminal', () => {
  let state = E.reset('cook');
  for (let i = 0; i < 10; i++) state = advance(state, 'WAIT', true);
  const answer = E.explain(state, 'counterfactual');
  let expected = E.snapshot(state);
  for (let i = 0; i < 3; i++) expected = advance(expected);
  assert.deepEqual(answer.forecast.state, expected); assert.equal(answer.forecast.steps, 3);
  state.turn = 119;
  const ending = E.explain(state, 'counterfactual');
  assert.equal(ending.forecast.steps, 1); assert.equal(ending.forecast.state.reason, 'timeout');
  const terminal = E.explain(ending.forecast.state, 'counterfactual');
  assert.equal(terminal.forecast.steps, 0);
});

test('pure dynamics and policy calls never mutate their inputs', () => {
  const state = E.reset();
  const freeze = object => { Object.freeze(object); for (const value of Object.values(object)) if (value && typeof value === 'object') freeze(value); };
  freeze(state);
  assert.doesNotThrow(() => E.step(state, 'UP'));
  assert.doesNotThrow(() => E.step(state, 'WAIT', { auto: true }));
  assert.doesNotThrow(() => E.decide(state, 'ai'));
  assert.doesNotThrow(() => E.explain(state, 'counterfactual'));
});

test('both automatic assignments finish two orders within 120 steps with deterministic replay', () => {
  for (const preset of ['supply', 'cook']) {
    const first = automatic(preset);
    const second = automatic(preset);
    assert.equal(first.state.orders, 2); assert.equal(first.state.reason, 'success');
    assert.ok(first.state.turn <= 120); assert.equal(first.state.turn, 104);
    assert.deepEqual(first.history, second.history);
    for (const frame of first.history) {
      assert.ok(frame.pot.ingredients + Object.values(frame.counters).filter(item => item === 'onion').length
        + frame.actors.filter(a => a.holding === 'onion').length <= 3, 'program should never overfeed');
    }
  }
});
