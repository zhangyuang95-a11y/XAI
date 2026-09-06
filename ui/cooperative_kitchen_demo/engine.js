(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.KitchenEngine = api;
}(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  const MAP = Object.freeze([
    '#########', '#I.X#X.D#', '#...C...#', '#H..#..A#',
    '#...C...#', '#S..#..P#', '#########',
  ]);
  const DIRECTIONS = Object.freeze({ UP: [-1, 0], DOWN: [1, 0], LEFT: [0, -1], RIGHT: [0, 1] });
  const ACTIONS = Object.freeze([...Object.keys(DIRECTIONS), 'INTERACT', 'WAIT']);
  const COUNTERS = Object.freeze(['2,4', '4,4']);
  const STATIONS = Object.freeze({ ingredient: [1, 1], plate: [1, 7], pot: [5, 7], serve: [5, 1], leftTrash: [1, 3], rightTrash: [1, 5] });
  const ITEMS = [null, 'onion', 'plate', 'soup'];
  const clone = value => JSON.parse(JSON.stringify(value));
  const same = (a, b) => a[0] === b[0] && a[1] === b[1];
  const key = p => p.join(',');
  const position = k => k.split(',').map(Number);
  const add = (p, d) => [p[0] + d[0], p[1] + d[1]];
  const tile = p => MAP[p[0]] && MAP[p[0]][p[1]];
  const walkable = p => ['.', 'H', 'A'].includes(tile(p));
  const actorById = (state, id) => state.actors.find(actor => actor.id === id);
  const countCounters = (state, item) => COUNTERS.filter(k => state.counters[k] === item).length;
  const counterPositions = (state, item) => COUNTERS.filter(k => state.counters[k] === item).map(position);

  function reset(preset = 'supply') {
    if (!['supply', 'cook'].includes(preset)) throw new Error('Unknown kitchen preset');
    const humanLeft = preset === 'supply';
    return {
      schema: 1, preset, turn: 0, maxSteps: 120, targetOrders: 2, orders: 0, done: false, reason: null,
      actors: [
        { id: 'human', side: humanLeft ? 'left' : 'right', position: humanLeft ? [3, 1] : [3, 7], facing: 'UP', holding: null },
        { id: 'ai', side: humanLeft ? 'right' : 'left', position: humanLeft ? [3, 7] : [3, 1], facing: 'UP', holding: null },
      ],
      pot: { ingredients: 0, remaining: 0, ready: false },
      counters: { '2,4': null, '4,4': null },
    };
  }

  function snapshot(state) { return clone(state); }

  function restore(saved) {
    const state = clone(saved);
    const integer = (value, min, max) => Number.isInteger(value) && value >= min && value <= max;
    if (!state || state.schema !== 1 || !['supply', 'cook'].includes(state.preset)
        || !integer(state.turn, 0, 120) || state.maxSteps !== 120 || state.targetOrders !== 2
        || !integer(state.orders, 0, 2) || typeof state.done !== 'boolean'
        || ![null, 'success', 'timeout'].includes(state.reason)) throw new Error('Invalid kitchen snapshot');
    if (!Array.isArray(state.actors) || state.actors.length !== 2
        || state.actors.map(a => a.id).sort().join(',') !== 'ai,human') throw new Error('Invalid actors');
    for (const actor of state.actors) {
      if (!['left', 'right'].includes(actor.side) || !Array.isArray(actor.position)
          || actor.position.length !== 2 || !actor.position.every(Number.isInteger)
          || !walkable(actor.position) || !DIRECTIONS[actor.facing] || !ITEMS.includes(actor.holding)
          || (actor.side === 'left' ? actor.position[1] > 3 : actor.position[1] < 5)) throw new Error('Invalid actor');
    }
    if (state.actors[0].side === state.actors[1].side
        || actorById(state, 'human').side !== (state.preset === 'supply' ? 'left' : 'right')) throw new Error('Invalid role assignment');
    if (!state.counters || Object.keys(state.counters).sort().join('|') !== COUNTERS.join('|')
        || !COUNTERS.every(k => ITEMS.includes(state.counters[k]))) throw new Error('Invalid counters');
    const pot = state.pot;
    if (!pot || !integer(pot.ingredients, 0, 3) || !integer(pot.remaining, 0, 4)
        || typeof pot.ready !== 'boolean'
        || (pot.ingredients < 3 && (pot.remaining !== 0 || pot.ready))
        || (pot.ingredients === 3 && ((pot.remaining === 0) !== pot.ready))) throw new Error('Invalid pot');
    if (state.done !== (state.reason !== null)
        || (state.reason === 'success' && state.orders !== 2)
        || (state.reason === 'timeout' && (state.turn !== 120 || state.orders === 2))
        || (!state.done && (state.orders === 2 || state.turn === 120))) throw new Error('Invalid terminal state');
    return state;
  }

  // The route ends facing a station. Turning toward a blocked station is a
  // regular directional action, just as it is for the human player.
  function routeTo(state, actor, target) {
    const other = state.actors.find(a => a.id !== actor.id);
    const queue = [{ p: actor.position, first: null, distance: 0, facing: actor.facing }];
    const visited = new Set([key(actor.position)]);
    let best = null;
    for (let index = 0; index < queue.length; index++) {
      const current = queue[index];
      for (const [direction, delta] of Object.entries(DIRECTIONS)) {
        if (!same(add(current.p, delta), target)) continue;
        const turnCost = current.facing === direction ? 0 : 1;
        const candidate = {
          action: current.first || (turnCost ? direction : 'INTERACT'),
          distance: current.distance + turnCost + 1,
        };
        if (!best || candidate.distance < best.distance) best = candidate;
      }
      for (const [direction, delta] of Object.entries(DIRECTIONS)) {
        const next = add(current.p, delta);
        if (!walkable(next) || visited.has(key(next)) || (other && same(next, other.position))) continue;
        visited.add(key(next));
        queue.push({ p: next, first: current.first || direction, distance: current.distance + 1, facing: direction });
      }
    }
    return best || { action: 'WAIT', distance: Infinity };
  }

  function closest(state, actor, targets) {
    return targets.map(target => ({ target, ...routeTo(state, actor, target) }))
      .sort((a, b) => a.distance - b.distance)[0];
  }

  function decide(state, actorId) {
    const actor = actorById(state, actorId);
    if (!actor) throw new Error('Unknown kitchen actor');
    const right = state.actors.find(a => a.side === 'right');
    const stagedOnions = countCounters(state, 'onion');
    const rightHeldOnions = right && right.holding === 'onion' ? 1 : 0;
    const neededOnions = Math.max(0, 3 - state.pot.ingredients - stagedOnions - rightHeldOnions);
    const facts = {
      side: actor.side, holding: actor.holding, orders: state.orders,
      pot: clone(state.pot), counterItems: clone(state.counters),
      stagedOnions, rightHeldOnions, neededOnions, waitingFor: null,
    };
    const result = (action, rule, target = null, extra = {}) => ({ action, rule, target: target && [...target], facts: { ...facts, ...extra } });
    const go = (rule, target, extra = {}) => result(routeTo(state, actor, target).action, rule, target, extra);
    const nearest = (rule, targets, extra = {}) => {
      const best = closest(state, actor, targets);
      return best ? result(best.action, rule, best.target, extra) : result('WAIT', 'wait_space', null, { waitingFor: 'space' });
    };
    const waitNear = (rule, target, waitingFor) => {
      const action = routeTo(state, actor, target).action;
      return result(action === 'INTERACT' ? 'WAIT' : action, rule, target, { waitingFor });
    };
    const empty = counterPositions(state, null);
    const onions = counterPositions(state, 'onion');
    const plates = counterPositions(state, 'plate');
    const soups = counterPositions(state, 'soup');
    const trash = actor.side === 'left' ? STATIONS.leftTrash : STATIONS.rightTrash;
    if (state.done) return result('WAIT', 'finished');

    if (actor.side === 'left') {
      if (actor.holding === 'soup') return go('serve_soup', STATIONS.serve);
      if (actor.holding === 'plate') return go('discard_plate', trash);
      if (actor.holding === 'onion') {
        if (right && right.holding === 'soup') return go('discard_for_handoff', trash);
        if (neededOnions === 0) return go('discard_extra_onion', trash);
        // Prefer the upper input counter; the lower one is the return path.
        if (empty.length) return go('handoff_onion', empty.find(p => p[0] === 2) || empty[0]);
        return result('WAIT', 'wait_space', null, { waitingFor: 'space' });
      }
      if (soups.length) return nearest('collect_soup', soups);
      if (plates.length) return nearest('clear_plate', plates);
      if (right && right.holding === 'soup') {
        if (!empty.length && onions.length) return nearest('clear_for_handoff', onions);
        return waitNear('wait_soup_handoff', [4, 4], 'soup_handoff');
      }
      if (state.pot.ingredients === 3 && onions.length) return nearest('clear_extra_onion', onions);
      if (neededOnions > 0) {
        if (empty.length) return go('get_onion', STATIONS.ingredient);
        return waitNear('wait_pickup', [4, 4], 'counter_pickup');
      }
      return waitNear('wait_soup', [4, 4], state.pot.ready ? 'soup_handoff' : state.pot.remaining ? 'cooking' : 'pot_loading');
    }

    if (actor.holding === 'soup') {
      if (empty.length) return go('handoff_soup', empty.find(p => p[0] === 4) || empty[0]);
      return result('WAIT', 'wait_space', null, { waitingFor: 'space' });
    }
    if (actor.holding === 'onion') {
      if (state.pot.ingredients < 3) return go('load_pot', STATIONS.pot);
      return go('discard_extra_onion', trash);
    }
    if (actor.holding === 'plate') {
      if (state.pot.ready) return go('plate_soup', STATIONS.pot);
      if (state.pot.remaining) return waitNear('wait_cooking', STATIONS.pot, 'cooking');
      return go('discard_plate', trash);
    }
    if (state.pot.ready) {
      if (!empty.length && onions.length) return nearest('clear_extra_onion', onions);
      if (plates.length) return nearest('get_counter_plate', plates);
      if (!empty.length) return result('WAIT', 'wait_space', null, { waitingFor: 'space' });
      return go('get_plate', STATIONS.plate);
    }
    if (state.pot.remaining) {
      if (!empty.length && onions.length) return nearest('clear_extra_onion', onions);
      // Move into position during cooking; do not take a plate before ready.
      return waitNear('wait_cooking', STATIONS.plate, 'cooking');
    }
    if (onions.length) return nearest('collect_onion', onions);
    if (plates.length) return nearest('clear_plate', plates);
    return waitNear('wait_onion', [2, 4], 'onion');
  }

  function interactionIntent(state, actor) {
    const target = add(actor.position, DIRECTIONS[actor.facing]);
    const cell = tile(target);
    const base = { actor: actor.id, target, item: actor.holding, resource: key(target) };
    if (cell === 'C') {
      const item = state.counters[key(target)];
      if (!actor.holding && item) return { ...base, type: 'pickup', item };
      if (actor.holding && !item) return { ...base, type: 'drop' };
      return { ...base, type: 'invalid_interaction', reason: actor.holding ? 'counter_occupied' : 'counter_empty' };
    }
    if (cell === 'I' || cell === 'D') {
      if (!actor.holding) return { ...base, type: 'take_source', item: cell === 'I' ? 'onion' : 'plate', resource: null };
      return { ...base, type: 'invalid_interaction', reason: 'hands_full' };
    }
    if (cell === 'X' && actor.holding) return { ...base, type: 'discard', resource: null };
    if (cell === 'S' && actor.holding === 'soup') return { ...base, type: 'serve', resource: 'orders' };
    if (cell === 'P') {
      if (actor.holding === 'onion' && state.pot.ingredients < 3) return { ...base, type: 'load' };
      if (actor.holding === 'plate' && state.pot.ready) return { ...base, type: 'plate' };
      return { ...base, type: 'invalid_interaction', reason: state.pot.remaining ? 'pot_cooking' : state.pot.ready ? 'plate_needed' : state.pot.ingredients === 3 ? 'pot_full' : 'onion_needed' };
    }
    return { ...base, type: 'invalid_interaction', reason: cell === 'S' ? 'soup_needed' : cell === 'X' ? 'hands_empty' : 'no_station' };
  }

  function step(before, playerAction, options = {}) {
    if (!ACTIONS.includes(playerAction)) throw new Error('Unknown kitchen action');
    const state = snapshot(before);
    const ai = decide(before, 'ai');
    const human = options.auto ? decide(before, 'human') : null;
    const actions = { human: human ? human.action : playerAction, ai: ai.action };
    const decisions = { human, ai };
    const events = [];
    if (before.done) return { state, events, decisions, actions };
    const priority = before.turn % 2 === 0 ? ['human', 'ai'] : ['ai', 'human'];
    const destinations = {};
    for (const actor of before.actors) {
      const action = actions[actor.id];
      destinations[actor.id] = DIRECTIONS[action] && walkable(add(actor.position, DIRECTIONS[action]))
        ? add(actor.position, DIRECTIONS[action]) : [...actor.position];
    }
    const oldHuman = actorById(before, 'human');
    const oldAi = actorById(before, 'ai');
    if (same(destinations.human, destinations.ai)) {
      // A stationary chef keeps the tile. Otherwise parity resolves contention.
      const stationary = before.actors.find(a => same(a.position, destinations[a.id]));
      const loser = stationary ? (stationary.id === 'human' ? 'ai' : 'human') : priority[1];
      destinations[loser] = [...actorById(before, loser).position];
    }
    if (same(destinations.human, oldAi.position) && same(destinations.ai, oldHuman.position)) {
      destinations.human = [...oldHuman.position]; destinations.ai = [...oldAi.position];
    }
    for (const actor of state.actors) {
      const action = actions[actor.id];
      if (DIRECTIONS[action]) {
        const oldPosition = [...actor.position];
        actor.position = destinations[actor.id]; actor.facing = action;
        events.push({ type: same(oldPosition, actor.position) ? 'blocked' : 'move', actor: actor.id, action, from: oldPosition, to: [...actor.position] });
      } else if (action === 'WAIT') events.push({ type: 'wait', actor: actor.id });
    }
    // Compute every intent before applying any of them. A newly dropped item
    // cannot be picked up in the same joint step, nor can cooking be rushed.
    const intents = before.actors.filter(a => actions[a.id] === 'INTERACT').map(a => interactionIntent(before, a));
    const claimed = new Set();
    for (const id of priority) {
      const intent = intents.find(i => i.actor === id);
      if (!intent) continue;
      const actor = actorById(state, id);
      if (intent.type === 'invalid_interaction') { events.push(intent); continue; }
      if (intent.resource && claimed.has(intent.resource)) {
        events.push({ type: 'conflict', actor: id, target: intent.target }); continue;
      }
      if (intent.resource) claimed.add(intent.resource);
      switch (intent.type) {
        case 'pickup': actor.holding = intent.item; state.counters[key(intent.target)] = null; break;
        case 'drop': state.counters[key(intent.target)] = actor.holding; actor.holding = null; break;
        case 'take_source': actor.holding = intent.item; break;
        case 'discard': actor.holding = null; break;
        case 'serve': actor.holding = null; state.orders += 1; break;
        case 'load':
          actor.holding = null; state.pot.ingredients += 1;
          if (state.pot.ingredients === 3) state.pot.remaining = 4;
          break;
        case 'plate': actor.holding = 'soup'; state.pot = { ingredients: 0, remaining: 0, ready: false }; break;
      }
      events.push(intent);
      if (intent.type === 'load' && state.pot.ingredients === 3) events.push({ type: 'cooking_started', remaining: 4 });
    }
    if (before.pot.remaining > 0 && state.pot.ingredients === 3) {
      state.pot.remaining = before.pot.remaining - 1;
      if (state.pot.remaining === 0) { state.pot.ready = true; events.push({ type: 'soup_ready' }); }
    }
    state.turn += 1;
    if (state.orders >= state.targetOrders) {
      state.orders = state.targetOrders; state.done = true; state.reason = 'success'; events.push({ type: 'success' });
    } else if (state.turn >= state.maxSteps) {
      state.done = true; state.reason = 'timeout'; events.push({ type: 'timeout' });
    }
    return { state, events, decisions, actions };
  }

  const RULE_TEXT = {
    serve_soup: ['我正把手里的汤送到出餐口，完成共同订单。', 'I am taking the soup I am holding to the serving station to complete our shared order.'],
    discard_plate: ['当前需要空出手来处理食材；我会把不需要的盘子放入垃圾桶。', 'I need a free hand to handle ingredients, so I am discarding the unneeded plate.'],
    discard_extra_onion: ['当前这锅已经有足够的食材，我会丢弃手里多出的洋葱，空出手继续配合。', 'The current pot has enough ingredients. I am discarding the extra onion to free my hand.'],
    discard_for_handoff: ['右侧厨师已经装好汤，我会先丢弃手里的洋葱，空出手来接汤出餐。', 'The chef on the right is holding plated soup. I will discard my onion to free my hand and collect the soup.'],
    clear_for_handoff: ['右侧厨师手里的汤需要交回，但两个工作台都已占满；我会先取走洋葱，腾出交接位置。', 'The chef on the right needs to return a soup, but both shared counters are full. I will remove an onion to create handoff space.'],
    handoff_onion: ['我正把洋葱放到共享工作台，供右侧厨师取走入锅。', 'I am handing an onion over through a shared counter so the chef on the right can add it to the pot.'],
    collect_soup: ['共享工作台上有已装盘的汤，我会优先取走并送去出餐。', 'There is plated soup on a shared counter. I will collect it first, then serve it.'],
    clear_plate: ['共享工作台上的盘子占用了交接位置；我会取走，再按当前烹饪状态使用或丢弃。', 'A plate is occupying a shared counter. I am collecting it, then will use or discard it according to the pot state.'],
    clear_extra_onion: ['锅里已有三份食材，工作台上的多余洋葱占用了交接位置；我会取走并丢弃。', 'The pot already has three ingredients. I am removing an extra onion from a shared counter to clear handoff space.'],
    get_onion: ['当前这锅仍缺少食材，我正去取洋葱。计数包含锅内、共享工作台和右侧厨师手里的洋葱。', 'The current pot still needs ingredients, so I am fetching an onion. This count includes onions in the pot, on shared counters, and held by the right-side chef.'],
    wait_pickup: ['交接位置已占用，我会靠近回传工作台等待对方取走物品。', 'The shared counters are occupied. I am moving near the return counter and waiting for the other chef to collect an item.'],
    wait_soup: ['当前这锅所需的洋葱已经备齐，我会靠近回传工作台，等待装好的汤。', 'The onions for the current pot are already supplied. I am moving near the return counter to wait for plated soup.'],
    wait_soup_handoff: ['右侧厨师已经装好汤，我会靠近回传工作台，准备取汤出餐。', 'The chef on the right has plated the soup. I am moving near the return counter, ready to collect and serve it.'],
    handoff_soup: ['我正把装好的汤放到共享工作台，交给左侧厨师出餐。', 'I am returning the plated soup through a shared counter so the chef on the left can serve it.'],
    load_pot: ['我手里拿着洋葱，锅内还未满三份；我正去把它加入锅中。', 'I am holding an onion and the pot has fewer than three ingredients, so I am taking it to the pot.'],
    plate_soup: ['汤已经煮好，而且我手里有盘子；我正去装汤。', 'The soup is ready and I am holding a plate, so I am going to plate it.'],
    wait_cooking: ['汤还在烹饪。我会先靠近接下来要操作的位置，等汤煮好后再装盘。', 'The soup is still cooking. I am moving near the next station and will plate it once it is ready.'],
    get_counter_plate: ['汤已煮好，共享工作台上有盘子；我会先取这个盘子，同时腾出交接位置。', 'The soup is ready and a plate is on a shared counter. I will collect that plate and free the counter.'],
    get_plate: ['汤已经煮好，我现在空手，需要先取盘子才能装汤。', 'The soup is ready and my hands are empty. I need to fetch a plate before I can plate it.'],
    collect_onion: ['共享工作台上有洋葱，锅内仍需要食材；我正去取洋葱入锅。', 'An onion is on a shared counter and the pot still needs ingredients. I am collecting it to load the pot.'],
    wait_onion: ['当前没有可取的洋葱，我会靠近上方工作台等待左侧厨师交接食材。', 'No onion is available yet. I am moving near the upper counter to wait for the left-side chef to hand over ingredients.'],
    wait_space: ['共享工作台已满，需要先腾出至少一个位置，才能继续交接；我会保持等待，不会暗中移除物品。', 'Both shared counters are full. At least one counter must be cleared before the handoff can continue. I will wait and keep every item in place.'],
    finished: ['本轮已经结束，不再执行新的任务动作。', 'This round has ended; no further task actions will be taken.'],
  };

  const ACTION_TEXT = {
    UP: ['向上', 'move up'], DOWN: ['向下', 'move down'], LEFT: ['向左', 'move left'], RIGHT: ['向右', 'move right'],
    INTERACT: ['交互', 'interact'], WAIT: ['等待', 'wait'],
  };

  function explain(state, kind, language = 'zh') {
    if (!['why', 'waiting', 'counterfactual'].includes(kind)) throw new Error('Unknown explanation kind');
    const en = language === 'en';
    const decision = decide(state, 'ai');
    const titles = { why: ['为什么选择这个动作', 'Why this action?'], waiting: ['你在等什么', 'What are you waiting for?'], counterfactual: ['如果我等待会怎样', 'What if I wait?'] };
    const answer = { title: titles[kind][en ? 1 : 0], text: '', frame: state.turn, kind, decision: clone(decision) };
    const prefix = en ? `Selected frame: step ${state.turn}. ` : `所选画面：第 ${state.turn} 步。`;
    const reason = RULE_TEXT[decision.rule][en ? 1 : 0];
    if (kind === 'why') {
      answer.text = prefix + (en ? `My next action is to ${ACTION_TEXT[decision.action][1]}. ` : `我下一步会${ACTION_TEXT[decision.action][0]}。`) + reason
        + (en ? ' This explanation comes from the program teammate’s actual decision rule.' : '这段解释来自程序队友实际采用的决策规则。');
      return answer;
    }
    if (kind === 'waiting') {
      const waiting = decision.facts.waitingFor;
      const conditions = {
        onion: ['等待左侧厨师把洋葱放到共享工作台。', 'I am waiting for the left-side chef to place an onion on a shared counter.'],
        space: ['等待任意共享工作台腾出一个位置。', 'I am waiting for an empty shared counter.'],
        counter_pickup: ['等待另一侧取走工作台上的物品，腾出交接位置。', 'I am waiting for the other side to collect an item and free a counter.'],
        cooking: [`等待烹饪完成，还需要 ${state.pot.remaining} 个联合步。`, `I am waiting for cooking to finish: ${state.pot.remaining} joint steps remain.`],
        soup_handoff: ['等待右侧厨师装盘并交回汤。', 'I am waiting for the right-side chef to plate and return the soup.'],
        pot_loading: ['等待已备好的洋葱全部入锅，之后还需要烹饪和装盘。', 'I am waiting for the supplied onions to be loaded; cooking and plating come afterwards.'],
      };
      answer.text = prefix + (waiting ? conditions[waiting][en ? 1 : 0] : en ? 'I am not waiting at this frame. ' : '这个画面下我没有在等待。')
        + (waiting && decision.action !== 'WAIT' ? (en ? ' While waiting, I am moving into position. ' : '等待条件满足前，我会先移动到操作位置。') : '')
        + (!waiting ? reason : '')
        + (en ? ' Reading this answer does not advance the kitchen.' : '阅读这段回答不会推进厨房时间。');
      return answer;
    }
    let forecastState = snapshot(state);
    const trajectory = [];
    for (let i = 0; i < 3 && !forecastState.done; i++) {
      const result = step(forecastState, 'WAIT');
      trajectory.push({ turn: forecastState.turn, action: result.actions.ai, rule: result.decisions.ai.rule, events: clone(result.events) });
      forecastState = result.state;
    }
    answer.forecast = { steps: trajectory.length, state: snapshot(forecastState), trajectory };
    const afterAi = actorById(forecastState, 'ai');
    const itemNames = { onion: ['洋葱', 'an onion'], plate: ['盘子', 'a plate'], soup: ['汤', 'soup'] };
    const holding = afterAi.holding ? itemNames[afterAi.holding][en ? 1 : 0] : en ? 'nothing' : '空手';
    const actionList = trajectory.map(frame => ACTION_TEXT[frame.action][en ? 1 : 0]).join(en ? ' → ' : ' → ');
    answer.text = prefix + (en
      ? `Assumption: you wait for the next ${trajectory.length} joint steps, while I keep using the same program. ${trajectory.length ? `My actions would be: ${actionList}. ` : ''}At step ${forecastState.turn}, we would have completed ${forecastState.orders} / 2 orders; I would be at row ${afterAi.position[0] + 1}, column ${afterAi.position[1] + 1}, holding ${holding}. `
      : `假设你接下来连续等待 ${trajectory.length} 个联合步，我继续按同一程序行动。${trajectory.length ? `我的动作依次为：${actionList}。` : ''}到第 ${forecastState.turn} 步，共完成 ${forecastState.orders} / 2 份订单；我位于第 ${afterAi.position[0] + 1} 行、第 ${afterAi.position[1] + 1} 列，${afterAi.holding ? '手持' + holding : holding}。`)
      + (forecastState.done ? (en ? `The round would end: ${forecastState.reason === 'success' ? 'goal completed' : 'step limit reached'}. ` : `本轮将结束：${forecastState.reason === 'success' ? '完成目标' : '达到步数上限'}。`) : '')
      + (en ? 'This is an isolated simulation, not a prediction of your later choices. Your live game is unchanged.' : '这是隔离模拟，不预测你之后的选择，也不会改变当前游戏。');
    return answer;
  }

  function eventText(event, language = 'zh') {
    const en = language === 'en';
    const name = event.actor === 'human' ? (en ? 'You' : '玩家') : (en ? 'Teammate' : '队友');
    const names = { onion: ['洋葱', 'an onion'], plate: ['盘子', 'a plate'], soup: ['汤', 'soup'] };
    const item = names[event.item] ? names[event.item][en ? 1 : 0] : '';
    const action = ACTION_TEXT[event.action] || ['', ''];
    const simple = {
      move: en ? `${name}: ${action[1]}.` : `${name}${action[0]}移动。`,
      blocked: en ? `${name} turned ${event.action ? ACTION_TEXT[event.action][1].replace('move ', '') : ''}; movement was blocked.` : `${name}调整朝向，位置未改变。`,
      wait: en ? `${name} waited.` : `${name}等待。`,
      pickup: en ? `${name} collected ${item} from a shared counter.` : `${name}从共享工作台取走${item}。`,
      drop: en ? `${name} placed ${item} on a shared counter.` : `${name}把${item}放到共享工作台。`,
      take_source: en ? `${name} took ${item}.` : `${name}取出${item}。`,
      discard: en ? `${name} discarded ${item}.` : `${name}丢弃${item}。`,
      serve: en ? `${name} served one soup.` : `${name}完成一份汤的出餐。`,
      load: en ? `${name} added an onion to the pot.` : `${name}向锅内加入一份洋葱。`,
      plate: en ? `${name} plated the cooked soup.` : `${name}将煮好的汤装盘。`,
      cooking_started: en ? 'Cooking started: four subsequent joint steps remain.' : '开始烹饪：还需要后续四个联合步。',
      soup_ready: en ? 'The soup is ready to plate.' : '汤已煮好，可以装盘。',
      conflict: en ? `${name}: simultaneous interaction lost the turn-priority tie-break.` : `${name}的同时交互按本步优先顺序未能执行。`,
      success: en ? 'Two orders completed. Goal achieved.' : '两份订单完成，达成目标。',
      timeout: en ? 'The 120-step limit was reached.' : '达到 120 步上限。',
    };
    if (event.type === 'invalid_interaction') {
      const reasons = {
        counter_occupied: ['工作台已有物品', 'the counter is occupied'], counter_empty: ['工作台为空', 'the counter is empty'],
        hands_full: ['手中已有物品', 'hands are full'], hands_empty: ['手中没有物品', 'hands are empty'],
        pot_cooking: ['汤仍在烹饪', 'the soup is still cooking'], plate_needed: ['装汤需要手持盘子', 'a held plate is required'],
        pot_full: ['锅内已有三份食材', 'the pot is full'], onion_needed: ['入锅需要手持洋葱', 'a held onion is required'],
        soup_needed: ['出餐需要手持汤', 'a held soup is required'], no_station: ['当前朝向没有可交互设施', 'no station is directly ahead'],
      };
      const reason = (reasons[event.reason] || reasons.no_station)[en ? 1 : 0];
      return en ? `${name}: no interaction (${reason}); one step used.` : `${name}未完成交互（${reason}），消耗一步。`;
    }
    return simple[event.type] || (en ? 'Kitchen state updated.' : '厨房状态已更新。');
  }

  return Object.freeze({ MAP, ACTIONS, reset, snapshot, restore, decide, step, explain, eventText });
}));
