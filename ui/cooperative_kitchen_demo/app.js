/* Local, non-research playground. Only confirmed joint steps enter history. */
(function () {
  'use strict';
  const E = window.KitchenEngine;
  const $ = id => document.getElementById(id);
  const STORAGE_KEY = 'policylens-kitchen-demo-v1';
  const COPY = {
    zh: {
      app_title: '协作厨房', demo_badge: '玩法演示 · 程序队友', language_label: '切换语言',
      instructions_label: '说明', play_label: '合作试玩', result_label: '结果',
      overview: '任务说明', intro_title: '一起完成两份洋葱汤',
      intro_text: '你和队友分别在厨房两侧工作。通过中间的工作台交接食材和成汤，共同完成订单。',
      recipe: '3 份洋葱 → 烹饪 4 步 → 拿盘装汤 → 交接出餐',
      instruction_movement: '方向键或 WASD 移动并调整朝向；面向设施按 E 交互，空格等待。',
      instruction_turns: '每次移动、交互或等待推进一步，撞墙和无效交互也消耗一步。阅读、解释和回放不消耗步数。',
      instruction_handoff: '每人只能拿一件物品。工作台各放一件物品，两侧都可以取放；拿错的物品可以放入垃圾桶。',
      instruction_goal: '120 步内完成两份汤。可以重新开始、交换分工，或先观看自动演示。',
      prototype_note: '这是一份合作玩法原型。队友使用可检查的程序规则，不是经过训练的神经策略。',
      start: '开始合作', resume: '继续试玩', map_title: '协作厨房', map_dimensions: '9 × 7 厨房',
      orders: '完成订单', steps: '已用步数', pot: '锅的状态', role: '你的分工',
      supply_role: '供料与出餐', cook_role: '烹饪与装盘',
      human: '玩家', ai: '程序队友', human_inventory: '你手中的物品', ai_inventory: '队友手中的物品',
      empty: '空手', onion: '洋葱', plate: '盘子', soup: '洋葱汤',
      controls: '操作控制', UP: '上', DOWN: '下', LEFT: '左', RIGHT: '右', INTERACT: '交互', WAIT: '等待',
      keyboard: '方向键 / WASD 移动 · E 交互 · 空格等待',
      interaction_hint: '先面向设施，再按 E。方向键撞到设施时仍会调整朝向。',
      teammate: '队友状态', next_action: '下一步', target: '目标',
      explanations: '行为解释', explanation_hint: '查看程序队友在所选状态下的决策。打开解释会暂停自动演示。',
      why: '为什么选择这个动作', waiting: '你在等什么', counterfactual: '如果我等待会怎样',
      explanation_source: '依据：程序决策记录与厨房模拟', close_explanation: '收起解释',
      replay: '行动回放', live: '回到当前', previous: '上一帧', next: '下一帧', history_frame: '历史步数',
      replay_hint: '回放时停止操作；回到当前后可继续。', replay_mode: '历史回放',
      restart: '重新开始', swap_role: '交换分工', auto_demo: '自动演示', pause_auto: '暂停演示',
      show_instructions: '查看说明', local_save: '试玩进度仅保存在本浏览器，不进入研究数据库。',
      playing: '合作进行中', automatic: '自动演示中', ended: '试玩结束',
      initial_feedback: '先移动到设施旁，面向它按 E 交互。也可以观看自动演示。',
      step: '步', decision_frame: '决策状态：第 {n} 步后',
      won: '两份汤已完成', timeout: '已达到步数上限', result_text: '合作完成 {orders} / 2 份汤，共使用 {turn} 步。',
      ready: '已煮熟，可装盘', cooking: '烹饪中 · 还需 {n} 步', filling: '食材 {n} / 3',
      restored: '已恢复本机试玩进度；自动演示保持暂停。',
      invalid_save: '旧试玩记录无法恢复，已打开一局新的演示。',
      storage_error: '浏览器暂时无法保存进度，本次仍可继续试玩；刷新后可能无法恢复。',
      action_error: '这一步未能完成，请重新开始试玩。',
      I: '洋葱供应', D: '盘子供应', P: '汤锅', S: '出餐口', X: '垃圾桶',
      upper_counter: '上方交接台', lower_counter: '下方交接台', floor: '地板',
      movement_event: '{actor}：{action}', no_event: '本步已完成。',
      board_label: '协作厨房地图，蓝色为玩家，橙色为程序队友',
      holding: '手持', automatic_note: '双方按程序执行。暂停后可以接管玩家。',
      handoff: '共享工作台', legend_onion: '食材', legend_plate: '盘子', legend_pot: '汤锅', legend_serve: '出餐', legend_trash: '垃圾桶',
    },
    en: {
      app_title: 'Cooperative Kitchen', demo_badge: 'Playable demo · Programmed teammate', language_label: 'Switch language',
      instructions_label: 'Instructions', play_label: 'Cooperative play', result_label: 'Results',
      overview: 'Task instructions', intro_title: 'Make two onion soups together',
      intro_text: 'You and your teammate work on opposite sides of the kitchen. Exchange ingredients and soup across the central counters to complete orders together.',
      recipe: '3 onions → Cook for 4 steps → Plate the soup → Hand over and serve',
      instruction_movement: 'Use arrows or WASD to move and face a direction. Press E to interact with a station in front of you, or Space to wait.',
      instruction_turns: 'Each move, interaction or wait advances one step. Wall bumps and invalid interactions also use a step. Reading, explanations and replay do not.',
      instruction_handoff: 'Each chef carries one item, and each shared counter holds one item. Either side can take or place items. Use a bin to discard unwanted items.',
      instruction_goal: 'Complete two soups within 120 steps. Restart, swap roles, or watch the automatic demonstration first.',
      prototype_note: 'This is a cooperative gameplay prototype. The teammate follows inspectable program rules, not a trained neural policy.',
      start: 'Start cooking', resume: 'Resume play', map_title: 'Cooperative kitchen', map_dimensions: '9 × 7 KITCHEN',
      orders: 'Completed orders', steps: 'Steps used', pot: 'Pot status', role: 'Your role',
      supply_role: 'Supply & serve', cook_role: 'Cook & plate',
      human: 'Player', ai: 'Programmed teammate', human_inventory: 'Your inventory', ai_inventory: 'Teammate inventory',
      empty: 'Empty hands', onion: 'Onion', plate: 'Plate', soup: 'Onion soup',
      controls: 'Controls', UP: 'Up', DOWN: 'Down', LEFT: 'Left', RIGHT: 'Right', INTERACT: 'Interact', WAIT: 'Wait',
      keyboard: 'Arrows / WASD to move · E to interact · Space to wait',
      interaction_hint: 'Face a station, then press E. Moving toward a blocked station still changes your facing direction.',
      teammate: 'Teammate status', next_action: 'Next action', target: 'Target',
      explanations: 'Behavior explanation', explanation_hint: 'Inspect the program decision at the selected state. Opening an explanation pauses the demonstration.',
      why: 'Why choose this action?', waiting: 'What are you waiting for?', counterfactual: 'What if I wait?',
      explanation_source: 'Sources: program decision records and kitchen simulation', close_explanation: 'Close explanation',
      replay: 'Action replay', live: 'Back to live', previous: 'Previous frame', next: 'Next frame', history_frame: 'Historical step',
      replay_hint: 'Controls pause during replay. Return to live to continue.', replay_mode: 'Historical replay',
      restart: 'Restart', swap_role: 'Swap roles', auto_demo: 'Automatic demo', pause_auto: 'Pause demo',
      show_instructions: 'Instructions', local_save: 'Progress stays in this browser and is not written to a research database.',
      playing: 'Cooperative play', automatic: 'Automatic demonstration', ended: 'Demo complete',
      initial_feedback: 'Move next to a station, face it, then press E. You can also watch the automatic demonstration.',
      step: 'Step', decision_frame: 'Decision state: after step {n}',
      won: 'Two soups completed', timeout: 'Step limit reached', result_text: 'Together you completed {orders} / 2 soups in {turn} steps.',
      ready: 'Ready to plate', cooking: 'Cooking · {n} steps left', filling: 'Ingredients {n} / 3',
      restored: 'Local progress restored. The automatic demonstration remains paused.',
      invalid_save: 'The previous demo could not be restored. A fresh game is ready.',
      storage_error: 'This browser could not save progress. You can keep playing, but refreshing may lose this session.',
      action_error: 'This step could not be completed. Please restart the demo.',
      I: 'Onion supply', D: 'Plate supply', P: 'Soup pot', S: 'Serving hatch', X: 'Bin',
      upper_counter: 'Upper counter', lower_counter: 'Lower counter', floor: 'Floor',
      movement_event: '{actor}: {action}', no_event: 'Step confirmed.',
      board_label: 'Cooperative kitchen map. Blue is the player; orange is the programmed teammate.',
      holding: 'Holding', automatic_note: 'Both chefs follow programs. Pause to take control of the player.',
      handoff: 'Shared counters', legend_onion: 'Ingredients', legend_plate: 'Plates', legend_pot: 'Pot', legend_serve: 'Serve', legend_trash: 'Bin',
    },
  };
  let state = E.reset('supply');
  let frames = [E.snapshot(state)], frameEvents = [[]];
  let language = 'zh', started = false, helpOpen = false, replayIndex = null;
  let explanationKind = null, noticeKey = null;
  let automatic = false, autoTimer = null, animationId = null, animating = false;
  const t = (key, values = {}) => Object.entries(values).reduce((value, [name, replacement]) => value.replaceAll(`{${name}}`, String(replacement)), COPY[language][key] || key);
  const currentView = () => replayIndex === null ? state : frames[replayIndex];
  const human = view => view.actors.find(actor => actor.id === 'human');
  const ai = view => view.actors.find(actor => actor.id === 'ai');

  function save() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({version: 1, state: E.snapshot(state), frames, frameEvents, language, started}));
    } catch (_) {
      noticeKey = 'storage_error';
      $('notice').hidden = false;
      $('notice').textContent = t(noticeKey);
    }
  }

  function restore() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return;
      const saved = JSON.parse(raw);
      if (saved.version !== 1 || !Array.isArray(saved.frames) || saved.frames.length < 1 || saved.frames.length > 121) throw new Error('Invalid demo history');
      const restored = E.restore(saved.state);
      const history = saved.frames.map(frame => E.restore(frame));
      if (history.length !== restored.turn + 1 || history.some((frame, index) => frame.turn !== index || frame.preset !== restored.preset) || JSON.stringify(history.at(-1)) !== JSON.stringify(restored)) throw new Error('Inconsistent demo history');
      state = restored;
      frames = history;
      frameEvents = Array.isArray(saved.frameEvents) && saved.frameEvents.length === frames.length ? saved.frameEvents : frames.map(() => []);
      language = saved.language === 'en' ? 'en' : 'zh';
      started = saved.started === true;
      noticeKey = 'restored';
    } catch (_) {
      noticeKey = 'invalid_save';
    }
  }

  function translate() {
    document.documentElement.lang = language;
    document.title = 'PolicyLens · ' + t('app_title');
    document.querySelectorAll('[data-i18n]').forEach(node => {node.textContent = node.dataset[language] || t(node.dataset.i18n);});
    document.querySelectorAll('[data-i18n-aria]').forEach(node => node.setAttribute('aria-label', t(node.dataset.i18nAria)));
    document.querySelectorAll('[data-action]').forEach(node => node.setAttribute('aria-label', t(node.dataset.action)));
    $('language').textContent = language === 'zh' ? 'EN' : '中文';
    $('language').setAttribute('aria-label', t('language_label'));
    $('board').setAttribute('aria-label', t('board_label'));
    $('timeline').setAttribute('aria-label', t('history_frame'));
    $('previous-frame').setAttribute('aria-label', t('previous'));
    $('next-frame').setAttribute('aria-label', t('next'));
    $('close-explanation').setAttribute('aria-label', t('close_explanation'));
    $('start').textContent = t(started ? 'resume' : 'start');
  }

  function stationName(position) {
    if (!position) return '';
    const [row, col] = position;
    const tile = E.MAP[row]?.[col];
    return tile === 'C' ? t(row === 2 ? 'upper_counter' : 'lower_counter') : t(tile && 'IDPSX'.includes(tile) ? tile : 'floor');
  }

  function canAct() {
    return started && !helpOpen && !state.done && replayIndex === null && !animating && !automatic;
  }

  function controls() {
    document.querySelectorAll('[data-action]').forEach(button => {button.disabled = !canAct();});
    $('auto-demo').disabled = !started || helpOpen || state.done;
    $('auto-demo').textContent = t(automatic ? 'pause_auto' : 'auto_demo');
    $('auto-demo').setAttribute('aria-pressed', String(automatic));
    $('previous-frame').disabled = (replayIndex ?? frames.length - 1) <= 0;
    $('next-frame').disabled = (replayIndex ?? frames.length - 1) >= frames.length - 1;
    $('live').disabled = replayIndex === null;
  }

  function stopAnimation() {
    if (animationId !== null) cancelAnimationFrame(animationId);
    animationId = null;
    animating = false;
  }

  function stopAuto() {
    automatic = false;
    if (autoTimer !== null) clearTimeout(autoTimer);
    autoTimer = null;
  }

  function draw(before = null, progress = 1) {
    if ($('play').hidden) return;
    KitchenRenderer.draw($('board'), currentView(), {before, progress, language, selectedActor: 'ai'});
  }

  function renderExplanation() {
    $('explanation').hidden = explanationKind === null;
    document.querySelectorAll('[data-explain]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.explain === explanationKind)));
    if (!explanationKind) return;
    const result = E.explain(currentView(), explanationKind, language);
    $('explanation-title').textContent = result.title;
    $('explanation-text').textContent = result.text;
    $('explanation-frame').textContent = t('decision_frame', {n: result.frame});
    $('explanation').dataset.frame = String(result.frame);
    $('explanation').dataset.kind = explanationKind;
  }

  function render({paint = true} = {}) {
    translate();
    $('instructions').hidden = started && !helpOpen;
    $('play').hidden = !started || helpOpen;
    $('notice').hidden = !noticeKey;
    $('notice').textContent = noticeKey ? t(noticeKey) : '';
    $('phase-label').textContent = t(!started || helpOpen ? 'instructions_label' : state.done ? 'ended' : automatic ? 'automatic' : 'playing');
    const phase = !started || helpOpen ? 'instructions' : state.done ? 'result' : 'play';
    const order = ['instructions', 'play', 'result'];
    document.querySelectorAll('[data-phase]').forEach(node => {
      node.classList.toggle('active', node.dataset.phase === phase);
      node.classList.toggle('done', order.indexOf(node.dataset.phase) < order.indexOf(phase));
      if (node.dataset.phase === phase) node.setAttribute('aria-current', 'step'); else node.removeAttribute('aria-current');
    });
    const view = currentView();
    document.querySelectorAll('.diagram-side .chef-token').forEach((token, index) => {
      const isHuman = human(view).side === (index === 0 ? 'left' : 'right');
      token.classList.toggle('human', isHuman);
      token.classList.toggle('ai', !isHuman);
      token.textContent = isHuman ? '1' : '2';
    });
    $('orders-count').textContent = `${view.orders} / ${view.targetOrders}`;
    $('turn-count').textContent = `${view.turn} / ${view.maxSteps}`;
    $('pot-status').textContent = view.pot.ready ? t('ready') : view.pot.remaining > 0 ? t('cooking', {n: view.pot.remaining}) : t('filling', {n: view.pot.ingredients});
    $('role-label').textContent = t(human(view).side === 'left' ? 'supply_role' : 'cook_role');
    $('human-held').textContent = t(human(view).holding || 'empty');
    $('ai-held').textContent = t(ai(view).holding || 'empty');
    const decision = E.decide(view, 'ai');
    $('ai-status').textContent = `${t('next_action')}：${t(decision.action)}` + (decision.target ? ` · ${t('target')}：${stationName(decision.target)}` : '');
    const events = frameEvents[replayIndex ?? frames.length - 1] || [];
    $('feedback').textContent = events.length ? events.map(event => typeof E.eventText === 'function' ? E.eventText(event, language) : t('movement_event', {actor: t(event.actor === 'human' ? 'human' : 'ai'), action: t(event.action || 'WAIT')})).filter(Boolean).join(' · ') : t('initial_feedback');
    $('timeline').max = String(frames.length - 1);
    $('timeline').value = String(replayIndex ?? frames.length - 1);
    $('timeline-label').textContent = `${view.turn} / ${state.turn}`;
    $('replay-badge').hidden = replayIndex === null;
    $('replay-badge').textContent = `${t('replay_mode')} · ${t('step')} ${view.turn}`;
    $('result').hidden = !view.done;
    $('result').replaceChildren();
    if (view.done) {
      const title = document.createElement('strong'), detail = document.createElement('p');
      title.textContent = t(view.orders >= view.targetOrders ? 'won' : 'timeout');
      detail.textContent = t('result_text', {orders: view.orders, turn: view.turn});
      $('result').append(title, detail);
    }
    renderExplanation();
    controls();
    if (paint) {stopAnimation(); draw(); controls();}
  }

  function animate(before) {
    stopAnimation();
    if (matchMedia('(prefers-reduced-motion: reduce)').matches) {draw(); controls(); return;}
    animating = true;
    controls();
    const began = performance.now();
    const tick = now => {
      const fraction = Math.min(1, (now - began) / 120);
      draw(before, fraction);
      if (fraction < 1) animationId = requestAnimationFrame(tick);
      else {animationId = null; animating = false; controls();}
    };
    animationId = requestAnimationFrame(tick);
  }

  function advance(action, auto = false) {
    if (!started || helpOpen || state.done || replayIndex !== null || animating) return;
    const before = E.snapshot(state);
    try {
      const result = E.step(state, action, {auto});
      state = result.state;
      frames.push(E.snapshot(state));
      frameEvents.push(result.events);
      explanationKind = null;
      noticeKey = null;
      if (state.done) stopAuto();
      save();
      render({paint: false});
      animate(before);
    } catch (error) {
      stopAuto(); noticeKey = 'action_error'; render();
      console.error(error);
    }
  }

  function scheduleAuto() {
    if (!automatic) return;
    autoTimer = setTimeout(() => {
      autoTimer = null;
      if (!automatic) return;
      advance('WAIT', true);
      if (automatic) scheduleAuto();
    }, 350);
  }

  function startPlay() {
    stopAuto(); stopAnimation();
    started = true; helpOpen = false; noticeKey = null;
    save(); render(); $('board').focus({preventScroll: true});
  }

  function reset(preset) {
    stopAuto(); stopAnimation();
    state = E.reset(preset);
    frames = [E.snapshot(state)]; frameEvents = [[]]; replayIndex = null;
    explanationKind = null; noticeKey = null; helpOpen = false; started = true;
    save(); render();
  }

  function replay(index) {
    stopAuto(); stopAnimation();
    replayIndex = index === null ? null : Math.max(0, Math.min(frames.length - 1, index));
    render();
  }

  $('start').onclick = startPlay;
  $('restart').onclick = () => reset(state.preset);
  $('swap-role').onclick = () => reset(state.preset === 'supply' ? 'cook' : 'supply');
  $('show-instructions').onclick = () => {stopAuto(); stopAnimation(); helpOpen = true; render();};
  $('auto-demo').onclick = () => {
    if (automatic) {stopAuto(); stopAnimation(); render(); return;}
    if (state.done) return;
    stopAnimation(); replayIndex = null; explanationKind = null;
    automatic = true; render(); scheduleAuto();
  };
  $('language').onclick = () => {stopAuto(); stopAnimation(); language = language === 'zh' ? 'en' : 'zh'; save(); render();};
  document.querySelectorAll('[data-action]').forEach(button => {button.onclick = () => {if (canAct()) advance(button.dataset.action);};});
  document.querySelectorAll('[data-explain]').forEach(button => {button.onclick = () => {stopAuto(); stopAnimation(); explanationKind = button.dataset.explain; render();};});
  $('close-explanation').onclick = () => {explanationKind = null; render();};
  $('timeline').oninput = () => replay(Number($('timeline').value));
  $('previous-frame').onclick = () => replay((replayIndex ?? frames.length - 1) - 1);
  $('next-frame').onclick = () => replay((replayIndex ?? frames.length - 1) + 1);
  $('live').onclick = () => replay(null);
  window.addEventListener('keydown', event => {
    if (event.repeat || event.ctrlKey || event.metaKey || event.altKey || ['INPUT', 'TEXTAREA', 'SELECT'].includes(event.target.tagName)) return;
    const key = event.key.length === 1 ? event.key.toLowerCase() : event.key;
    if (key === ' ' && event.target.closest('button, a[href], [role="button"]')) return;
    const action = {ArrowUp: 'UP', ArrowDown: 'DOWN', ArrowLeft: 'LEFT', ArrowRight: 'RIGHT', w: 'UP', s: 'DOWN', a: 'LEFT', d: 'RIGHT', e: 'INTERACT', ' ': 'WAIT'}[key];
    if (!action || !started || helpOpen) return;
    event.preventDefault();
    if (automatic) {stopAuto(); stopAnimation(); render(); return;}
    if (canAct()) advance(action);
  });
  window.addEventListener('resize', () => {stopAnimation(); draw(); controls();});
  document.addEventListener('visibilitychange', () => {if (document.hidden && automatic) {stopAuto(); stopAnimation(); render();}});
  window.KitchenDemo = Object.freeze({
    getState: () => E.snapshot(state),
    getHistory: () => frames.map(frame => E.snapshot(frame)),
    getUI: () => ({language, replayIndex, autoRunning: automatic, started, helpOpen, explanationKind}),
  });
  restore(); render();
})();
