/* The server owns experiment state, permissions, decisions and scoring. */
(function () {
  'use strict';
  const $ = id => document.getElementById(id);
  const KEYS = {pending: 'policylens-kitchen-study-pending-v1', language: 'policylens-kitchen-study-language-v1', survey: 'policylens-kitchen-study-survey-v1', sync: 'policylens-kitchen-study-sync-v1'};
  const PARTICIPANT_ID = /^[A-Za-z][A-Za-z0-9_-]{2,31}$/;
  const AUTHORITY_CHECK_MS = 5000;
  let language = 'zh', status = {}, view = null, busy = false, pending = null, notice = '', helpOpen = false;
  let participantIdError = null;
  let replayEpisode = null, replayFrame = null, historyRequest = 0, animation = null, animating = false;
  let answer = null, questionJob = null, pollTimer = null, questionEpoch = 0;
  let autoRunning = false, autoTimer = null, autoEpoch = 0;
  let surveyDraft = {}, surveyRun = null, surveySignature = '', surveyTimer = null, surveyDirty = false;
  let syncRefresh = null, syncRefreshQueued = false;
  const syncWatermarks = new Map();
  const syncChannel = typeof window.BroadcastChannel === 'function' ? new BroadcastChannel('policylens-kitchen-study-sync-v1') : null;
  const histories = new Map(), historyPermissions = new Map(), shownAnswers = new Set();
  const tr = (zh, en) => language === 'en' ? en : zh;
  const localized = value => typeof value === 'string' ? value : value?.[language] || value?.zh || value?.en || '';
  const clone = value => JSON.parse(JSON.stringify(value));
  const newId = () => crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(36).slice(2)}`;
  const run = () => view?.run;
  const currentEpisode = () => run()?.episode_id || view?.state?.episode_id || view?.episode_id || view?.episodes?.find(e => e.index === run()?.episode_index)?.id || null;
  const selectedEpisode = () => replayEpisode || currentEpisode();
  const phase = () => run()?.phase || 'lobby';
  const isLive = () => replayEpisode === null && replayFrame === null;
  const displayedState = () => isLive() ? view?.state : histories.get(selectedEpisode())?.[replayFrame ?? 0] || view?.state;
  const frameNumber = () => displayedState()?.turn ?? 0;
  const phaseText = value => ({consent: tr('参与说明', 'Participant information'), instructions: tr('任务说明', 'Instructions'), practice: tr('共同练习', 'Practice'), task1: 'Task 1', task2: 'Task 2', questionnaire: tr('问卷', 'Questionnaire'), complete: tr('已完成', 'Complete'), freeplay: tr('自由试玩', 'Free play')})[value] || value;
  const itemText = value => ({onion: tr('洋葱', 'Onion'), plate: tr('盘子', 'Plate'), soup: tr('汤', 'Soup')})[typeof value === 'object' ? value?.type : value] || tr('空手', 'Empty');
  const actionText = value => ({UP: tr('向上', 'Move up'), DOWN: tr('向下', 'Move down'), LEFT: tr('向左', 'Move left'), RIGHT: tr('向右', 'Move right'), INTERACT: tr('交互', 'Interact'), WAIT: tr('等待', 'Wait')})[value] || value;
  const roundText = episode => episode ? `${phaseText(episode.phase)}${['task1', 'task2'].includes(episode.phase) ? ` · ${((episode.index - 1) % 3) + 1} / 3` : ''}` : phaseText(phase());
  const currentEpisodeRecord = () => view?.episodes?.find(e => e.id === selectedEpisode());
  const canAsk = () => Boolean(view?.can_ask && historyPermissions.get(selectedEpisode()) !== false && !helpOpen && displayedState() && (!currentEpisodeRecord() || currentEpisodeRecord().phase === 'task1' || run()?.mode === 'freeplay'));
  const canAct = () => Boolean(view?.can_act && !busy && !pending && !animating && !helpOpen && isLive());
  const canAuto = () => Boolean(view?.can_auto && run()?.mode === 'freeplay' && isLive() && !helpOpen && !answer && !questionJob && !document.hidden);
  const enrollmentEnabled = () => status.enrollment?.enabled === true;

  function storageSet(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); return true; } catch (_) { return false; } }
  function storageGet(key) { try { return JSON.parse(localStorage.getItem(key) || 'null'); } catch (_) { return null; } }
  function storageDelete(key) { try { localStorage.removeItem(key); } catch (_) {} }
  const phaseRank = value => ({lobby: 0, consent: 1, instructions: 2, practice: 3, task1: 4, task2: 5, questionnaire: 6, survey: 6, complete: 7, completed: 7, technical_retry_closed: 8})[value] ?? 0;
  function validSync(signal) { return signal?.schema === 1 && typeof signal.run_id === 'string' && Number.isInteger(signal.version) && typeof signal.phase === 'string'; }
  function compareSync(left, right) { if (!validSync(left) || !validSync(right) || left.run_id !== right.run_id) return 0; return left.version - right.version || phaseRank(left.phase) - phaseRank(right.phase); }
  function rememberSync(signal) {
    if (!validSync(signal)) return null;
    const previous = syncWatermarks.get(signal.run_id);
    if (!previous || compareSync(signal, previous) > 0) syncWatermarks.set(signal.run_id, signal);
    return syncWatermarks.get(signal.run_id);
  }
  rememberSync(storageGet(KEYS.sync));
  language = storageGet(KEYS.language) === 'en' ? 'en' : 'zh';
  pending = storageGet(KEYS.pending);
  if (pending && (pending.schema !== 1 || !['/api/session', '/api/command', '/api/question'].includes(pending.path) || !pending.body?.operation_id)) { pending = null; storageDelete(KEYS.pending); }
  // A pending enrollment from an older contract must not create a new user ID.
  if (pending?.path === '/api/session' && pending.body.mode === 'pilot' && !PARTICIPANT_ID.test(pending.body.participant_id || '')) { pending = null; storageDelete(KEYS.pending); }
  if (pending?.path === '/api/session' && pending.body.mode === 'pilot') $('user-id').value = pending.body.participant_id;

  async function request(path, body) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 20000);
    try {
      const response = await fetch(path, {method: body === undefined ? 'GET' : 'POST', credentials: 'same-origin', cache: 'no-store', headers: body === undefined ? {} : {'Content-Type': 'application/json'}, body: body === undefined ? undefined : JSON.stringify(body), signal: controller.signal});
      let data;
      try { data = await response.json(); } catch (_) { throw new Error(tr('服务器响应无法读取。', 'The server response could not be read.')); }
      if (!response.ok) { const error = new Error(data.error || `HTTP ${response.status}`); error.status = response.status; error.code = data.code; throw error; }
      return data;
    } finally { clearTimeout(timeout); }
  }

  function stopAnimation() { if (animation !== null) cancelAnimationFrame(animation); animation = null; animating = false; }
  function stopAuto() { autoRunning = false; autoEpoch++; if (autoTimer !== null) clearTimeout(autoTimer); autoTimer = null; renderControls(); }
  function startAuto() {
    if (!canAuto() || !canAct()) return;
    autoRunning = true; const epoch = ++autoEpoch; renderControls();
    const tick = async () => {
      autoTimer = null;
      if (!autoRunning || epoch !== autoEpoch) return;
      if (!canAuto() || pending) { stopAuto(); return; }
      if (busy || animating) { autoTimer = setTimeout(tick, 100); return; }
      const confirmed = await command('auto_step', {}, {animateAction: true});
      if (!autoRunning || epoch !== autoEpoch) return;
      if (!confirmed || !canAuto()) { stopAuto(); return; }
      autoTimer = setTimeout(tick, 450);
    };
    autoTimer = setTimeout(tick, 0);
  }
  function draw(before = null, progress = 1) { if (!$('play').hidden && displayedState()) window.KitchenRenderer.draw($('board'), displayedState(), {before, progress, language}); }
  function animate(before) {
    stopAnimation();
    if (matchMedia('(prefers-reduced-motion: reduce)').matches || document.hidden) { draw(); renderControls(); return; }
    animating = true; renderControls(); const began = performance.now();
    const tick = now => { const progress = Math.min(1, (now - began) / 130); draw(before, progress); if (progress < 1) animation = requestAnimationFrame(tick); else { animation = null; animating = false; renderControls(); } };
    animation = requestAnimationFrame(tick);
  }

  function translate() {
    document.documentElement.lang = language; document.title = tr('PolicyLens · 协作厨房实验', 'PolicyLens · Cooperative Kitchen Study');
    document.querySelectorAll('[data-zh][data-en]').forEach(node => { node.textContent = node.dataset[language]; });
    document.querySelectorAll('[data-placeholder-zh]').forEach(node => { node.placeholder = node.dataset[language === 'en' ? 'placeholderEn' : 'placeholderZh']; });
    $('language').textContent = language === 'zh' ? 'EN' : '中文';
    $('language').setAttribute('aria-label', tr('Switch to English', '切换为中文'));
    $('previous-frame').setAttribute('aria-label', tr('上一帧', 'Previous frame')); $('next-frame').setAttribute('aria-label', tr('下一帧', 'Next frame'));
    $('timeline').setAttribute('aria-label', tr('回放步数', 'Replay turn'));
    document.querySelectorAll('[data-action]').forEach(node => node.setAttribute('aria-label', actionText(node.dataset.action)));
    const enabled = enrollmentEnabled(), internal = status.enrollment?.mode === 'internal_pilot';
    $('status-label').textContent = enabled ? internal ? tr('内部预实验 · 候选版本', 'Internal pilot · Candidate version') : tr('人机协作研究', 'Human–AI cooperation study') : status.policy_kind === 'program_baseline' ? tr('候选版本 · 程序队友', 'Candidate · Program teammate') : tr('候选版本 · 实验未开放', 'Candidate · Study unavailable');
    $('join-study').textContent = internal ? tr('开始预实验', 'Start pilot') : tr('开始实验', 'Start study');
    $('study-availability').textContent = enabled ? internal ? tr('内部预实验已开放，用于检查任务与流程。当前候选版本尚未通过正式实验验收。', 'The internal pilot is open to check the task and procedure. This candidate has not passed formal study acceptance.') : tr('请输入用户 ID 开始。', 'Enter a user ID to begin.') : tr('当前暂未开放参与者实验。可以使用自由试玩查看环境。', 'Participant sessions are currently unavailable; free play is available to inspect the environment.');
    renderParticipantIdError();
    const limit = status.protocol?.max_steps || status.max_steps || view?.state?.maxSteps || 180;
    const target = status.protocol?.target_orders || status.target_orders || view?.state?.targetOrders || 2;
    $('instruction-goal').textContent = tr(`每个小回合在 ${limit} 步内合作完成 ${target} 份汤。先进行共同练习，再完成 Task 1 和 Task 2，每个任务包含三个小回合。`, `Complete ${target} soups together within ${limit} turns in each round. Practice first, then complete Task 1 and Task 2, with three rounds in each task.`);
  }

  function cancelQuestion() { questionEpoch++; if (pollTimer) clearTimeout(pollTimer); pollTimer = null; questionJob = null; answer = null; shownAnswers.clear(); }
  function acceptView(next, {animateAction = false} = {}) {
    if (!next?.run && next?.view) next = next.view;
    if (!next?.run) { if (next?.session === null || next?.run === null) { view = null; cancelQuestion(); } render(); return; }
    rememberSync(storageGet(KEYS.sync));
    const floor = syncWatermarks.get(next.run.id);
    const incoming = {schema: 1, run_id: next.run.id, version: next.run.version, phase: next.run.phase};
    if (floor && compareSync(incoming, floor) < 0) { scheduleAuthorityRefresh(); return; }
    const oldRun = run(), oldState = view?.state, oldEpisode = currentEpisode();
    if (oldRun?.id === next.run.id && next.run.version < oldRun.version) return;
    const changed = oldRun?.id !== next.run.id || oldRun?.phase !== next.run.phase;
    view = next; language = run().language === 'en' ? 'en' : 'zh'; storageSet(KEYS.language, language);
    if (next.notice) notice = localized(next.notice);
    if (changed) { stopAuto(); cancelQuestion(); replayEpisode = null; replayFrame = null; helpOpen = false; }
    if (oldEpisode !== currentEpisode()) { stopAuto(); replayEpisode = null; replayFrame = null; answer = null; }
    if (!view.can_auto) stopAuto();
    if (!view.can_ask) cancelQuestion();
    if (view.state && currentEpisode()) { const frames = histories.get(currentEpisode()) || []; frames[view.state.turn] = clone(view.state); histories.set(currentEpisode(), frames); historyPermissions.set(currentEpisode(), Boolean(view.can_ask)); }
    const shouldAnimate = animateAction && oldEpisode === currentEpisode() && oldState && view.state?.turn === oldState.turn + 1 && isLive();
    render({paint: !shouldAnimate}); if (shouldAnimate) animate(oldState);
    if (currentEpisode() && !histories.get(currentEpisode())?.[0]) loadHistory(currentEpisode(), false).catch(() => {});
    if (view.can_ask && !questionJob && !answer) { const job = (view.questions || []).find(q => ['pending', 'queued', 'running'].includes(q.status)); if (job) beginPolling(job); }
    publishSync();
  }

  async function sessionView() { try { return await request('/api/view'); } catch (error) { if ([401, 404].includes(error.status)) return {run: null}; throw error; } }
  async function refresh() { const next = await sessionView(); acceptView(next); return next; }
  function serverErrorText(error) {
    const wait = status.qa_limits?.min_interval_seconds ?? 2;
    const messages = {
      question_episode_limit: tr('本小回合的提问次数已用完。', 'The question limit for this round has been reached.'),
      question_run_limit: tr('本次实验的提问次数已用完。', 'The question limit for this study run has been reached.'),
      question_budget_exhausted: tr('本次内部预实验的问答额度已用完，请联系研究者。', 'The question budget for this internal pilot has been used. Please contact the researcher.'),
      question_rate_limit: tr(`请等待 ${wait} 秒后再提问。`, `Please wait ${wait} seconds before asking another question.`),
      question_limit: tr('已有回答正在处理中，请等待完成。', 'Answers are already being processed. Please wait for them to finish.')
    };
    return messages[error.code] || error.message;
  }
  function publishSync() {
    if (!run()) return;
    const signal = {schema: 1, run_id: run().id, version: run().version, phase: run().phase};
    rememberSync(storageGet(KEYS.sync)); const floor = syncWatermarks.get(signal.run_id);
    if (floor && compareSync(signal, floor) < 0) { scheduleAuthorityRefresh(); return; }
    rememberSync(signal); storageSet(KEYS.sync, signal); if (syncChannel) syncChannel.postMessage(signal);
  }
  function scrubForAuthorityRefresh() {
    stopAuto(); stopAnimation(); cancelQuestion(); historyPermissions.clear();
    if (view) { view.can_ask = false; view.questions = []; }
    render();
  }
  function scheduleAuthorityRefresh() {
    syncRefreshQueued = true;
    if (syncRefresh) return;
    syncRefresh = Promise.resolve().then(async () => {
      // A newer cross-tab watermark can arrive while an older authoritative
      // response is still in flight. Keep the wake-up and fetch again after
      // rejecting that response instead of leaving this tab on the old phase.
      while (syncRefreshQueued) {
        syncRefreshQueued = false;
        try { await refresh(); } catch (_) {}
      }
    }).finally(() => {
      syncRefresh = null;
      if (syncRefreshQueued) scheduleAuthorityRefresh();
    });
  }
  function receiveSync(signal) {
    const strongest = rememberSync(signal); if (!strongest || compareSync(signal, strongest) < 0 || !run()) return;
    if (signal.run_id === run().id && compareSync(signal, {schema: 1, run_id: run().id, version: run().version, phase: run().phase}) <= 0) return;
    scrubForAuthorityRefresh(); scheduleAuthorityRefresh();
  }
  async function submit(path, body, {animateAction = false, existing = false} = {}) {
    if (busy || (pending && !existing)) return null;
    if (!existing) { pending = {schema: 1, path, run_id: run()?.id || null, body: {...body, operation_id: body.operation_id || newId()}}; storageSet(KEYS.pending, pending); }
    busy = true; notice = ''; renderControls(); renderNotice();
    const outgoing = pending;
    try {
      const result = await request(outgoing.path, outgoing.body);
      pending = null; storageDelete(KEYS.pending);
      if (result.run || result.view) acceptView(result, {animateAction});
      else if (run() && Number.isInteger(result.version)) {
        const floor = syncWatermarks.get(run().id);
        if (floor && result.version < floor.version) { scheduleAuthorityRefresh(); return null; }
        if (result.version >= run().version) { run().version = result.version; publishSync(); }
      }
      return result;
    } catch (error) {
      stopAuto();
      if (error.status) {
        pending = null; storageDelete(KEYS.pending);
        if (outgoing.path === '/api/session' && error.code === 'participant_id_taken') {
          participantIdError = 'taken'; notice = ''; renderParticipantIdError();
        } else if (outgoing.path === '/api/session' && error.code === 'participant_session_conflict') {
          notice = tr('当前浏览器已有参与者会话。请刷新页面恢复该会话。', 'This browser already has a participant session. Refresh the page to resume it.');
          try { await refresh(); } catch (_) {}
        } else {
          notice = error.status === 409 ? tr('进度已更新，已重新同步。请确认画面后继续。', 'Progress has changed and was resynchronized. Check the current state before continuing.') : serverErrorText(error);
          if ([409, 403].includes(error.status)) { try { await refresh(); } catch (_) {} }
        }
      }
      else notice = tr('连接中断，操作是否完成尚未确认。请重试；同一个操作不会执行两次。', 'Connection interrupted. This operation is unconfirmed. Retry it; the same operation cannot advance twice.');
      return null;
    } finally { busy = false; renderControls(); renderNotice(); if (participantIdError && !run()) $('user-id').focus(); }
  }

  const command = (name, fields = {}, options = {}) => submit('/api/command', {command: name, version: run()?.version, ...fields}, options);
  function renderParticipantIdError() {
    const text = participantIdError === 'taken' ? tr('这个用户 ID 已被使用。请在后面加 _1 或换一个 ID，例如 user_01_1。', 'This user ID is already in use. Add _1 or choose another ID, for example user_01_1.') : '';
    $('user-id-error').hidden = !text; $('user-id-error').textContent = text;
    $('user-id').setAttribute('aria-invalid', String(Boolean(text)));
    $('user-id').setCustomValidity(text || (PARTICIPANT_ID.test($('user-id').value) ? '' : tr('请输入 3–32 个字符，以英文字母开头，仅使用英文字母、数字、下划线或连字符。', 'Enter 3–32 characters, starting with a letter, using only letters, numbers, underscores or hyphens.')));
  }
  function renderNotice() { $('notice').hidden = !notice && !pending; $('notice-text').textContent = notice || (pending ? tr('仍有一个操作等待服务器确认。', 'One operation is awaiting server confirmation.') : ''); $('retry-request').hidden = !pending || busy; }
  function renderControls() {
    document.querySelectorAll('[data-action]').forEach(node => { node.disabled = !canAct(); });
    $('language').disabled = busy || Boolean(pending); $('join-study').disabled = busy || Boolean(pending) || !enrollmentEnabled();
    $('user-id').disabled = busy || Boolean(pending);
    $('freeplay').disabled = busy || Boolean(pending) || status.freeplay_available === false;
    $('accept-consent').disabled = !$('consent-check').checked || busy || Boolean(pending); $('start-practice').disabled = busy || Boolean(pending);
    $('recover-restart').disabled = busy || Boolean(pending) || !view?.can_restart;
    $('resume-technical-retry').disabled = busy || Boolean(pending);
    $('restart').hidden = !view?.can_restart; $('swap-role').hidden = !view?.can_swap; $('next').hidden = !view?.can_next;
    $('auto-controls').hidden = run()?.mode !== 'freeplay';
    $('auto-demo').disabled = !autoRunning && (!canAuto() || !canAct());
    $('auto-demo').textContent = autoRunning ? tr('暂停演示', 'Pause demonstration') : tr('自动演示', 'Auto demonstration');
    $('auto-demo').setAttribute('aria-pressed', String(autoRunning));
    $('auto-description').textContent = (view?.policy_kind || status.policy_kind) === 'program_baseline' ? tr('程序玩家 · 程序队友', 'Program player · Program teammate') : tr('程序玩家 · 神经策略队友', 'Program player · Neural teammate');
    for (const id of ['restart', 'swap-role', 'next']) $(id).disabled = busy || Boolean(pending) || animating || !isLive();
    const canQuestion = canAsk() && !busy && !pending && !questionJob;
    document.querySelectorAll('[data-explain]').forEach(node => { node.disabled = !canQuestion; });
    $('ask-question').disabled = !canQuestion; $('question-input').disabled = !canAsk() || Boolean(questionJob);
    $('question-controls').hidden = !canAsk(); $('qa-unavailable').hidden = canAsk(); $('explanation-card').classList.toggle('question-locked', !canAsk());
    $('question-pending').hidden = !questionJob || !canAsk(); $('submit-survey').disabled = busy || Boolean(pending);
    $('connection-status').textContent = busy ? tr('正在确认…', 'Confirming…') : pending ? tr('等待重试', 'Retry needed') : tr('进度已保存', 'Progress saved');
    const frames = histories.get(selectedEpisode()) || [], index = replayFrame ?? view?.state?.turn ?? 0;
    $('previous-frame').disabled = !displayedState() || index <= 0; $('next-frame').disabled = !displayedState() || index >= frames.length - 1;
    $('live').disabled = isLive(); $('timeline').disabled = !displayedState() || frames.length <= 1;
    $('retry-request').hidden = !pending || busy;
  }

  function renderResult(state) {
    $('result').hidden = !state?.done; $('result').replaceChildren(); if (!state?.done) return;
    const heading = document.createElement('strong'); heading.textContent = state.orders >= state.targetOrders ? tr('本小回合完成', 'Round completed') : tr('本小回合结束', 'Round ended');
    const detail = document.createElement('span'); detail.textContent = tr(`完成 ${state.orders} / ${state.targetOrders} 份汤，使用 ${state.turn} 步。`, `${state.orders} / ${state.targetOrders} soups served in ${state.turn} turns.`);
    $('result').append(heading, detail);
  }

  function renderPlay() {
    const state = displayedState(); if (!state) return;
    $('participant-label').textContent = run()?.mode === 'freeplay' ? tr('自由试玩', 'Free play') : tr(`参与者 ${run()?.participant_id || ''}`, `Participant ${run()?.participant_id || ''}`);
    $('phase-label').textContent = phaseText(phase());
    const record = view?.episodes?.find(e => e.id === currentEpisode()); $('round-label').textContent = ['task1', 'task2'].includes(phase()) ? tr(`小回合 ${((record?.index ?? run()?.episode_index ?? 1) - 1) % 3 + 1} / 3`, `Round ${((record?.index ?? run()?.episode_index ?? 1) - 1) % 3 + 1} / 3`) : '';
    $('map-label').textContent = `${state.map?.[0]?.length || 9} × ${state.map?.length || 7}`;
    $('orders-count').textContent = `${state.orders} / ${state.targetOrders}`; $('turn-count').textContent = `${state.turn} / ${state.maxSteps}`;
    $('score-count').textContent = Number.isFinite(state.score) ? String(Math.round(state.score * 100) / 100) : '—';
    $('pot-status').textContent = state.pot.ready ? tr('已煮熟', 'Ready') : state.pot.remaining ? tr(`还需 ${state.pot.remaining} 步`, `${state.pot.remaining} turns left`) : tr(`${state.pot.ingredients} / 3 洋葱`, `${state.pot.ingredients} / 3 onions`);
    $('human-held').textContent = itemText(state.actors.find(a => a.id === 'human')?.holding); $('ai-held').textContent = itemText(state.actors.find(a => a.id === 'ai')?.holding);
    $('feedback').textContent = localized(view?.feedback) || (state.done ? '' : tr('面向设施按 E 交互。', 'Face a station and press E to interact.'));
    renderResult(state);
    const episodes = view?.episodes || []; const previousSelection = selectedEpisode(); $('episode-select').replaceChildren();
    for (const episode of episodes) { const option = document.createElement('option'); option.value = episode.id; option.textContent = roundText(episode); $('episode-select').append(option); }
    $('episode-select').value = previousSelection || ''; $('episode-select').disabled = episodes.length <= 1;
    const frames = histories.get(selectedEpisode()) || []; const maximum = Math.max(0, frames.length - 1, selectedEpisode() === currentEpisode() ? view.state.turn : 0);
    $('timeline').max = maximum; $('timeline').value = state.turn; $('timeline-label').textContent = `${state.turn} / ${maximum}`;
    $('replay-badge').hidden = isLive(); $('replay-badge').textContent = tr(`回放 · 第 ${state.turn} 步`, `Replay · Turn ${state.turn}`);
    $('question-anchor').textContent = `${roundText(currentEpisodeRecord())} · ${tr('第', 'Turn')} ${state.turn}${tr(' 步', '')}`;
    $('next').textContent = phase() === 'practice' ? tr('进入 Task 1', 'Begin Task 1') : phase() === 'task2' && run()?.episode_index >= 6 ? tr('填写问卷', 'Questionnaire') : tr('下一小回合', 'Next round');
    if (run()?.mode === 'freeplay') $('next').textContent = tr('继续', 'Continue');
    renderAnswer();
  }

  function renderAnswer() {
    $('explanation').hidden = !answer || !canAsk(); $('past-answers').replaceChildren();
    if (!canAsk()) { $('explanation-text').textContent = ''; $('explanation-title').textContent = ''; $('explanation-frame').textContent = ''; $('explanation-source').textContent = ''; return; }
    if (answer) {
      const value = answer.answer || answer;
      $('explanation-title').textContent = localized(value.title) || answer.question || tr('行为解释', 'Behavior explanation');
      $('explanation-frame').textContent = `${roundText(view?.episodes?.find(e => e.id === answer.episode_id))} · ${tr('所选画面：第', 'Selected frame: turn')} ${answer.frame ?? value.frame ?? 0}${tr(' 步', '')}`;
      $('explanation-text').textContent = localized(value.text || value.answer || value.response) || tr('未能生成可核验的回答，请换一种方式提问。', 'A verified answer could not be generated. Please rephrase your question.');
      $('explanation-source').textContent = localized(value.source_summary) || (value.verified ? tr('来源：所选状态、实际策略输出及核验结果。', 'Source: selected state, actual policy output and verification.') : tr('仅展示系统能够确认的信息。', 'Only information confirmed by the system is shown.'));
    }
    for (const previous of (view?.questions || []).filter(q => q.answer && q.episode_id === selectedEpisode() && q.id !== answer?.id).slice(-8).reverse()) {
      const button = document.createElement('button'); button.type = 'button'; button.className = 'past-answer'; button.textContent = previous.question || localized(previous.answer.title);
      const label = document.createElement('span'); label.textContent = tr(`第 ${previous.frame} 步的回答`, `Answer for turn ${previous.frame}`); button.append(label);
      button.onclick = () => showAnswer(previous); $('past-answers').append(button);
    }
  }

  function renderSurvey() {
    const survey = view?.survey; if (!survey) return;
    if (surveyRun !== run()?.id) { surveyRun = run()?.id; surveyDraft = {...survey.draft}; const local = storageGet(KEYS.survey); if (local?.run_id === surveyRun) surveyDraft = {...surveyDraft, ...local.answers}; surveySignature = ''; surveyDirty = false; }
    const signature = `${run()?.id}:${language}:${JSON.stringify(survey.items)}`;
    if (surveySignature === signature) return; surveySignature = signature; $('survey-items').replaceChildren();
    for (const [index, entry] of (survey.items || []).entries()) {
      const field = document.createElement('fieldset'); field.className = 'survey-item'; field.dataset.item = entry.id;
      const legend = document.createElement('legend'); legend.textContent = `${index + 1}. ${localized(entry.prompt)}`; field.append(legend);
      if (entry.assumption) { const assumption = document.createElement('p'); assumption.className = 'survey-assumption muted'; assumption.textContent = `${tr('假设：', 'Assumption: ')}${localized(entry.assumption)}`; field.append(assumption); }
      if (Number.isInteger(entry.frame)) { const frame = document.createElement('p'); frame.className = 'small muted'; frame.textContent = tr(`题目画面：第 ${entry.frame} 步。`, `Question frame: turn ${entry.frame}.`); field.append(frame); }
      if (entry.state) { const canvas = document.createElement('canvas'); canvas.className = 'prediction-board'; canvas.dataset.item = entry.id; field.append(canvas); window.KitchenRenderer.draw(canvas, entry.state, {language}); }
      const options = document.createElement('div'); options.className = 'survey-options';
      for (const choice of entry.options || []) {
        const label = document.createElement('label'); label.className = 'survey-option'; const input = document.createElement('input'); input.type = 'radio'; input.name = entry.id; input.value = choice.value; input.required = true; input.checked = String(surveyDraft[entry.id]) === String(choice.value);
        const text = document.createElement('span'); text.textContent = localized(choice.label); label.append(input, text); options.append(label);
      }
      field.append(options); $('survey-items').append(field);
    }
  }

  function render({paint = true} = {}) {
    translate(); for (const id of ['lobby', 'consent', 'release-recovery', 'technical-retry', 'instructions', 'play', 'survey', 'complete']) $(id).hidden = true;
    const stage = phase(); if (!run()) $('lobby').hidden = false;
    else if (view.requires_restart) $('release-recovery').hidden = false;
    else if (stage === 'technical_retry_closed') $('technical-retry').hidden = false;
    else if (stage === 'consent') {
      $('consent').hidden = false; const doc = view.consent || status.consent; $('consent-content').textContent = localized(doc?.text) || tr('参与者说明暂不可用，请联系研究者后再继续。', 'Participant information is unavailable. Please contact the researcher before continuing.');
    } else if (stage === 'instructions' || helpOpen) { $('instructions').hidden = false; $('start-practice').hidden = helpOpen; $('close-instructions').hidden = !helpOpen; $('start-practice').textContent = run().mode === 'freeplay' ? tr('开始试玩', 'Start free play') : tr('开始练习', 'Start practice'); }
    else if (stage === 'questionnaire' || stage === 'survey') { $('survey').hidden = false; renderSurvey(); }
    else if (stage === 'complete' || stage === 'completed') { $('complete').hidden = false; $('completion-code').textContent = view.completion_code || ''; storageDelete(KEYS.survey); }
    else if (view?.state) { $('play').hidden = false; renderPlay(); }
    const current = stage === 'task1' ? 'task1' : stage === 'task2' ? 'task2' : ['questionnaire', 'survey', 'complete', 'completed'].includes(stage) ? 'questionnaire' : 'practice';
    document.querySelectorAll('[data-phase]').forEach(node => { node.classList.toggle('active', node.dataset.phase === current); if (node.dataset.phase === current) node.setAttribute('aria-current', 'step'); else node.removeAttribute('aria-current'); });
    renderControls(); renderNotice(); if (paint) { stopAnimation(); draw(); renderControls(); }
  }

  async function loadHistory(episodeId, select = true) {
    if (select) stopAuto();
    const generation = ++historyRequest; const response = await request(`/api/history?episode_id=${encodeURIComponent(episodeId)}`);
    const floor = run() ? syncWatermarks.get(run().id) : null;
    if (Number.isInteger(response?.version) && (response.version < (run()?.version ?? 0) || floor && response.version < floor.version)) { scheduleAuthorityRefresh(); return; }
    const frames = Array.isArray(response) ? response : response.frames || []; const existing = histories.get(episodeId) || [];
    if (typeof response.can_ask === 'boolean') historyPermissions.set(episodeId, response.can_ask);
    for (const frame of frames) { const state = frame.public || frame.state || frame; if (Number.isInteger(state.turn)) existing[state.turn] = state; }
    histories.set(episodeId, existing);
    if (select && generation === historyRequest) { replayEpisode = episodeId; replayFrame = Math.max(0, existing.length - 1); answer = null; render(); }
    else if (selectedEpisode() === episodeId) render();
  }
  async function selectFrame(index) { stopAuto(); stopAnimation(); const id = selectedEpisode(); if (!id) return; if (!histories.get(id)?.[index]) { try { await loadHistory(id, false); } catch (_) { notice = tr('回放暂时无法加载，请稍后重试。', 'Replay could not be loaded. Please try again.'); renderNotice(); return; } } if (!histories.get(id)?.[index]) return; replayEpisode = id; replayFrame = index; answer = null; render(); }

  function exposure(question, event) { if (!question?.id || !view?.can_ask) return; const data = {operation_id: newId(), question_id: question.id, event, version: run()?.version}; request('/api/exposure', data).catch(() => { setTimeout(() => { if (view?.can_ask) request('/api/exposure', data).catch(() => {}); }, 1200); }); }
  function showAnswer(value) { const floor = run() ? syncWatermarks.get(run().id) : null; if (!canAsk() || floor && Number.isInteger(value?.version) && value.version < floor.version || value.episode_id && !view?.episodes?.some(e => e.id === value.episode_id)) return; stopAuto(); answer = value; renderAnswer(); renderControls(); if (!shownAnswers.has(value.id)) { shownAnswers.add(value.id); exposure(value, 'shown'); } }
  function beginPolling(job) {
    if (!view?.can_ask || !job?.id) return; if (pollTimer) clearTimeout(pollTimer);
    const epoch = ++questionEpoch, originalPhase = phase(), originalRun = run()?.id; questionJob = job; renderControls();
    const poll = async () => {
      if (epoch !== questionEpoch || phase() !== originalPhase || run()?.id !== originalRun || !view?.can_ask) return;
      try {
        const result = await request(`/api/question/${encodeURIComponent(job.id)}`);
        const floor = syncWatermarks.get(originalRun);
        if (floor && Number.isInteger(result.version) && result.version < floor.version) { questionJob = null; pollTimer = null; scrubForAuthorityRefresh(); scheduleAuthorityRefresh(); return; }
        if (epoch !== questionEpoch || phase() !== originalPhase || run()?.id !== originalRun || !view?.can_ask) return;
        if (Number.isInteger(result.version) && result.version > run().version) run().version = result.version;
        if (['complete', 'completed', 'done', 'failed', 'error', 'cancelled'].includes(result.status) || result.answer) {
          questionJob = null; pollTimer = null;
          if (result.answer) { const full = {...job, ...result}; view.questions = (view.questions || []).filter(q => q.id !== full.id).concat(full); showAnswer(full); }
          else { notice = localized(result.error) || tr('本次问题未能完成，请重新提问。', 'This question could not be completed. Please ask again.'); renderNotice(); }
          renderControls(); return;
        }
        pollTimer = setTimeout(poll, 900);
      } catch (error) { if (error.status === 403 || error.status === 404) { questionJob = null; answer = null; try { await refresh(); } catch (_) {} renderControls(); renderAnswer(); return; } if (epoch === questionEpoch) pollTimer = setTimeout(poll, 1800); }
    };
    pollTimer = setTimeout(poll, 200);
  }
  async function ask(kind, question) {
    stopAuto();
    if (!canAsk() || busy || pending || questionJob) return; stopAnimation();
    const bound = {episode_id: selectedEpisode(), frame: frameNumber(), kind, question}; const origin = `${run()?.id}:${phase()}`;
    const result = await submit('/api/question', {operation_id: newId(), version: run()?.version, ...bound});
    if (!result || !view?.can_ask || origin !== `${run()?.id}:${phase()}`) return;
    const job = {...bound, ...result}; if (job.answer) showAnswer(job); else beginPolling(job);
  }

  async function saveSurvey() {
    if (!surveyDirty || !['questionnaire', 'survey'].includes(phase())) return;
    if (busy || pending) { surveyTimer = setTimeout(saveSurvey, 700); return; }
    const sent = clone(surveyDraft); $('survey-save-status').textContent = tr('正在保存…', 'Saving…');
    const result = await command('survey_save', {answers: sent});
    if (result) { surveyDirty = JSON.stringify(sent) !== JSON.stringify(surveyDraft); $('survey-save-status').textContent = tr('草稿已保存', 'Draft saved'); if (surveyDirty) surveyTimer = setTimeout(saveSurvey, 600); }
    else $('survey-save-status').textContent = tr('草稿尚未确认保存，请恢复连接后重试。', 'The draft is not yet confirmed. Retry after reconnecting.');
  }

  $('user-id').oninput = () => { participantIdError = null; renderParticipantIdError(); };
  $('join-form').onsubmit = async event => { event.preventDefault(); const field = $('user-id'); field.value = field.value.trim(); renderParticipantIdError(); if (!enrollmentEnabled() || !$('join-form').reportValidity()) return; const result = await submit('/api/session', {operation_id: newId(), participant_id: field.value, language, mode: 'pilot'}); if (result) { field.value = ''; participantIdError = null; renderParticipantIdError(); } };
  $('freeplay').onclick = () => submit('/api/session', {operation_id: newId(), language, mode: 'freeplay'});
  $('consent-check').onchange = renderControls; $('accept-consent').onclick = () => { if ($('consent-check').checked) command('consent', {accepted: true}); };
  $('start-practice').onclick = () => command('next'); $('next').onclick = () => { closeAnswer(); command('next'); };
  $('restart').onclick = () => { stopAuto(); closeAnswer(); command('restart'); }; $('swap-role').onclick = () => { stopAuto(); closeAnswer(); command('swap'); };
  $('auto-demo').onclick = () => { if (autoRunning) stopAuto(); else startAuto(); };
  $('recover-restart').onclick = () => command('restart');
  $('resume-technical-retry').onclick = () => submit('/api/session', {operation_id: newId(), participant_id: run()?.participant_id, language, mode: 'pilot'});
  $('language').onclick = async () => { stopAuto(); stopAnimation(); if (run()) await command('language', {language: language === 'zh' ? 'en' : 'zh'}); else { language = language === 'zh' ? 'en' : 'zh'; storageSet(KEYS.language, language); render(); } };
  $('show-instructions').onclick = () => { stopAuto(); stopAnimation(); helpOpen = true; render(); }; $('close-instructions').onclick = () => { helpOpen = false; render(); $('board').focus(); };
  document.querySelectorAll('[data-action]').forEach(node => { node.onpointerdown = stopAuto; node.onclick = () => { stopAuto(); if (canAct()) command('action', {action: node.dataset.action}, {animateAction: true}); }; });
  $('episode-select').onchange = () => { stopAnimation(); loadHistory($('episode-select').value).catch(() => { notice = tr('回放暂时无法加载。', 'Replay could not be loaded.'); renderNotice(); }); };
  $('timeline').oninput = event => selectFrame(Number(event.target.value)); $('previous-frame').onclick = () => selectFrame(Math.max(0, frameNumber() - 1)); $('next-frame').onclick = () => selectFrame(Math.min((histories.get(selectedEpisode())?.length || 1) - 1, frameNumber() + 1));
  $('live').onclick = () => { stopAuto(); replayEpisode = null; replayFrame = null; answer = null; render(); };
  document.querySelectorAll('[data-explain]').forEach(node => { node.onclick = () => ask(node.dataset.explain, node.textContent); });
  $('question-form').onsubmit = event => { event.preventDefault(); const text = $('question-input').value.trim(); if (text) ask('free', text); };
  $('question-input').onfocus = stopAuto;
  function closeAnswer() { if (answer) exposure(answer, 'closed'); answer = null; renderAnswer(); renderControls(); }
  $('close-explanation').onclick = closeAnswer;
  $('retry-request').onclick = async () => { if (!pending) return; const outgoing = pending; const result = await submit(outgoing.path, outgoing.body, {existing: true}); if (result && outgoing.path === '/api/question') beginPolling({...outgoing.body, ...result}); };
  $('survey-form').onchange = event => { if (!event.target.name) return; surveyDraft[event.target.name] = event.target.value; surveyDirty = true; storageSet(KEYS.survey, {run_id: run()?.id, answers: surveyDraft}); $('survey-save-status').textContent = tr('正在保存草稿…', 'Saving draft…'); if (surveyTimer) clearTimeout(surveyTimer); surveyTimer = setTimeout(saveSurvey, 400); };
  $('survey-form').onsubmit = async event => { event.preventDefault(); if (!$('survey-form').reportValidity() || busy || pending) return; if (surveyTimer) clearTimeout(surveyTimer); const result = await command('survey_submit', {answers: clone(surveyDraft)}); if (result) { surveyDirty = false; storageDelete(KEYS.survey); } };
  document.addEventListener('keydown', event => {
    if (event.repeat || event.ctrlKey || event.metaKey || event.altKey || event.target.closest('input,textarea,select,[contenteditable=true]')) return;
    if (event.target.closest('button') && [' ', 'Enter'].includes(event.key)) return;
    const keys = {ArrowUp: 'UP', ArrowDown: 'DOWN', ArrowLeft: 'LEFT', ArrowRight: 'RIGHT', w: 'UP', W: 'UP', s: 'DOWN', S: 'DOWN', a: 'LEFT', A: 'LEFT', d: 'RIGHT', D: 'RIGHT', e: 'INTERACT', E: 'INTERACT', ' ': 'WAIT'};
    const action = keys[event.key]; if (!action || $('play').hidden) return; event.preventDefault(); stopAuto(); if (canAct()) command('action', {action}, {animateAction: true});
  });
  window.addEventListener('resize', () => { stopAnimation(); draw(); renderControls(); });
  document.addEventListener('visibilitychange', () => { if (document.hidden) { stopAuto(); stopAnimation(); draw(); renderControls(); } else if (run() && !busy && !pending) refresh().catch(() => {}); });
  window.addEventListener('storage', event => { if (event.key !== KEYS.sync || !event.newValue) return; try { receiveSync(JSON.parse(event.newValue)); } catch (_) {} });
  if (syncChannel) syncChannel.onmessage = event => receiveSync(event.data);
  // Standard UI actions synchronize immediately. This sparse research-only
  // check also catches state changes made outside the page (for example an
  // administered retry or a direct same-origin API request) and removes stale
  // Task 1 explanations shortly after the authority advances.
  setInterval(() => {
    if (!document.hidden && run()?.mode === 'pilot' && view?.can_ask && !busy && !pending) scheduleAuthorityRefresh();
  }, AUTHORITY_CHECK_MS);
  window.addEventListener('pageshow', event => { if (event.persisted && run()) { scrubForAuthorityRefresh(); scheduleAuthorityRefresh(); } });
  window.addEventListener('focus', () => { if (run() && !busy && !pending) { scrubForAuthorityRefresh(); scheduleAuthorityRefresh(); } });
  window.addEventListener('offline', stopAuto);
  window.addEventListener('online', () => { if (!pending) refresh().catch(() => {}); });

  async function boot() {
    busy = true; render();
    try { const results = await Promise.all([request('/api/status'), sessionView()]); status = results[0]; acceptView(results[1]);
      if (pending && pending.run_id && pending.run_id !== run()?.id) { pending = null; storageDelete(KEYS.pending); notice = tr('当前会话已改变，旧会话的未确认请求没有继续发送。', 'The session has changed. Its previous unconfirmed request was not sent.'); }
    } catch (_) { notice = tr('暂时无法连接实验服务器，请刷新页面或恢复连接后重试。', 'The study server is unavailable. Refresh this page or reconnect and retry.'); }
    finally { busy = false; render(); }
    if (pending) { const outgoing = pending; const result = await submit(outgoing.path, outgoing.body, {existing: true}); if (result && outgoing.path === '/api/question') beginPolling({...outgoing.body, ...result}); }
  }
  boot();
}());
