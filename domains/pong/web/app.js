const $ = id => document.getElementById(id);
const API = '/pong/api';
let running = false;
let currentAction = 'stay';
let view = null;
let replayPlaying = false;
let replayTimer = null;
let replayFrame = null;
let replayLocked = false;
let reviewRequestId = 0;
let refreshInFlight = false;
let latestFrame = null;
let latestFrameReceivedAt = 0;
let animationRequest = null;
const heldKeys = new Set();

async function request(path, options = {}) {
  const response = await fetch(API + path, { cache: 'no-store', ...options });
  return response.json();
}
async function post(path, payload = {}) {
  return request(path, { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload) });
}

function setAction(action) {
  if (!running || currentAction === action) return;
  currentAction = action;
  void post('/input', {action});
}

function updateKeyboardAction() {
  if (heldKeys.has('left') && heldKeys.has('right')) return setAction('stay');
  if (heldKeys.has('left')) return setAction('left');
  if (heldKeys.has('right')) return setAction('right');
  setAction('stay');
}

function visualSeconds() {
  // Server snapshots correct the picture; rAF fills the gaps without turning
  // network cadence into visible movement cadence. The short cap prevents a
  // stalled request from showing a fictitious future catch or bounce.
  return Math.min(.07, Math.max(0, (performance.now() - latestFrameReceivedAt) / 1000));
}

function drawCourt(frame, { extrapolate = false } = {}) {
  if (!frame) return;
  const cols = Math.max(1, Number(view?.grid_columns || 30));
  const rows = Math.max(1, Number(view?.grid_rows || 18));
  const paddleY = Number(view?.paddle_y ?? Math.max(0, rows - 2));
  $('court').style.setProperty('--grid-columns', String(cols));
  $('court').style.setProperty('--grid-rows', String(rows));
  $('court').style.setProperty('--board-aspect', `${cols} / ${rows}`);
  const paddleWidth = Number(view?.paddle_width_cells || view?.paddle_width || 4);
  $('court').style.setProperty('--paddle-width', `${(paddleWidth / cols) * 100}%`);
  const paddleHeight = Number(view?.paddle_height_cells || 1);
  $('court').style.setProperty('--paddle-height', `${(paddleHeight / rows) * 100}%`);
  const playerCol = Number(frame.player_col ?? frame.player_x ?? cols / 2);
  const aiCol = Number(frame.ai_col ?? frame.ai_x ?? cols / 2);
  const paddleTop = (paddleY / rows) * 100;
  $('playerPaddle').style.left = `${(playerCol / cols) * 100}%`;
  $('playerPaddle').style.top = `${paddleTop}%`;
  $('aiPaddle').style.left = `${(aiCol / cols) * 100}%`;
  $('aiPaddle').style.top = `${paddleTop}%`;
  const layer = $('balls');
  const present = new Set();
  (frame.balls || []).forEach(ball => {
    present.add(ball.ball_id);
    let node = layer.querySelector(`[data-ball-id="${ball.ball_id}"]`);
    if (!node) {
      node = document.createElement('i');
      node.dataset.ballId = ball.ball_id;
      layer.appendChild(node);
    }
    const assigned = view?.group === 'A' && view?.task === 1 && view?.intent_bubble?.target_ball_id === ball.ball_id;
    node.className = `ball ${ball.kind === 'large' ? 'large' : 'small'}${assigned ? ' assigned' : ''}`;
    node.textContent = ball.ball_id;
    const widthCells = Math.max(1, Number(ball.width_cells || (ball.kind === 'large' ? 2 : 1)));
    const heightCells = Math.max(1, Number(ball.height_cells || (ball.kind === 'large' ? 2 : 1)));
    const elapsed = extrapolate ? visualSeconds() : 0;
    const x = Math.max(0, Math.min(cols - widthCells, Number(ball.x ?? 0) + Number(ball.vx ?? 0) * elapsed));
    const y = Math.max(0, Math.min(rows - heightCells, Number(ball.y ?? 0) + Number(ball.vy ?? 0) * elapsed));
    node.style.left = `${(x / cols) * 100}%`;
    node.style.top = `${(y / rows) * 100}%`;
    node.style.setProperty('--ball-width', `${(widthCells / cols) * 100}%`);
    node.style.setProperty('--ball-height', `${(heightCells / rows) * 100}%`);
  });
  layer.querySelectorAll('[data-ball-id]').forEach(node => {
    if (!present.has(node.dataset.ballId)) node.remove();
  });
}

function animateCourt() {
  if (latestFrame && !replayLocked && view && !view.frame?.terminal) {
    drawCourt(latestFrame, { extrapolate: true });
  }
  animationRequest = requestAnimationFrame(animateCourt);
}

function draw(next) {
  view = next;
  if (!view || !view.started) return;
  $('task').textContent = view.task;
  const frame = view.frame;
  $('time').textContent = Number(frame.time_seconds || 0).toFixed(1);
  $('missed').textContent = frame.missed_balls;
  $('opportunities').textContent = frame.total_opportunities;
  latestFrame = frame;
  latestFrameReceivedAt = performance.now();
  drawCourt(frame, { extrapolate: true });
  const bubble = $('intentBubble');
  const showBubble = view.group === 'A' && view.task === 1 && !frame.terminal && view.intent_bubble;
  bubble.hidden = !showBubble;
  const bubbleColumns = Math.max(1, Number(view.grid_columns || 30));
  const bubbleAiCol = Number(frame.ai_col ?? frame.ai_x ?? bubbleColumns / 2);
  bubble.style.setProperty('--bubble-x', `${(bubbleAiCol / bubbleColumns) * 100}%`);
  if (showBubble) {
    $('intentText').textContent = view.intent_bubble.text;
    $('intentDistance').textContent = view.intent_bubble.detail_text || view.intent_bubble.distance_label || '';
  } else {
    $('intentText').textContent = '';
    $('intentDistance').textContent = '';
  }
  const ended = view.phase === 'review' || (frame.terminal && view.task === 2);
  $('status').textContent = view.phase === 'review'
    ? 'Task 1 已结束，可以回放提问'
    : (frame.terminal && view.task === 1)
      ? 'Task 1 已结束，准备好后点击进入 Task 2'
      : ended ? '本局结束' : view.paused ? '已暂停' : '连续运行中';
  $('pause').textContent = view.paused ? '继续' : '暂停';
  ['left', 'pause', 'right'].forEach(id => { $(id).hidden = Boolean(frame.terminal); });
  $('task2').hidden = !view.task2_available;
  $('task2').disabled = !view.task2_available;
  if (view.review_available) {
    $('review').hidden = false;
    const max = Math.max(0, Number(view.replay_frame_count || 1) - 1);
    $('timeline').max = String(max);
    if (replayFrame === null || replayFrame > max) replayFrame = max;
    if (!$('timeline').dataset.userMoving) $('timeline').value = String(replayFrame);
    $('replayFrame').textContent = replayFrame;
    renderReviewEvents(view.replay_events || [], null);
  } else {
    $('review').hidden = true;
    replayFrame = null;
  }
}

async function refresh() {
  if (refreshInFlight) return;
  refreshInFlight = true;
  try {
    const next = await request('/view');
    if (next.started) {
      running = (!next.frame.terminal && next.task === 1) || (!next.frame.terminal && next.task === 2);
      $('setup').hidden = true;
      $('game').hidden = false;
      // Polling remains live while a participant is reviewing a historical
      // frame, but it must not replace the selected replay picture.
      if (!replayLocked) draw(next);
      else {
        view = next;
        renderReviewEvents(next.replay_events || [], null);
      }
    }
  } catch (error) {
    $('status').textContent = '服务连接中断，请稍后刷新。';
  } finally {
    refreshInFlight = false;
  }
}

async function loadReview(frame) {
  replayFrame = Number(frame);
  replayLocked = true;
  const requestId = ++reviewRequestId;
  const data = await request(`/review?frame=${encodeURIComponent(frame)}`);
  if (requestId !== reviewRequestId || !replayLocked) return;
  if (!data.allowed) { $('answer').textContent = '当前阶段不提供回放问答。'; return; }
  const state = data.state;
  $('replayFrame').textContent = data.frame;
  $('replayTime').textContent = Number(state.time_seconds || 0).toFixed(1);
  $('replayState').textContent = `双方位置：你第${state.player_col ?? Number(state.player_x).toFixed(1)}列 · 机器人2第${state.ai_col ?? Number(state.ai_x).toFixed(1)}列`;
  drawCourt(state);
  renderReviewEvents(data.events || [], data.event || null);
}

function renderReviewEvents(events, activeEvent) {
  const container = $('reviewEvents');
  if (!container) return;
  container.replaceChildren();
  if (!events || !events.length) {
    const empty = document.createElement('div');
    empty.className = 'review-events-empty';
    empty.textContent = '暂无关键事件';
    container.appendChild(empty);
    return;
  }
  events.forEach(event => {
    const button = document.createElement('button');
    button.className = 'review-event';
    button.dataset.frame = String(event.frame);
    button.textContent = `第${event.frame}帧 · ${event.label}`;
    if (activeEvent && event.event_id === activeEvent.event_id) button.classList.add('active');
    button.onclick = () => {
      $('timeline').value = String(event.frame);
      void loadReview(event.frame);
    };
    container.appendChild(button);
  });
}

$('start').onclick = async () => {
  const result = await post('/start', {
    participant_id: $('participant').value,
    group: $('group').value,
  });
  running = true;
  currentAction = 'stay';
  $('setup').hidden = true;
  $('game').hidden = false;
  draw(result);
};

function bindDirectionButton(id, action) {
  const button = $(id);
  button.onpointerdown = event => { event.preventDefault(); setAction(action); };
  const release = event => { event.preventDefault(); setAction('stay'); };
  button.onpointerup = release;
  button.onpointerleave = release;
  button.onpointercancel = release;
}
bindDirectionButton('left', 'left');
bindDirectionButton('right', 'right');
window.addEventListener('keydown', event => {
  if (event.target && ['INPUT', 'SELECT', 'TEXTAREA', 'BUTTON'].includes(event.target.tagName)) return;
  const key = event.key.toLowerCase();
  if (event.key === 'ArrowLeft' || key === 'a') { event.preventDefault(); heldKeys.add('left'); updateKeyboardAction(); }
  if (event.key === 'ArrowRight' || key === 'd') { event.preventDefault(); heldKeys.add('right'); updateKeyboardAction(); }
});
window.addEventListener('keyup', event => {
  const key = event.key.toLowerCase();
  if (event.key === 'ArrowLeft' || key === 'a') { event.preventDefault(); heldKeys.delete('left'); updateKeyboardAction(); }
  if (event.key === 'ArrowRight' || key === 'd') { event.preventDefault(); heldKeys.delete('right'); updateKeyboardAction(); }
});
window.addEventListener('blur', () => {
  heldKeys.clear();
  currentAction = 'stay';
  if (running) void post('/pause', {paused: true});
});
window.addEventListener('focus', () => { if (running) void post('/pause', {paused: false}); });

$('pause').onclick = async () => {
  if (!view) return;
  const result = await post('/pause', {paused: !view.paused});
  draw({...view, ...result});
};
$('task2').onclick = async () => {
  const result = await post('/task2');
  replayPlaying = false;
  if (replayTimer) clearInterval(replayTimer);
  replayLocked = false;
  reviewRequestId += 1;
  replayFrame = null;
  $('review').hidden = true;
  draw(result);
};
$('timeline').oninput = async event => {
  $('timeline').dataset.userMoving = 'true';
  replayFrame = Number(event.target.value);
  await loadReview(event.target.value);
};
$('timeline').onchange = event => { delete $('timeline').dataset.userMoving; void loadReview(event.target.value); };
$('prevFrame').onclick = () => {
  replayFrame = Math.max(0, Number($('timeline').value) - 1);
  $('timeline').value = String(replayFrame);
  void loadReview(replayFrame);
};
$('nextFrame').onclick = () => {
  replayFrame = Math.min(Number($('timeline').max), Number($('timeline').value) + 1);
  $('timeline').value = String(replayFrame);
  void loadReview(replayFrame);
};
$('playReplay').onclick = () => {
  replayPlaying = !replayPlaying;
  $('playReplay').textContent = replayPlaying ? '暂停回放' : '播放';
  if (replayPlaying) {
    replayTimer = setInterval(() => {
      const next = Number($('timeline').value) + 1;
      if (next > Number($('timeline').max)) { replayPlaying = false; clearInterval(replayTimer); $('playReplay').textContent = '播放'; return; }
      replayFrame = next;
      $('timeline').value = String(next);
      void loadReview(next);
    }, 80);
  } else if (replayTimer) clearInterval(replayTimer);
};
document.querySelectorAll('[data-question]').forEach(button => button.onclick = () => {
  $('question').value = button.dataset.question;
  $('ask').click();
});
$('ask').onclick = async () => {
  const result = await post('/ask', {
    question: $('question').value,
    frame: Number($('timeline').value),
    ball_id: $('ballSelect').value || null,
  });
  $('answer').textContent = result.answer || '当前阶段不提供解释。';
};

// Snapshots are deliberately less frequent than rendering. Motion stays
// smooth through requestAnimationFrame and is corrected by each authoritative
// snapshot; the in-flight guard avoids request queues on slower machines.
setInterval(() => { if (!document.hidden) void refresh(); }, 50);
if (!animationRequest) animationRequest = requestAnimationFrame(animateCourt);
void refresh();
