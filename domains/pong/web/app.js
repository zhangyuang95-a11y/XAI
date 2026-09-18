/* Cooperative Pong runs entirely in the browser after this file is loaded.
 * The server only hosts the static page; it is not part of the 60 Hz game
 * loop. This prevents network delay, Render cold starts, and HTTP polling
 * from changing a 90-second round into a longer one. */
const $ = id => document.getElementById(id);

const SPEC = Object.freeze({
  width: 24,
  height: 14,
  paddleY: 11,
  paddleWidth: 4,
  paddleSpeed: 5,
  ballSpeed: 2,
  fixedDt: 1 / 60,
  durationSeconds: 90,
  snapshotEvery: 6,
});
const EPSILON = 1e-6;
const heldKeys = new Set();

let game = null;
let running = false;
let currentAction = 'stay';
let replayPlaying = false;
let replayTimer = null;
let replayIndex = null;
let replayLocked = false;

function clone(value) { return JSON.parse(JSON.stringify(value)); }

function reflect(position, velocity, lower, upper) {
  let next = position;
  let speed = velocity;
  while (next < lower - EPSILON || next > upper + EPSILON) {
    if (next < lower) { next = lower + (lower - next); speed = Math.abs(speed); }
    else { next = upper - (next - upper); speed = -Math.abs(speed); }
  }
  return [Math.max(lower, Math.min(upper, next)), speed];
}

function paddleCovers(left, cell) {
  return left - EPSILON <= cell && cell < left + SPEC.paddleWidth - EPSILON;
}

function paddleTargetLeft(left, cell) {
  const maximum = SPEC.width - SPEC.paddleWidth;
  if (paddleCovers(left, cell)) return Math.max(0, Math.min(maximum, left));
  return Math.max(0, Math.min(maximum, cell < left ? cell : cell - SPEC.paddleWidth + EPSILON));
}

function travelSeconds(distance) { return Math.max(0, distance) / SPEC.paddleSpeed; }
function ballLabel(ball) { return `${ball.kind === 'large' ? '合作大球' : '小球'}${ball.ball_id}`; }
function cells(value) { return String(Math.max(0, Math.round(Number(value || 0)))); }

class OfflinePong {
  constructor({ group, participantId, task, seed }) {
    this.group = group;
    this.participantId = participantId || 'local';
    this.task = task;
    this.seed = seed;
    this.frame = 0;
    this.playerX = 4.72;
    this.aiX = 15.28;
    this.paused = false;
    this.phase = 'active';
    this.missedBalls = 0;
    this.totalOpportunities = 0;
    this.successfulOpportunities = 0;
    this.missedByType = { small: 0, large: 0 };
    this.commitment = null;
    this.latestDecision = null;
    this.lastDecisionFrame = -1;
    this.intentBubble = null;
    this.bubbleSignature = null;
    this.bubbleUpdatedAt = -Infinity;
    this.history = [];
    this.reviewEvents = [];
    this.balls = this.initialBalls(seed);
    this.latestDecision = this.chooseAI();
    this.updateBubble(true);
    this.recordSnapshot(true);
  }

  initialBalls(seed) {
    const reversed = Number(seed) % 2 === 1;
    const signs = reversed ? [-1, 1, -1, 1, -1] : [1, -1, 1, -1, 1];
    const starts = [
      ['A1', 'small', 3, 2], ['A2', 'small', 18, 5], ['A3', 'small', 12, 8],
      ['B1', 'large', 7, 1], ['B2', 'large', 16, 6],
    ];
    return starts.map(([ballId, kind, x, y], index) => ({
      ball_id: ballId, kind, x, y, vx: signs[index] * SPEC.ballSpeed, vy: SPEC.ballSpeed,
      width_cells: kind === 'large' ? 2 : 1, height_cells: kind === 'large' ? 2 : 1,
      descendingEncounter: false, pendingMiss: false, pendingMissId: null, encounterIndex: 0,
    }));
  }

  get terminal() { return this.phase === 'terminal'; }
  get timeSeconds() { return this.frame * SPEC.fixedDt; }

  movePaddle(left, action) {
    const direction = action === 'left' ? -1 : action === 'right' ? 1 : 0;
    return Math.max(0, Math.min(SPEC.width - SPEC.paddleWidth,
      left + direction * SPEC.paddleSpeed * SPEC.fixedDt));
  }

  predictContact(ball) {
    const probe = clone(ball);
    let elapsed = 0;
    for (let update = 1; update <= 720; update += 1) {
      const oldX = probe.x;
      const oldY = probe.y;
      [probe.x, probe.vx] = reflect(probe.x + probe.vx * SPEC.fixedDt, probe.vx,
        0, SPEC.width - probe.width_cells);
      [probe.y, probe.vy] = reflect(probe.y + probe.vy * SPEC.fixedDt, probe.vy,
        0, SPEC.height - probe.height_cells);
      elapsed += SPEC.fixedDt;
      const oldBottom = oldY + probe.height_cells;
      const newBottom = probe.y + probe.height_cells;
      if (probe.vy > 0 && !probe.descendingEncounter && oldBottom < SPEC.paddleY && newBottom >= SPEC.paddleY) {
        const fraction = (SPEC.paddleY - oldBottom) / Math.max(EPSILON, newBottom - oldBottom);
        const contactX = oldX + (probe.x - oldX) * fraction;
        return {
          ball_id: probe.ball_id, kind: probe.kind,
          opportunity_id: `${probe.ball_id}:${probe.encounterIndex + 1}`,
          time_until_contact: Math.max(0, elapsed - SPEC.fixedDt + fraction * SPEC.fixedDt),
          contact_cells: probe.kind === 'large' ? [contactX, contactX + 1] : [contactX],
        };
      }
      if (probe.vy > 0 && probe.y >= SPEC.height - probe.height_cells - EPSILON) probe.descendingEncounter = false;
    }
    return null;
  }

  candidateFor(ball) {
    const prediction = this.predictContact(ball);
    if (!prediction) return null;
    if (prediction.kind === 'small') {
      const cell = prediction.contact_cells[0];
      const targetX = paddleTargetLeft(this.aiX, cell);
      const playerTarget = paddleTargetLeft(this.playerX, cell);
      const aiDistance = Math.abs(targetX - this.aiX);
      const playerDistance = Math.abs(playerTarget - this.playerX);
      const aiViable = travelSeconds(aiDistance) <= prediction.time_until_contact + EPSILON;
      return {
        prediction, targetX, contactSide: null, requiresPartner: false, aiDistance, playerDistance,
        aiViable, viable: aiViable, handoff: paddleCovers(this.playerX, cell), penalty: 1,
      };
    }
    const options = [[prediction.contact_cells[0], prediction.contact_cells[1], 'left'],
      [prediction.contact_cells[1], prediction.contact_cells[0], 'right']].map(([own, teammate, side]) => {
      const targetX = paddleTargetLeft(this.aiX, own);
      const playerTarget = paddleTargetLeft(this.playerX, teammate);
      const aiDistance = Math.abs(targetX - this.aiX);
      const playerDistance = Math.abs(playerTarget - this.playerX);
      return { targetX, side, aiDistance, playerDistance,
        cost: travelSeconds(aiDistance) + travelSeconds(playerDistance) };
    });
    const option = options.sort((left, right) => left.cost - right.cost)[0];
    const aiViable = travelSeconds(option.aiDistance) <= prediction.time_until_contact + EPSILON;
    return {
      prediction, targetX: option.targetX, contactSide: option.side, requiresPartner: true,
      aiDistance: option.aiDistance, playerDistance: option.playerDistance, aiViable,
      viable: aiViable && travelSeconds(option.playerDistance) <= prediction.time_until_contact + EPSILON,
      handoff: false, penalty: 3,
    };
  }

  chooseAI() {
    const candidates = this.balls.map(ball => this.candidateFor(ball)).filter(Boolean);
    const playerClosest = candidates.slice().sort((left, right) => left.playerDistance - right.playerDistance)[0] || null;
    const current = this.commitment && candidates.find(item => item.requiresPartner
      && item.prediction.ball_id === this.commitment.ball_id
      && item.prediction.opportunity_id === this.commitment.opportunity_id
      && item.contactSide === this.commitment.contactSide);
    let selected = current && current.aiViable ? current : null;
    if (!selected) {
      this.commitment = null;
      const available = candidates.filter(item => item.viable && !item.handoff);
      selected = available.sort((left, right) => {
        const leftSlack = Math.max(0, left.prediction.time_until_contact - travelSeconds(left.aiDistance)) / left.penalty;
        const rightSlack = Math.max(0, right.prediction.time_until_contact - travelSeconds(right.aiDistance)) / right.penalty;
        return leftSlack - rightSlack || right.penalty - left.penalty || left.aiDistance - right.aiDistance
          || left.prediction.ball_id.localeCompare(right.prediction.ball_id);
      })[0] || null;
      if (selected?.requiresPartner) {
        this.commitment = { ball_id: selected.prediction.ball_id,
          opportunity_id: selected.prediction.opportunity_id, contactSide: selected.contactSide };
      }
    }
    if (!selected) {
      return { action: 'stay', intentType: 'no_urgent', targetBallId: null, targetKind: null,
        targetX: this.aiX, contactSide: null, distance: 0,
        playerDistance: playerClosest?.playerDistance ?? null,
        playerClosestId: playerClosest?.prediction.ball_id ?? null, requiresPartner: false,
        smallFirstId: null, smallFirstFeasible: null, commitmentState: 'none', candidates };
    }
    const difference = selected.targetX - this.aiX;
    const action = difference > .025 ? 'right' : difference < -.025 ? 'left' : 'stay';
    let smallFirstId = null;
    let smallFirstFeasible = null;
    if (selected.requiresPartner) {
      const small = candidates.filter(item => item.prediction.kind === 'small' && item.viable && !item.handoff)
        .sort((left, right) => left.prediction.time_until_contact - right.prediction.time_until_contact)[0];
      if (small) {
        smallFirstId = small.prediction.ball_id;
        const leaveSmallAt = Math.max(travelSeconds(small.aiDistance), small.prediction.time_until_contact);
        const transfer = travelSeconds(Math.abs(selected.targetX - small.targetX));
        smallFirstFeasible = leaveSmallAt + transfer <= selected.prediction.time_until_contact + EPSILON
          && travelSeconds(selected.playerDistance) <= selected.prediction.time_until_contact + EPSILON;
      }
    }
    return {
      action, intentType: selected.requiresPartner ? (action === 'stay' ? 'hold_large' : 'catch_large')
        : (action === 'stay' ? 'hold_target' : 'catch_small'),
      targetBallId: selected.prediction.ball_id, targetKind: selected.prediction.kind, targetX: selected.targetX,
      contactSide: selected.contactSide, distance: selected.aiDistance, playerDistance: selected.playerDistance,
      playerClosestId: playerClosest?.prediction.ball_id ?? null, requiresPartner: selected.requiresPartner,
      smallFirstId, smallFirstFeasible,
      commitmentState: selected.requiresPartner ? (action === 'stay' ? 'waiting_for_large' : 'approaching_large') : 'none',
      candidates, opportunityId: selected.prediction.opportunity_id,
    };
  }

  advanceBall(ball) {
    const oldX = ball.x;
    const oldY = ball.y;
    [ball.x, ball.vx] = reflect(oldX + ball.vx * SPEC.fixedDt, ball.vx,
      0, SPEC.width - ball.width_cells);
    if (ball.vy < 0) [ball.y, ball.vy] = reflect(oldY + ball.vy * SPEC.fixedDt, ball.vy,
      0, SPEC.height - ball.height_cells);
    else ball.y = oldY + ball.vy * SPEC.fixedDt;
    const events = [];
    const oldBottom = oldY + ball.height_cells;
    const newBottom = ball.y + ball.height_cells;
    if (ball.vy > 0 && !ball.descendingEncounter && oldBottom < SPEC.paddleY && newBottom >= SPEC.paddleY) {
      const fraction = (SPEC.paddleY - oldBottom) / Math.max(EPSILON, newBottom - oldBottom);
      const contactX = oldX + (ball.x - oldX) * fraction;
      const contacts = ball.kind === 'large' ? [contactX, contactX + 1] : [contactX];
      const player = contacts.map(cell => paddleCovers(this.playerX, cell));
      const ai = contacts.map(cell => paddleCovers(this.aiX, cell));
      const caught = ball.kind === 'large'
        ? ((player[0] && ai[1]) || (player[1] && ai[0])) : player[0] || ai[0];
      ball.encounterIndex += 1;
      const encounterId = `${ball.ball_id}:${ball.encounterIndex}`;
      this.totalOpportunities += 1;
      if (caught) this.successfulOpportunities += 1;
      else { ball.pendingMiss = true; ball.pendingMissId = encounterId; }
      events.push({ event: 'encounter', encounter_id: encounterId, ball_id: ball.ball_id,
        kind: ball.kind, outcome: caught ? 'caught' : 'missed', contact_cells: contacts,
        player_coverage: player, ai_coverage: ai, frame: this.frame });
      ball.descendingEncounter = true;
      if (caught) { ball.pendingMiss = false; ball.pendingMissId = null; ball.y = SPEC.paddleY - ball.height_cells; ball.vy = -Math.abs(ball.vy); }
    }
    const lowerTop = SPEC.height - ball.height_cells;
    if (ball.vy > 0 && ball.y >= lowerTop - EPSILON) {
      ball.y = lowerTop;
      ball.vy = -Math.abs(ball.vy);
      ball.descendingEncounter = false;
      if (ball.pendingMiss) {
        const penalty = ball.kind === 'large' ? 3 : 1;
        this.missedBalls += penalty;
        this.missedByType[ball.kind] += penalty;
        events.push({ event: 'miss_scored', encounter_id: ball.pendingMissId, ball_id: ball.ball_id,
          kind: ball.kind, miss_penalty: penalty, frame: this.frame });
        ball.pendingMiss = false;
        ball.pendingMissId = null;
      }
      events.push({ event: 'bottom_bounce', ball_id: ball.ball_id, kind: ball.kind, frame: this.frame });
    }
    if (ball.vy < 0 && ball.y + ball.height_cells < SPEC.paddleY - EPSILON) ball.descendingEncounter = false;
    return events;
  }

  updateBubble(force = false) {
    if (this.group !== 'A' || this.task !== 1 || this.terminal) return;
    const decision = this.latestDecision;
    const signature = [decision.intentType, decision.targetBallId, decision.contactSide,
      decision.requiresPartner, decision.commitmentState, Math.round((decision.distance || 0) / 2)].join('|');
    if (!force && signature === this.bubbleSignature && this.timeSeconds - this.bubbleUpdatedAt < .75) return;
    this.bubbleSignature = signature;
    this.bubbleUpdatedAt = this.timeSeconds;
    if (!decision.targetBallId) {
      this.intentBubble = { text: '目前没有我能及时承担的来球，我先守住当前位置。', detail: '有新的可接球出现时，我会立刻重新分工。', target_ball_id: null };
      return;
    }
    const ball = { ball_id: decision.targetBallId, kind: decision.targetKind };
    if (decision.requiresPartner) {
      const own = decision.contactSide === 'left' ? '左侧' : '右侧';
      const other = decision.contactSide === 'left' ? '右侧' : '左侧';
      const plan = decision.smallFirstId
        ? (decision.smallFirstFeasible ? `先接小球${decision.smallFirstId}后仍来得及合拢。` : `若先接小球${decision.smallFirstId}会赶不上，所以优先接这颗大球。`)
        : '现在优先保证两侧同时到位。';
      this.intentBubble = { target_ball_id: ball.ball_id,
        text: `我负责${ballLabel(ball)}的${own}接球格；请你覆盖${other}。`,
        detail: `我离接球格约${cells(decision.distance)}格；你离另一侧约${cells(decision.playerDistance)}格。${plan}` };
      return;
    }
    const playerHint = decision.playerClosestId
      ? `你当前更接近${decision.playerClosestId}，我来处理${ball.ball_id}。`
      : `我离${ball.ball_id}的接球格约${cells(decision.distance)}格。`;
    this.intentBubble = { target_ball_id: ball.ball_id,
      text: `我去接${ballLabel(ball)}，它是我现在能及时覆盖的来球。`,
      detail: `${playerHint} 你可补另一侧来球。` };
  }

  step(playerAction) {
    if (this.paused || this.terminal) return;
    if (!this.latestDecision || this.frame - this.lastDecisionFrame >= SPEC.snapshotEvery) {
      this.latestDecision = this.chooseAI();
      this.lastDecisionFrame = this.frame;
      this.updateBubble();
    }
    const decision = clone(this.latestDecision);
    this.playerX = this.movePaddle(this.playerX, playerAction);
    this.aiX = this.movePaddle(this.aiX, decision.action);
    this.frame += 1;
    const events = this.balls.flatMap(ball => this.advanceBall(ball));
    if (events.some(event => event.event === 'encounter' && event.ball_id === this.commitment?.ball_id)) this.commitment = null;
    if (this.frame >= Math.round(SPEC.durationSeconds / SPEC.fixedDt)) this.phase = 'terminal';
    if (events.length > 0 || this.frame % SPEC.snapshotEvery === 0 || this.terminal) {
      const index = this.recordSnapshot(true, decision, events);
      events.filter(event => event.event === 'encounter' || event.event === 'miss_scored').forEach(event => {
        const outcome = event.event === 'encounter' ? (event.outcome === 'caught' ? '接住' : '漏接') : `漏接计${event.miss_penalty}`;
        this.reviewEvents.push({ index, frame: this.frame, ball_id: event.ball_id, label: `${event.ball_id} ${outcome}`, event: clone(event) });
      });
    }
  }

  snapshot(decision = this.latestDecision, events = []) {
    return { frame: this.frame, time_seconds: this.timeSeconds, terminal: this.terminal, phase: this.phase,
      player_x: this.playerX, ai_x: this.aiX, player_col: this.playerX, ai_col: this.aiX,
      balls: this.balls.map(ball => ({ ball_id: ball.ball_id, kind: ball.kind, x: ball.x, y: ball.y, vx: ball.vx, vy: ball.vy,
        width_cells: ball.width_cells, height_cells: ball.height_cells })),
      missed_balls: this.missedBalls, total_opportunities: this.totalOpportunities,
      successful_opportunities: this.successfulOpportunities, missed_by_type: clone(this.missedByType),
      decision: decision ? clone(decision) : null, events: clone(events) };
  }

  recordSnapshot(force = false, decision = this.latestDecision, events = []) {
    if (!force && this.frame % SPEC.snapshotEvery !== 0) return this.history.length - 1;
    this.history.push(this.snapshot(decision, events));
    return this.history.length - 1;
  }
}

function drawCourt(frame) {
  if (!frame) return;
  $('court').style.setProperty('--grid-columns', String(SPEC.width));
  $('court').style.setProperty('--grid-rows', String(SPEC.height));
  $('court').style.setProperty('--board-aspect', `${SPEC.width} / ${SPEC.height}`);
  $('court').style.setProperty('--paddle-width', `${(SPEC.paddleWidth / SPEC.width) * 100}%`);
  $('court').style.setProperty('--paddle-height', `${100 / SPEC.height}%`);
  const top = `${(SPEC.paddleY / SPEC.height) * 100}%`;
  $('playerPaddle').style.left = `${(frame.player_x / SPEC.width) * 100}%`;
  $('playerPaddle').style.top = top;
  $('aiPaddle').style.left = `${(frame.ai_x / SPEC.width) * 100}%`;
  $('aiPaddle').style.top = top;
  const layer = $('balls');
  const present = new Set();
  frame.balls.forEach(ball => {
    present.add(ball.ball_id);
    let node = layer.querySelector(`[data-ball-id="${ball.ball_id}"]`);
    if (!node) { node = document.createElement('i'); node.dataset.ballId = ball.ball_id; layer.appendChild(node); }
    const assigned = game?.group === 'A' && game.task === 1 && game.intentBubble?.target_ball_id === ball.ball_id;
    node.className = `ball ${ball.kind === 'large' ? 'large' : 'small'}${assigned ? ' assigned' : ''}`;
    node.textContent = ball.ball_id;
    node.style.left = `${(ball.x / SPEC.width) * 100}%`;
    node.style.top = `${(ball.y / SPEC.height) * 100}%`;
    node.style.setProperty('--ball-width', `${(ball.width_cells / SPEC.width) * 100}%`);
    node.style.setProperty('--ball-height', `${(ball.height_cells / SPEC.height) * 100}%`);
  });
  layer.querySelectorAll('[data-ball-id]').forEach(node => { if (!present.has(node.dataset.ballId)) node.remove(); });
}

function renderReviewEvents(activeIndex = null) {
  const container = $('reviewEvents');
  container.replaceChildren();
  if (!game?.reviewEvents.length) {
    const empty = document.createElement('div');
    empty.className = 'review-events-empty';
    empty.textContent = '暂无关键接球事件';
    container.appendChild(empty);
    return;
  }
  game.reviewEvents.forEach(event => {
    const button = document.createElement('button');
    button.className = 'review-event';
    if (event.index === activeIndex) button.classList.add('active');
    button.textContent = `第${event.frame}帧 · ${event.label}`;
    button.onclick = () => loadReview(event.index);
    container.appendChild(button);
  });
}

function render(frame = game?.snapshot()) {
  if (!game || !frame) return;
  $('task').textContent = String(game.task);
  $('time').textContent = Number(frame.time_seconds).toFixed(1);
  $('missed').textContent = String(frame.missed_balls);
  $('opportunities').textContent = String(frame.total_opportunities);
  drawCourt(frame);
  const showBubble = game.group === 'A' && game.task === 1 && !frame.terminal && game.intentBubble;
  $('intentBubble').hidden = !showBubble;
  if (showBubble) {
    $('intentText').textContent = game.intentBubble.text;
    $('intentDistance').textContent = game.intentBubble.detail;
    $('intentBubble').style.setProperty('--bubble-x', `${(frame.ai_x / SPEC.width) * 100}%`);
  }
  const reviewAvailable = game.group === 'A' && game.task === 1 && frame.terminal;
  $('review').hidden = !reviewAvailable;
  if (reviewAvailable) {
    const max = Math.max(0, game.history.length - 1);
    $('timeline').max = String(max);
    if (replayIndex === null || replayIndex > max) replayIndex = max;
    $('timeline').value = String(replayIndex);
    $('replayFrame').textContent = String(game.history[replayIndex].frame);
    renderReviewEvents(replayLocked ? replayIndex : null);
  }
  $('task2').hidden = !(game.task === 1 && frame.terminal);
  $('task2').disabled = !(game.task === 1 && frame.terminal);
  ['left', 'pause', 'right'].forEach(id => { $(id).hidden = frame.terminal; });
  $('pause').textContent = game.paused ? '继续' : '暂停';
  $('status').textContent = reviewAvailable ? 'Task 1 已结束，可以回放提问'
    : frame.terminal ? '本局结束，点击进入 Task 2'
      : game.paused ? '已暂停' : '本地离线连续运行中';
}

function loadReview(index) {
  if (!game) return;
  replayPlaying = false;
  if (replayTimer) clearInterval(replayTimer);
  replayLocked = true;
  replayIndex = Math.max(0, Math.min(game.history.length - 1, Number(index)));
  const frame = game.history[replayIndex];
  $('timeline').value = String(replayIndex);
  $('replayFrame').textContent = String(frame.frame);
  $('replayTime').textContent = Number(frame.time_seconds).toFixed(1);
  $('replayState').textContent = `双方位置：你第${Number(frame.player_x).toFixed(1)}列 · 机器人2第${Number(frame.ai_x).toFixed(1)}列`;
  drawCourt(frame);
  renderReviewEvents(replayIndex);
}

function answerForReview(frame, requestedBallId) {
  const decision = frame.decision;
  const ballId = requestedBallId || decision?.targetBallId;
  const ball = frame.balls.find(item => item.ball_id === ballId);
  const encounter = frame.events.find(event => event.event === 'encounter' && (!ballId || event.ball_id === ballId));
  if (encounter) {
    const penalty = encounter.kind === 'large' ? 3 : 1;
    return encounter.outcome === 'caught'
      ? `${ballLabel(encounter)}的接球格被两块板子正确覆盖，因此这次接住了。`
      : `${ballLabel(encounter)}的实际接球格没有被完整覆盖，因此这次漏接${penalty}次。`;
  }
  if (!decision?.targetBallId) return '这一帧没有机器人2能及时承担的来球，它保持当前位置观察下一次机会。';
  const target = { ball_id: decision.targetBallId, kind: decision.targetKind };
  if (decision.requiresPartner) {
    const own = decision.contactSide === 'left' ? '左侧' : '右侧';
    const other = decision.contactSide === 'left' ? '右侧' : '左侧';
    return `机器人2此时负责${ballLabel(target)}的${own}接球格，距离约${cells(decision.distance)}格；机器人1需要覆盖${other}接球格，距离约${cells(decision.playerDistance)}格。${decision.smallFirstId ? (decision.smallFirstFeasible ? `先处理${decision.smallFirstId}后仍可赶上。` : `先处理${decision.smallFirstId}会错过这颗大球，所以优先合拢。`) : ''}`;
  }
  const location = ball ? `该球当时在第${Number(ball.x).toFixed(1)}列附近。` : '';
  return `机器人2此时去接${ballLabel(target)}，距预计接球格约${cells(decision.distance)}格。${location}`;
}

function setAction(action) {
  if (!running || currentAction === action) return;
  currentAction = action;
}
function updateKeyboardAction() {
  if (heldKeys.has('left') && heldKeys.has('right')) return setAction('stay');
  if (heldKeys.has('left')) return setAction('left');
  if (heldKeys.has('right')) return setAction('right');
  setAction('stay');
}

function simulationTick() {
  if (running && game && !game.paused && !game.terminal && !replayLocked) {
    game.step(currentAction);
    if (game.terminal) {
      running = false;
      currentAction = 'stay';
      replayIndex = game.group === 'A' && game.task === 1 ? game.history.length - 1 : null;
    }
    render();
  }
}

$('start').onclick = () => {
  game = new OfflinePong({ group: $('group').value, participantId: $('participant').value, task: 1, seed: 260918 });
  running = true;
  currentAction = 'stay';
  replayLocked = false;
  replayIndex = null;
  $('setup').hidden = true;
  $('game').hidden = false;
  render();
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
window.addEventListener('blur', () => { heldKeys.clear(); currentAction = 'stay'; });
$('pause').onclick = () => { if (game && !game.terminal) { game.paused = !game.paused; render(); } };
$('task2').onclick = () => {
  if (!game?.terminal || game.task !== 1) return;
  game = new OfflinePong({ group: game.group, participantId: game.participantId, task: 2, seed: 260919 });
  running = true;
  currentAction = 'stay';
  replayPlaying = false;
  if (replayTimer) clearInterval(replayTimer);
  replayLocked = false;
  replayIndex = null;
  $('review').hidden = true;
  render();
};
$('timeline').oninput = event => loadReview(event.target.value);
$('prevFrame').onclick = () => loadReview((replayIndex ?? 0) - 1);
$('nextFrame').onclick = () => loadReview((replayIndex ?? 0) + 1);
$('playReplay').onclick = () => {
  if (!game?.history.length) return;
  replayPlaying = !replayPlaying;
  $('playReplay').textContent = replayPlaying ? '暂停回放' : '播放';
  if (!replayPlaying) { if (replayTimer) clearInterval(replayTimer); return; }
  replayTimer = setInterval(() => {
    const next = (replayIndex ?? 0) + 1;
    if (next >= game.history.length) {
      replayPlaying = false;
      clearInterval(replayTimer);
      $('playReplay').textContent = '播放';
      return;
    }
    loadReview(next);
  }, 100);
};
document.querySelectorAll('[data-question]').forEach(button => button.onclick = () => { $('question').value = button.dataset.question; $('ask').click(); });
$('ask').onclick = () => {
  if (!game || game.group !== 'A' || game.task !== 1 || !game.terminal) {
    $('answer').textContent = '当前阶段不提供回放问答。';
    return;
  }
  const frame = game.history[replayIndex ?? game.history.length - 1];
  $('answer').textContent = answerForReview(frame, $('ballSelect').value || null);
};

window.setInterval(simulationTick, 1000 / 60);
