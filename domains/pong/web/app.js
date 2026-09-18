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
let rangeSelected = false;
let nnModel = null;
let nnProgram = null;
let nnController = { controller_mode: 'pure_nn' };
let nnLoadError = null;
const completedRuns = [];

function validateProgramFeatures(program, featureNames) {
  if (!program?.root) return;
  const known = new Set(featureNames);
  const visit = node => {
    if (node.probabilities) return;
    if (!known.has(node.feature)) throw new Error(`解释程序特征版本不兼容：${node.feature}`);
    visit(node.left); visit(node.right);
  };
  visit(program.root);
}

async function loadFrozenNN() {
  nnLoadError = null;
  try {
    const response = await fetch('nn_model.json', { cache: 'no-store' });
    if (!response.ok) throw new Error(`模型文件不可用 (${response.status})`);
    const candidate = await response.json();
    if (candidate.format !== 'pong-browser-float32.v2' || candidate.actions?.join('|') !== 'left|right|stay') throw new Error('模型动作或特征版本不兼容；请导出 Pong v2 模型');
    if (candidate.signature?.domain_id !== 'pong' || candidate.signature?.version !== 'pong-continuous-24x14-v7'
      || !Array.isArray(candidate.signature?.feature_names)) {
      throw new Error('模型观察签名与当前 Pong v2 物理规则不兼容');
    }
    let candidateProgram = null;
    const programResponse = await fetch('program.json', { cache: 'no-store' });
    if (programResponse.ok) {
      candidateProgram = await programResponse.json();
      validateProgramFeatures(candidateProgram, candidate.signature.feature_names);
    }
    const controllerResponse = await fetch('controller_config.json', { cache: 'no-store' });
    nnController = controllerResponse.ok ? await controllerResponse.json() : { controller_mode: 'pure_nn' };
    if (!['pure_nn', 'hybrid', 'rule_only', 'coordinated'].includes(nnController.controller_mode)) throw new Error('不支持的 Pong 控制器模式');
    if (nnController.controller_mode === 'hybrid' && nnController.rule_version !== 'pong-limited-assist.v2.2') throw new Error('规则辅助版本不兼容');
    if (nnController.controller_mode === 'coordinated' && nnController.rule_version !== 'pong-coordinated.v2.3') throw new Error('规则协调版本不兼容');
    if (nnController.model_sha256 && nnController.model_sha256 !== candidate.model_sha256)
      throw new Error('控制器绑定的冻结 Actor 哈希与当前模型不一致');
    nnModel = candidate;
    nnProgram = candidateProgram;
  } catch (error) {
    nnModel = null; nnProgram = null; nnController = { controller_mode: 'pure_nn' }; nnLoadError = String(error.message || error);
  }
}

function tanh(value) { return Math.tanh(value); }
function nnForward(features) {
  const missing = nnModel.signature.feature_names.filter(name => !Object.hasOwn(features, name));
  if (missing.length) throw new Error(`浏览器缺少模型所需特征：${missing.slice(0, 3).join(', ')}`);
  let values = nnModel.signature.feature_names.map(name => Number(features[name]));
  nnModel.layers.forEach(layer => {
    values = layer.weight.map((row, index) => {
      const result = row.reduce((total, weight, column) => total + weight * values[column], Number(layer.bias[index]));
      return layer.activation === 'tanh' ? tanh(result) : result;
    });
  });
  const maximum = Math.max(...values); const exps = values.map(value => Math.exp(value - maximum)); const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(value => value / sum);
}

function executeProgram(features) {
  if (!nnProgram?.root) return null;
  let node = nnProgram.root; const trace = [];
  while (!node.probabilities) {
    if (!Object.hasOwn(features, node.feature)) {
      throw new Error(`浏览器缺少解释程序所需特征：${node.feature}`);
    }
    const observed = Number(features[node.feature]); const result = observed <= Number(node.threshold);
    trace.push({ feature: node.feature, threshold: node.threshold, observed, result }); node = result ? node.left : node.right;
  }
  return { probabilities: node.probabilities, trace };
}

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
    this.playerLastAction = 'stay';
    this.aiLastAction = 'stay';
    this.paused = false;
    this.phase = 'active';
    this.missedBalls = 0;
    this.totalOpportunities = 0;
    this.successfulOpportunities = 0;
    this.missedByType = { small: 0, large: 0 };
    this.commitment = null;
    this.hybridCommitment = null;
    this.coordinator = new PongCoordinator({ horizonSeconds: Number(nnController.horizon_seconds ?? 10),
      maxOpportunities: Number(nnController.max_opportunities ?? 3), switchGain: Number(nnController.switch_gain ?? 1.5) });
    this.constructorSpec = SPEC;
    this.paddleCoversAt = paddleCovers;
    this.latestDecision = null;
    this.lastDecisionFrame = -1;
    this.intentBubble = null;
    this.bubbleSignature = null;
    this.bubbleUpdatedAt = -Infinity;
    this.history = [];
    this.questionHistory = [];
    this.reviewEvents = [];
    this.balls = this.initialBalls(seed);
    this.controllerSource = nnModel && nnController.controller_mode !== 'rule_only'
      ? (nnController.controller_mode === 'coordinated' ? 'coordinated'
        : nnController.controller_mode === 'hybrid' ? 'hybrid' : 'frozen_nn') : 'rule_demo';
    this.latestDecision = this.chooseAI();
    this.lastDecisionFrame = this.frame;
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
      descendingEncounter: false, pendingMiss: false, pendingMissId: null, encounterIndex: 0, active: true,
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
    if (!ball.active || Math.abs(ball.vy) <= EPSILON) return null;
    const speedY = Math.abs(ball.vy);
    const targetTop = SPEC.paddleY - ball.height_cells;
    const lower = 0;
    const upper = SPEC.height - ball.height_cells;
    let y = Math.max(lower, Math.min(upper, ball.y));
    let elapsed = 0;
    if (!ball.descendingEncounter && ball.vy > 0 && y <= targetTop + EPSILON) {
      elapsed = Math.max(0, targetTop - y) / speedY;
    } else {
      if (ball.vy > 0) {
        elapsed = Math.max(0, upper - y) / speedY;
        y = upper;
      } else {
        elapsed = Math.max(0, y - lower) / speedY;
        y = lower;
      }
      if (y > lower + EPSILON) elapsed += (y - lower) / speedY;
      elapsed += targetTop / speedY;
    }
    const span = SPEC.width - ball.width_cells;
    let contactX = ball.x;
    if (span <= EPSILON) contactX = 0;
    else if (Math.abs(ball.vx) > EPSILON) {
      const phase = ball.vx > 0 ? ball.x : 2 * span - ball.x;
      const wrapped = (phase + Math.abs(ball.vx) * elapsed) % (2 * span);
      contactX = wrapped <= span ? wrapped : 2 * span - wrapped;
    }
    const updates = Math.max(1, Math.ceil(elapsed / SPEC.fixedDt - EPSILON));
    if (this.frame + updates > Math.round(SPEC.durationSeconds / SPEC.fixedDt)) return null;
    return {
      ball_id: ball.ball_id, kind: ball.kind,
      opportunity_id: ball.ball_id + ':' + (ball.encounterIndex + 1),
      time_until_contact: elapsed,
      contact_cells: ball.kind === 'large' ? [contactX, contactX + 1] : [contactX],
    };
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

  featureMap(agent = 'ai') {
    const own = agent === 'ai' ? this.aiX : this.playerX;
    const other = agent === 'ai' ? this.playerX : this.aiX;
    const ownAction = agent === 'ai' ? this.aiLastAction : this.playerLastAction;
    const otherAction = agent === 'ai' ? this.playerLastAction : this.aiLastAction;
    const actionValue = { left: -1, stay: 0, right: 1 };
    const values = { 'self.paddle_x': own / SPEC.width, 'other.paddle_x': other / SPEC.width,
      'time.progress': Math.min(1, this.frame / Math.round(SPEC.durationSeconds / SPEC.fixedDt)),
      'self.last_action': actionValue[ownAction], 'other.last_action': actionValue[otherAction] };
    this.balls.forEach(ball => {
      const prefix = 'ball.' + ball.ball_id;
      let prediction = ball.active && !ball.descendingEncounter ? this.predictContact(ball) : null;
      const currentOpportunity = Boolean(ball.active && !ball.descendingEncounter && prediction);
      if (!currentOpportunity) prediction = null;
      const contactLeft = prediction ? prediction.contact_cells[0] : 0;
      const contactRight = prediction && ball.kind === 'large' ? prediction.contact_cells[prediction.contact_cells.length - 1] : contactLeft;
      const ownTarget = prediction ? paddleTargetLeft(own, contactLeft) : own;
      const otherTarget = prediction ? paddleTargetLeft(other, contactLeft) : other;
      const ownDistance = prediction ? Math.abs(ownTarget - own) : 0;
      const otherDistance = prediction ? Math.abs(otherTarget - other) : 0;
      const eta = prediction ? prediction.time_until_contact : 0;
      const ownReachable = Boolean(prediction && travelSeconds(ownDistance) <= eta + EPSILON);
      const otherReachable = Boolean(prediction && travelSeconds(otherDistance) <= eta + EPSILON);
      const ownCovered = Boolean(prediction && paddleCovers(own, contactLeft));
      const otherCovered = Boolean(prediction && paddleCovers(other, contactLeft));
      let selfLeftDistance = 0; let otherRightDistance = 0; let selfRightDistance = 0; let otherLeftDistance = 0;
      let selfLeftAssignmentReachable = false; let selfRightAssignmentReachable = false;
      if (prediction && ball.kind === 'large') {
        const [left, right] = prediction.contact_cells;
        selfLeftDistance = Math.abs(paddleTargetLeft(own, left) - own);
        otherRightDistance = Math.abs(paddleTargetLeft(other, right) - other);
        selfRightDistance = Math.abs(paddleTargetLeft(own, right) - own);
        otherLeftDistance = Math.abs(paddleTargetLeft(other, left) - other);
        selfLeftAssignmentReachable = travelSeconds(selfLeftDistance) <= eta + EPSILON
          && travelSeconds(otherRightDistance) <= eta + EPSILON;
        selfRightAssignmentReachable = travelSeconds(selfRightDistance) <= eta + EPSILON
          && travelSeconds(otherLeftDistance) <= eta + EPSILON;
      }
      values[prefix + '.x'] = ball.x / SPEC.width; values[prefix + '.y'] = ball.y / SPEC.height;
      values[prefix + '.vx'] = ball.vx / SPEC.ballSpeed; values[prefix + '.vy'] = ball.vy / SPEC.ballSpeed;
      values[prefix + '.large'] = ball.kind === 'large' ? 1 : 0; values[prefix + '.descending'] = ball.vy > 0 ? 1 : 0;
      values[prefix + '.encounters'] = ball.encounterIndex;
      values[prefix + '.active'] = ball.active ? 1 : 0;
      values[prefix + '.opportunity_valid'] = currentOpportunity ? 1 : 0;
      values[prefix + '.contact_left'] = contactLeft / SPEC.width; values[prefix + '.contact_right'] = contactRight / SPEC.width;
      values[prefix + '.time_to_contact'] = Math.min(1, eta / SPEC.durationSeconds);
      values[prefix + '.self_distance'] = ownDistance / SPEC.width; values[prefix + '.other_distance'] = otherDistance / SPEC.width;
      values[prefix + '.self_reachable'] = ownReachable ? 1 : 0; values[prefix + '.other_reachable'] = otherReachable ? 1 : 0;
      values[prefix + '.self_covers'] = ownCovered ? 1 : 0; values[prefix + '.other_covers'] = otherCovered ? 1 : 0;
      values[prefix + '.self_left_distance'] = selfLeftDistance / SPEC.width;
      values[prefix + '.other_right_distance'] = otherRightDistance / SPEC.width;
      values[prefix + '.self_right_distance'] = selfRightDistance / SPEC.width;
      values[prefix + '.other_left_distance'] = otherLeftDistance / SPEC.width;
      values[prefix + '.self_left_assignment_reachable'] = selfLeftAssignmentReachable ? 1 : 0;
      values[prefix + '.self_right_assignment_reachable'] = selfRightAssignmentReachable ? 1 : 0;
    });
    values['role.is_ai'] = agent === 'ai' ? 1 : 0;
    return values;
  }

  chooseAI() {
    if (!nnModel || nnController.controller_mode === 'rule_only') return this.chooseRuleAI();
    const features = this.featureMap('ai'); const probabilities = nnForward(features);
    const selected = probabilities.indexOf(Math.max(...probabilities));
    const context = this.balls.map(ball => this.candidateFor(ball)).filter(Boolean)
      .sort((left, right) => left.prediction.time_until_contact - right.prediction.time_until_contact)[0] || null;
    const trace = executeProgram(features);
    const proposal = nnModel.actions[selected];
    if (nnController.controller_mode === 'coordinated') {
      const coordinated = this.coordinator.choose(this, proposal);
      const evidence = coordinated.evidence;
      return { ...coordinated.base, action: coordinated.action, controllerSource: 'coordinated',
        decisionId: `${this.task}:${this.frame}`, decisionFrame: this.frame,
        probabilities, nnProposedAction: proposal, controllerSelectedAction: coordinated.action,
        intervened: proposal !== coordinated.action, interventionReason: coordinated.reason,
        ruleVersion: coordinated.ruleVersion, planEvidence: evidence,
        actorHash: nnModel.model_sha256, programHash: nnController.program_sha256 || null,
        programTrace: trace?.trace || [], programProbabilities: trace?.probabilities || null,
        intentType: 'coordinated', targetBallId: evidence.target_ball_id,
        targetKind: evidence.target_kind, targetX: evidence.target_x,
        contactSide: evidence.contact_side, distance: evidence.self_distance,
        playerDistance: evidence.partner_distance,
        requiresPartner: evidence.requires_partner, opportunityId: evidence.opportunity_id,
        commitmentState: evidence.holding ? 'holding' : 'approaching' };
    }
    const assist = nnController.controller_mode === 'hybrid' ? this.chooseHybrid(proposal)
      : { action: proposal, reason: null, evidence: {}, ruleVersion: null };
    return { action: assist.action, controllerSource: nnController.controller_mode === 'hybrid' ? 'hybrid' : 'frozen_nn', probabilities,
      nnProposedAction: proposal, controllerSelectedAction: assist.action,
      intervened: proposal !== assist.action, interventionReason: assist.reason,
      ruleVersion: assist.ruleVersion, candidateEvidence: assist.evidence,
      actorHash: nnModel.model_sha256, programHash: nnController.program_sha256 || null,
      programTrace: trace?.trace || [], programProbabilities: trace?.probabilities || null,
      intentType: 'nn_action', targetBallId: context?.prediction.ball_id || null, targetKind: context?.prediction.kind || null,
      targetX: context?.targetX || this.aiX, contactSide: context?.contactSide || null, distance: context?.aiDistance || null,
      playerDistance: context?.playerDistance || null, requiresPartner: Boolean(context?.requiresPartner),
      opportunityId: context?.prediction.opportunity_id || null, commitmentState: 'nn', candidates: [] };
  }

  chooseHybrid(proposal) {
    const own = this.aiX; const other = this.playerX;
    const maximum = SPEC.width - SPEC.paddleWidth;
    const duration = SPEC.snapshotEvery * SPEC.fixedDt;
    const step = SPEC.paddleSpeed * duration;
    const window = Number(nnController.urgent_window_seconds ?? .6);
    const tradeoffWindow = Math.max(window, Number(nnController.tradeoff_window_seconds ?? 1.2));
    const margin = Number(nnController.minimum_recovery_margin_seconds ?? .1);
    const candidates = []; const valid = new Set();
    this.balls.forEach(ball => {
      if (!ball.active || ball.descendingEncounter) return;
      const prediction = this.predictContact(ball);
      if (!prediction) return;
      valid.add(prediction.opportunity_id);
      if (prediction.time_until_contact > tradeoffWindow) return;
      const contact = prediction.contact_cells;
      let side; let target;
      if (contact.length === 1) {
        side = 'single'; target = contact[0];
        if (paddleCovers(other, target)) return;
      } else {
        const otherLeft = paddleCovers(other, contact[0]);
        const otherRight = paddleCovers(other, contact[1]);
        if (otherLeft && !otherRight) { side = 'right'; target = contact[1]; }
        else if (otherRight && !otherLeft) { side = 'left'; target = contact[0]; }
        else return;
      }
      const targetLeft = paddleTargetLeft(own, target);
      const slack = prediction.time_until_contact - Math.abs(targetLeft - own) / SPEC.paddleSpeed;
      if (slack < margin && !paddleCovers(own, target)) return;
      candidates.push({ ball_id: ball.ball_id, opportunity_id: prediction.opportunity_id,
        ball_kind: ball.kind, side, target_cell: target,
        time_until_contact: prediction.time_until_contact, target_left: targetLeft,
        slack, weight: ball.kind === 'large' ? 3 : 1 });
    });
    if (this.hybridCommitment && !valid.has(this.hybridCommitment.opportunity_id)) this.hybridCommitment = null;
    if (this.hybridCommitment && !candidates.some(c => c.opportunity_id === this.hybridCommitment.opportunity_id
      && c.side === this.hybridCommitment.side)) this.hybridCommitment = null;
    const ruleVersion = 'pong-limited-assist.v2.2';
    if ((own <= 1e-6 && proposal === 'left') || (own >= maximum - 1e-6 && proposal === 'right')) {
      return { action: 'stay', reason: 'boundary_no_motion', ruleVersion,
        evidence: { agent: 'ai', position: own, boundary: proposal === 'left' ? 0 : maximum,
          excluded_future_player_action: true } };
    }
    if (!candidates.length) return { action: proposal, reason: null, ruleVersion,
      evidence: { agent: 'ai', reason: 'no_proven_near_contact_correction' } };
    if (!candidates.some(c => c.time_until_contact <= window)) return { action: proposal, reason: null, ruleVersion,
      evidence: { agent: 'ai', reason: 'no_urgent_verified_intervention' } };
    candidates.sort((a, b) => b.weight - a.weight || a.time_until_contact - b.time_until_contact
      || a.ball_id.localeCompare(b.ball_id));
    const actions = ['left', 'right', 'stay'];
    const positions = Object.fromEntries(actions.map(action => [action,
      Math.max(0, Math.min(maximum, own + (action === 'right' ? step : action === 'left' ? -step : 0)))]));
    const protectedBy = (action, candidate) => {
      const contactTime = candidate.time_until_contact;
      if (contactTime <= duration + 1e-9) {
        const delta = SPEC.paddleSpeed * Math.max(0, contactTime);
        const atContact = Math.max(0, Math.min(maximum, own + (action === 'right' ? delta : action === 'left' ? -delta : 0)));
        return paddleCovers(atContact, candidate.target_cell);
      }
      const position = positions[action];
      const remaining = contactTime - duration;
      const residual = Math.abs(paddleTargetLeft(position, candidate.target_cell) - position);
      return residual / SPEC.paddleSpeed <= remaining - margin + 1e-6;
    };
    const ordered = candidates.slice().sort((a, b) => a.time_until_contact - b.time_until_contact
      || b.weight - a.weight || a.ball_id.localeCompare(b.ball_id));
    const bestSequence = action => {
      const visit = (index, position, availableAt, chosen) => {
        if (index === ordered.length) return { score: chosen.reduce((sum, c) => sum + c.weight, 0), chosen };
        const skipped = visit(index + 1, position, availableAt, chosen);
        const candidate = ordered[index];
        const contactAt = candidate.time_until_contact;
        let nextPosition; let nextTime;
        if (contactAt <= duration + 1e-9) {
          if (!protectedBy(action, candidate)) return skipped;
          nextPosition = position; nextTime = availableAt;
        } else {
          nextPosition = paddleTargetLeft(position, candidate.target_cell);
          const arrival = availableAt + Math.abs(nextPosition - position) / SPEC.paddleSpeed;
          if (arrival > contactAt - margin + 1e-6) return skipped;
          nextTime = contactAt;
        }
        const taken = visit(index + 1, nextPosition, nextTime, [...chosen, candidate]);
        return taken.score > skipped.score ? taken : skipped;
      };
      return visit(0, positions[action], duration, []);
    };
    const plans = Object.fromEntries(actions.map(action => [action, bestSequence(action)]));
    const scores = Object.fromEntries(actions.map(action => [action, plans[action].score]));
    const alternatives = actions.filter(action => scores[action] > scores[proposal]);
    if (!alternatives.length) return { action: proposal, reason: null, ruleVersion,
      evidence: { agent: 'ai', candidate_scores: scores, reason: 'proposal_preserves_feasible_opportunities' } };
    alternatives.sort((a, b) => scores[b] - scores[a]
      || candidates.reduce((sum, c) => sum + Math.abs(positions[a] - c.target_left), 0)
       - candidates.reduce((sum, c) => sum + Math.abs(positions[b] - c.target_left), 0)
      || Number(a !== 'stay') - Number(b !== 'stay'));
    const chosen = alternatives[0];
    const priorIds = new Set(plans[proposal].chosen.map(c => c.opportunity_id));
    const saved = plans[chosen].chosen.find(c => !priorIds.has(c.opportunity_id));
    if (!saved) return { action: proposal, reason: null, ruleVersion,
      evidence: { agent: 'ai', reason: 'ambiguous_correction' } };
    let reason = saved.ball_kind === 'large' ? 'large_side_rescue' : 'near_miss_rescue';
    if (paddleCovers(own, saved.target_cell) && chosen === 'stay') reason = 'protect_existing_coverage';
    if (saved.ball_kind === 'large') this.hybridCommitment = { ball_id: saved.ball_id,
      opportunity_id: saved.opportunity_id, side: saved.side,
      created_frame: this.frame, last_validated_frame: this.frame, release_reason: null };
    return { action: chosen, reason, ruleVersion,
      evidence: { agent: 'ai', candidate_scores: scores, target: saved,
        other_position: other, decision_seconds: duration, excluded_future_player_action: true } };
  }

  chooseRuleAI() {
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
    if (decision.controllerSource === 'coordinated') {
      const evidence = decision.planEvidence || {};
      const signature = [evidence.opportunity_id, evidence.contact_side, evidence.holding,
        evidence.partner_status, evidence.reason].join('|');
      if (!force && signature === this.bubbleSignature) return;
      this.bubbleSignature = signature;
      this.bubbleUpdatedAt = this.timeSeconds;
      this.intentBubble = PongExplanations.bubble(decision);
      return;
    }
    const signature = [decision.intentType, decision.targetBallId, decision.contactSide,
      decision.requiresPartner, decision.commitmentState, Math.round((decision.distance || 0) / 2)].join('|');
    if (!force && signature === this.bubbleSignature && this.timeSeconds - this.bubbleUpdatedAt < .75) return;
    this.bubbleSignature = signature;
    this.bubbleUpdatedAt = this.timeSeconds;
    if (decision.controllerSource === 'hybrid' && decision.intervened) {
      const direction = { left: '向左', right: '向右', stay: '暂时停留' };
      const target = decision.candidateEvidence?.target;
      const reason = decision.interventionReason === 'boundary_no_motion'
        ? '继续朝边界移动不会产生位移。'
        : target ? `临近${target.ball_id}的本次接球，原动作可能失去当前可达的覆盖。`
          : '有限规则发现当前动作会失去已验证的接球机会。';
      this.intentBubble = { target_ball_id: target?.ball_id || null,
        text: `NN建议${direction[decision.nnProposedAction]}；辅助改为${direction[decision.action]}。`,
        detail: `${reason}这是规则辅助的判断，不是NN的内在意图。` };
      return;
    }
    if (decision.controllerSource === 'frozen_nn' || decision.controllerSource === 'hybrid') {
      const action = { left: '向左', right: '向右', stay: '停留' }[decision.action];
      const confidence = Math.round(100 * Math.max(...decision.probabilities));
      const contextual = decision.targetBallId ? `当前最临近的接球机会是${decision.targetBallId}${decision.requiresPartner ? '，需要双方覆盖接触格' : ''}。` : '当前没有即将到达接球线的球。';
      this.intentBubble = { target_ball_id: decision.targetBallId, text: `机器人2${action}（NN该动作概率${confidence}%）。`, detail: `${contextual} 球的关联来自公开几何位置，不代表NN显式选择了该球；程序路径只用于近似解释。` };
      return;
    }
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
    const beforeStep = { frame: this.frame, balls: clone(this.balls), player_x: this.playerX,
      ai_x: this.aiX, player_last_action: this.playerLastAction, ai_last_action: this.aiLastAction,
      commitment: clone(this.commitment), hybrid_commitment: clone(this.hybridCommitment),
      coordinator_small_commitment: this.coordinator.smallCommitment,
      missed_balls: this.missedBalls, missed_by_type: clone(this.missedByType),
      total_opportunities: this.totalOpportunities,
      successful_opportunities: this.successfulOpportunities };
    this.playerX = this.movePaddle(this.playerX, playerAction);
    const oldAiX = this.aiX;
    this.aiX = this.movePaddle(this.aiX, decision.action);
    decision.environmentExecutedAction = Math.abs(this.aiX - oldAiX) <= EPSILON ? 'stay' : decision.action;
    decision.environmentActionResult = Math.abs(this.aiX - oldAiX) <= EPSILON && decision.action !== 'stay' ? 'boundary_no_motion' : 'executed';
    this.playerLastAction = playerAction;
    this.aiLastAction = decision.action;
    this.frame += 1;
    const events = this.balls.flatMap(ball => this.advanceBall(ball));
    if (events.some(event => event.event === 'encounter' && event.ball_id === this.commitment?.ball_id)) this.commitment = null;
    if (this.frame >= Math.round(SPEC.durationSeconds / SPEC.fixedDt)) this.phase = 'terminal';
    if (true) {
      const index = this.recordSnapshot(true, decision, events, beforeStep);
      events.filter(event => event.event === 'encounter' || event.event === 'miss_scored').forEach(event => {
        const outcome = event.event === 'encounter' ? (event.outcome === 'caught' ? '接住' : '漏接') : `漏接计${event.miss_penalty}`;
        this.reviewEvents.push({ index, frame: this.frame, ball_id: event.ball_id, label: `${event.ball_id} ${outcome}`, event: clone(event) });
      });
    }
  }

  snapshot(decision = this.latestDecision, events = [], beforeStep = null) {
    return { frame: this.frame, time_seconds: this.timeSeconds, terminal: this.terminal, phase: this.phase,
      player_x: this.playerX, ai_x: this.aiX, player_col: this.playerX, ai_col: this.aiX,
      balls: this.balls.map(ball => ({ ball_id: ball.ball_id, kind: ball.kind, x: ball.x, y: ball.y, vx: ball.vx, vy: ball.vy,
        width_cells: ball.width_cells, height_cells: ball.height_cells })),
      missed_balls: this.missedBalls, total_opportunities: this.totalOpportunities,
      successful_opportunities: this.successfulOpportunities, missed_by_type: clone(this.missedByType),
      decision: decision ? clone(decision) : null, events: clone(events),
      decision_pre_state: beforeStep ? clone(beforeStep) : null,
      physical_state: { balls: clone(this.balls), player_x: this.playerX, ai_x: this.aiX,
        player_last_action: this.playerLastAction, ai_last_action: this.aiLastAction,
        commitment: clone(this.commitment), hybrid_commitment: clone(this.hybridCommitment),
        coordinator_small_commitment: this.coordinator.smallCommitment } };
  }

  recordSnapshot(force = false, decision = this.latestDecision, events = [], beforeStep = null) {
    if (!force && this.frame % SPEC.snapshotEvery !== 0) return this.history.length - 1;
    this.history.push(this.snapshot(decision, events, beforeStep));
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
  const reviewAvailable = game.group === 'A' && game.task === 1 && (frame.terminal || game.paused);
  $('review').hidden = !reviewAvailable;
  if (reviewAvailable) {
    const max = Math.max(0, game.history.length - 1);
    $('timeline').max = String(max);
    if (replayIndex === null || replayIndex > max) replayIndex = max;
    $('timeline').value = String(replayIndex); $('rangeStart').max = String(max);
    $('rangeStart').value = rangeSelected ? String(Math.min(Number($('rangeStart').value), replayIndex)) : String(replayIndex);
    $('replayFrame').textContent = String(game.history[replayIndex].frame);
    renderTechnical(game.history[replayIndex]);
    renderReviewEvents(replayLocked ? replayIndex : null);
  }
  $('task2').hidden = !(game.task === 1 && frame.terminal);
  $('task2').disabled = !(game.task === 1 && frame.terminal);
  ['left', 'pause', 'right'].forEach(id => { $(id).hidden = frame.terminal; });
  $('pause').textContent = game.paused ? '继续' : '暂停';
  $('reviewTitle').textContent = frame.terminal ? 'Task 1 回放与提问' : '已暂停：查看并提问';
  const controllerStatus = game.group !== 'A' || game.task !== 1 ? '游戏在浏览器本地运行'
    : game.controllerSource === 'coordinated' ? '冻结 NN 建议＋规则协调接球，在浏览器本地运行'
    : game.controllerSource === 'hybrid' ? '冻结 NN＋有限规则辅助在浏览器本地运行'
    : game.controllerSource === 'frozen_nn' ? '冻结 NN 本地运行中'
    : nnLoadError ? `规则比较模式（NN 未加载：${nnLoadError}）` : '规则比较模式（未加载 NN 包）';
  $('status').textContent = reviewAvailable ? (frame.terminal ? 'Task 1 已结束，可以回放提问' : '已暂停：A组可查看当前帧或一段过程')
    : frame.terminal ? (game.task === 2 ? 'Task 2 已结束，请填写问卷' : '本局结束，点击进入 Task 2')
      : game.paused ? '已暂停' : controllerStatus;
}

function renderTechnical(frame) {
  const decision = frame?.decision || {};
  const source = decision.controllerSource === 'coordinated' ? '冻结NN建议＋规则协调'
    : decision.controllerSource || '未知';
  $('technicalEvidenceText').textContent = JSON.stringify({
    decision_id: decision.decisionId || null,
    decision_before_frame: decision.decisionFrame ?? decision.planEvidence?.decision_frame ?? null,
    executed_frame: frame?.frame ?? null,
    controller_source: source,
    actor_sha256: decision.actorHash || null,
    extracted_program: decision.programHash ? '存在匹配程序，可作为NN建议的近似证据' : '没有匹配的抽取程序，不提供NN内部归因',
    nn_proposed_action: decision.nnProposedAction || null,
    controller_selected_action: decision.controllerSelectedAction || decision.action || null,
    environment_executed_action: decision.environmentExecutedAction || null,
    plan: decision.planEvidence || null,
    events_after_action: frame?.events || [],
  }, null, 2);
}

function loadReview(index) {
  if (!game) return;
  replayPlaying = false;
  if (replayTimer) clearInterval(replayTimer);
  replayLocked = true;
  replayIndex = Math.max(0, Math.min(game.history.length - 1, Number(index)));
  if (!rangeSelected) $('rangeStart').value = String(replayIndex);
  else if (Number($('rangeStart').value) > replayIndex) $('rangeStart').value = String(replayIndex);
  const frame = game.history[replayIndex];
  renderTechnical(frame);
  $('timeline').value = String(replayIndex);
  $('replayFrame').textContent = String(frame.frame);
  $('replayTime').textContent = Number(frame.time_seconds).toFixed(1);
  $('replayState').textContent = `双方位置：你第${Number(frame.player_x).toFixed(1)}列 · 机器人2第${Number(frame.ai_x).toFixed(1)}列`;
  drawCourt(frame);
  renderReviewEvents(replayIndex);
}

function counterfactualReview(frame, question, ballId, index) {
  if (!/如果|假如|要是/.test(question)) return null;
  const action = /不动|不移动|停留|等待/.test(question) ? 'stay'
    : /向左|左移/.test(question) ? 'left' : /向右|右移/.test(question) ? 'right' : null;
  if (!action) return '请说明假设你持续停留、向左还是向右；系统不会猜测你未执行的动作。';
  if (!frame.decision_pre_state || frame.terminal) return '选定帧缺少行动前快照，无法可靠模拟。';
  const replay = new OfflinePong({ group: 'B', participantId: 'counterfactual', task: 1, seed: game.seed });
  const state = frame.decision_pre_state;
  replay.frame = state.frame;
  replay.playerX = state.player_x; replay.aiX = state.ai_x;
  replay.playerLastAction = state.player_last_action; replay.aiLastAction = state.ai_last_action;
  replay.balls = clone(state.balls);
  replay.commitment = clone(state.commitment);
  replay.hybridCommitment = clone(state.hybrid_commitment);
  replay.coordinator.smallCommitment = state.coordinator_small_commitment;
  replay.missedBalls = state.missed_balls;
  replay.missedByType = clone(state.missed_by_type);
  replay.totalOpportunities = state.total_opportunities;
  replay.successfulOpportunities = state.successful_opportunities;
  // Preserve the already submitted decision for this physics frame. Replan
  // only when the ordinary decision interval ends in the simulated future.
  replay.latestDecision = clone(frame.decision);
  replay.lastDecisionFrame = frame.decision?.decisionFrame ?? state.frame;
  replay.history = [];
  const target = ballId || frame.decision?.planEvidence?.target_ball_id || null;
  let simulated = null;
  for (let count = 0; count < 300 && !replay.terminal; count += 1) {
    replay.step(action);
    simulated = replay.history.at(-1)?.events?.find(event => event.event === 'encounter'
      && (!target || event.ball_id === target));
    if (simulated) break;
  }
  if (!simulated) return `假设你持续${action === 'stay' ? '停留' : action === 'left' ? '向左' : '向右'}，短回放内还没有${target || '目标球'}的接球结算；无法据此判断是否会接住。`;
  const outcome = simulated.outcome === 'caught' ? '接住' : '漏接';
  const actual = game.history.slice(index).flatMap(item => item.events || [])
    .find(event => event.event === 'encounter' && event.encounter_id === simulated.encounter_id);
  const actualText = actual ? `真实回合中该次接球${actual.outcome === 'caught' ? '接住' : '漏接'}。` : '';
  return `假设你从选定帧起持续${action === 'stay' ? '停留' : action === 'left' ? '向左' : '向右'}，并让机器人2按相同冻结控制器继续行动，短回放中${simulated.ball_id}${outcome}。${actualText}这只是明确输入条件下的模拟结果。`;
}

function answerForReview(frame, requestedBallId, question = '', index = null) {
  const decision = frame.decision;
  if (decision?.controllerSource === 'coordinated') {
    const selectedIndex = index ?? game.history.indexOf(frame);
    const target = (String(question).toUpperCase().match(/(?:A[123]|B[12])/) || [requestedBallId])[0];
    const counterfactual = counterfactualReview(frame, question, target, selectedIndex);
    return PongExplanations.answer(question, frame, game.history, selectedIndex, requestedBallId, counterfactual);
  }
  const ballId = requestedBallId || decision?.targetBallId;
  const ball = frame.balls.find(item => item.ball_id === ballId);
  const encounter = frame.events.find(event => event.event === 'encounter' && (!ballId || event.ball_id === ballId));
  if (encounter) {
    const penalty = encounter.kind === 'large' ? 3 : 1;
    return encounter.outcome === 'caught'
      ? `${ballLabel(encounter)}的接球格被两块板子正确覆盖，因此这次接住了。`
      : `${ballLabel(encounter)}的实际接球格没有被完整覆盖，因此这次漏接${penalty}次。`;
  }
  if (decision?.controllerSource === 'frozen_nn' || decision?.controllerSource === 'hybrid') {
    const choices = nnModel.actions.map((action, index) => `${{ left: '左', right: '右', stay: '停留' }[action]} ${Math.round(decision.probabilities[index] * 100)}%`).join('、');
    const trace = decision.programTrace?.length ? `近似程序此次参考了${decision.programTrace.slice(0, 2).map(item => item.feature).join('、')}。` : '这次没有可用的程序路径，因此不能可靠归因到某个单一条件。';
    const outcome = decision.targetBallId ? `当时最近的接球机会是${decision.targetBallId}${decision.requiresPartner ? '，它需要两块球拍共同覆盖。' : '。'}` : '';
    const origin = decision.intervened ? `NN原本建议${decision.nnProposedAction}；有限规则因${decision.interventionReason === 'boundary_no_motion' ? '边界方向没有位移' : `${decision.candidateEvidence?.target?.ball_id || '临近来球'}的可达覆盖风险`}改为${decision.action}。这不是NN内部意图。` : '';
    return `冻结NN在这一决策中的动作分布为：${choices}。${origin}实际提交${{ left: '向左', right: '向右', stay: '停留' }[decision.action]}，环境结果为${decision.environmentExecutedAction || decision.action}。${outcome}${trace}`;
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

function advanceSimulation() {
  if (running && game && !game.paused && !game.terminal && !replayLocked) {
    game.step(currentAction);
    if (game.terminal) {
      running = false;
      currentAction = 'stay';
      replayIndex = game.group === 'A' && game.task === 1 ? game.history.length - 1 : null;
      completedRuns.push(game.snapshot());
      if (game.task === 2) showQuestionnaire();
      else if (game.group === 'B') beginTask2();
    }
    return true;
  }
  return false;
}

function simulationTick() {
  if (advanceSimulation()) render();
}

function answerForRange(start, end, question) {
  const frames = game.history.slice(start, end + 1); const decisions = frames.map(item => item.decision).filter(Boolean);
  if (decisions[0]?.controllerSource === 'coordinated') return PongExplanations.answerRange(question, frames);
  const changes = decisions.filter((item, index) => index === 0 || item.action !== decisions[index - 1].action).length;
  const events = frames.flatMap(item => item.events || []).filter(item => item.event === 'encounter' || item.event === 'miss_scored');
  const source = decisions[0]?.controllerSource === 'hybrid' ? '冻结NN与有限规则辅助' : decisions[0]?.controllerSource === 'frozen_nn' ? '冻结NN' : '规则比较控制器';
  const result = events.length ? `期间发生：${events.map(item => `${item.ball_id}${item.event === 'miss_scored' ? `漏接计${item.miss_penalty}` : item.outcome === 'caught' ? '接住' : '未覆盖'}`).join('、')}。` : '期间没有接球结算。';
  return `第${frames[0].frame}至第${frames.at(-1).frame}帧中，机器人2使用${source}，动作改变${changes}次。${result} 这段回答只描述已记录的动作和结果；若要解释单次选择，请把起点拖到终点后提问。`;
}

function showQuestionnaire() {
  if (!game || game.task !== 2 || !game.terminal) return;
  $('questionnaire').hidden = false;
  const line = completedRuns.map((run, index) => `Task ${index + 1}：小球漏接${run.missed_by_type.small || 0}，大球漏接${(run.missed_by_type.large || 0) / 3}，加权漏接${run.missed_balls}，合作接住${run.successful_opportunities}`).join('； ');
  $('scoreSummary').textContent = line;
  const shared = ['我能理解机器人2正在做什么。','我能预测机器人2接下来的动作。','我知道自己和机器人2应该如何分工接球。','我能与机器人2配合接住大球。','机器人2的行为符合我的预期。'];
  const extra = game.group === 'A' ? ['游戏中的气泡清楚地说明了机器人2的行为。','暂停后的问答帮助我理解了所选片段。','解释帮助我判断自己应该去接哪个球。'] : [];
  const holder = $('surveyItems'); holder.replaceChildren();
  [...shared, ...extra].forEach((prompt, index) => {
    const label = document.createElement('fieldset'); label.innerHTML = `<legend>${prompt}</legend>`;
    for (let value = 1; value <= 5; value += 1) label.insertAdjacentHTML('beforeend', `<label><input required type="radio" name="q${index}" value="${value}">${value}</label>`);
    if (game.group === 'A' && index === shared.length + 1) label.insertAdjacentHTML('beforeend', `<label><input type="radio" name="q${index}" value="na">未使用／不适用</label>`);
    holder.appendChild(label);
  });
}

$('start').onclick = async () => {
  await loadFrozenNN();
  if (!nnModel) {
    $('setupError').textContent = `无法启动：${nnLoadError || '冻结模型不可用'}。请检查本地导出包。`;
    return;
  }
  $('setupError').textContent = '';
  completedRuns.length = 0;
  try {
    game = new OfflinePong({ group: $('group').value, participantId: $('participant').value, task: 1, seed: 260918 });
  } catch (error) {
    $('setupError').textContent = `无法启动：${String(error.message || error)}`;
    return;
  }
  running = true;
  currentAction = 'stay';
  replayLocked = false;
  replayIndex = null;
  rangeSelected = false;
  $('setup').hidden = true;
  $('game').hidden = false;
  $('questionnaire').hidden = true; $('completed').hidden = true;
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
window.addEventListener('blur', () => { heldKeys.clear(); currentAction = 'stay'; if (game && !game.terminal) { game.paused = true; render(); } });
$('pause').onclick = () => { if (game && !game.terminal) { game.paused = !game.paused; heldKeys.clear(); currentAction = 'stay'; replayLocked = false; replayIndex = null; rangeSelected = false; render(); } };
function beginTask2() {
  if (!game?.terminal || game.task !== 1) return;
  game = new OfflinePong({ group: game.group, participantId: game.participantId, task: 2, seed: 260919 });
  $('answer').textContent = '';
  $('question').value = '';
  $('technicalEvidenceText').textContent = '';
  running = true;
  currentAction = 'stay';
  replayPlaying = false;
  if (replayTimer) clearInterval(replayTimer);
  replayLocked = false;
  replayIndex = null;
  rangeSelected = false;
  $('review').hidden = true;
  render();
}
$('task2').onclick = beginTask2;
$('timeline').oninput = event => loadReview(event.target.value);
$('rangeStart').oninput = event => { rangeSelected = true; if (Number(event.target.value) > (replayIndex ?? 0)) event.target.value = String(replayIndex ?? 0); };
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
  if (!game || game.group !== 'A' || game.task !== 1 || (!game.terminal && !game.paused)) {
    $('answer').textContent = '当前阶段不提供回放问答。';
    return;
  }
  const end = replayIndex ?? game.history.length - 1; const start = Number($('rangeStart').value || end);
  const question = $('question').value;
  $('answer').textContent = start < end ? answerForRange(start, end, question)
    : answerForReview(game.history[end], $('ballSelect').value || null, question, end);
  game.questionHistory.push({ version: 'pong-question.v2.3', question,
    answer: $('answer').textContent, start_frame: game.history[start]?.frame,
    end_frame: game.history[end]?.frame, selected_ball_id: $('ballSelect').value || null,
    decision_id: game.history[end]?.decision?.decisionId || null });
};

$('downloadReview').onclick = () => {
  if (!game || game.group !== 'A' || game.task !== 1) return;
  const body = { version: 'pong-review-log.v2.3', participant_id: game.participantId,
    group: game.group, task: game.task, controller_source: game.controllerSource,
    actor_hash: nnModel?.model_sha256 || null, controller_version: nnController.rule_version || null,
    frames: game.history, questions: game.questionHistory };
  const url = URL.createObjectURL(new Blob([JSON.stringify(body, null, 2)], { type: 'application/json' }));
  const link = document.createElement('a');
  link.href = url; link.download = `pong-task1-review-${Date.now()}.json`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};

$('surveyForm').onsubmit = event => {
  event.preventDefault(); if (!game || !game.terminal || game.task !== 2) return;
  const values = Object.fromEntries(new FormData(event.target).entries());
  const record = { version: 'pong-questionnaire.v1', participant_id: game.participantId, group: game.group, runs: completedRuns, answers: values, note: $('surveyNote').value, saved_at: new Date().toISOString() };
  localStorage.setItem(`pong.questionnaire.${game.participantId}.${Date.now()}`, JSON.stringify(record));
  $('surveyStatus').textContent = '已保存到这台设备。'; $('questionnaire').hidden = true; $('completed').hidden = false;
};

let lastAnimationTime = null;
let physicsAccumulator = 0;
function animationLoop(now) {
  if (lastAnimationTime !== null && running && game && !game.paused && !replayLocked) {
    physicsAccumulator += Math.min(1000, Math.max(0, now - lastAnimationTime));
    const interval = SPEC.fixedDt * 1000;
    let advanced = false;
    let updates = 0;
    while (physicsAccumulator >= interval && updates < 12 && running) {
      advanced = advanceSimulation() || advanced;
      physicsAccumulator -= interval;
      updates += 1;
    }
    if (advanced) render();
  } else {
    physicsAccumulator = 0;
  }
  lastAnimationTime = now;
  requestAnimationFrame(animationLoop);
}
requestAnimationFrame(animationLoop);
