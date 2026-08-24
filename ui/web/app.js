"use strict";

const $ = (id) => document.getElementById(id);
const PAGE_ID = crypto.randomUUID ? crypto.randomUUID() : `page-${Date.now()}`;
const DEFAULT_LOCALE = "en";
const state = {
  view: null,
  locale: DEFAULT_LOCALE,
  busy: false,
  demoPlaying: false,
  pendingBeginTask1: false,
  timer: null,
  animationToken: 0,
  animationFrame: null,
  visualFrame: null,
  referenceTrajectory: null,
  referenceIndex: 1,
  referenceSettledIndex: 1,
  referenceSettledAt: 0,
  browseEvents: [],
  scrubTimer: null,
  scrubbing: false,
  questionTimer: null,
};

const COPY = {
  zh: {
    appTitle: "双机器人协作配送实验", workflowDemo: "说明与演示", task1: "任务 1", explanation: "解释", task2: "任务 2", survey: "问卷",
    warehouse: "10×11 仓库", liveScene: "协作配送现场", step: "步数", score: "总分", deliveries: "配送", collisions: "碰撞", shutdowns: "断电", detours: "绕路单位",
    shelf: "货架", pickup: "取货点", dropoff: "交付点", charger: "充电站", robots: "机器人",
    participantSetup: "参与者登记", welcome: "开始协作配送实验", overview: "你将固定控制机器人 1，与 AI 控制的机器人 2 完成两轮 120 步配送任务。",
    participantId: "参与者编号", agreement: "我已阅读并理解实验说明。", start: "开始实验",
    requiredDemo: "AI–AI 协作演示（可提前结束）", demoText: "您可以完整观看演示，也可以随时提前结束并开始任务 1。演示展示两台机器人认领、交付、协调让路和充电。",
    ruleJobs: "地图始终有两个未预分配的 A→B 共享任务。", ruleControl: "方向键、WASD 或按钮每次提交一个联合决策步；空格表示等待。",
    ruleCharge: "成功移动耗电 2；在充电站等待恢复 10；断电会提前结束本轮。", ruleScore: "计分：配送 +100、机器人碰撞 −200、断电 −50、每步 −1、参与者绕路每单位 −2。",
    playDemo: "播放演示", pauseDemo: "暂停演示", beginTask1: "开始任务 1", endDemoEarly: "提前结束演示并开始任务 1", roundInstruction: "控制机器人 1，与机器人 2 协作配送",
    roundHint: "选择一个动作后，服务器会同时取得机器人 2 的确定性动作，并推进一步。", up: "上", down: "下", left: "左", right: "右", wait: "等待",
    task1Complete: "任务 1 已完成", timeLeft: "剩余时间", aiAiExplanationTitle: "AI–AI 解释轨迹", aiAiExplanationScene: "AI–AI 固定参考轨迹", referenceMetrics: "以下为固定 AI–AI 参考轨迹数据，不是您的任务 1 成绩。", explanationTimelineLabel: "AI–AI 参考轨迹帧", explanationHint: "这里显示的是与开场演示相同的固定 AI–AI 参考轨迹，不是您刚完成的任务 1。拖动时间轴选择任意实际动作帧，再选择机器人 1 或机器人 2 并自由提问。每次提问都会使用新的生成种子；可以零次提问并随时结束。",
    question: "你的问题", questionPlaceholder: "也可以在这里输入自己的问题。", ask: "询问所选机器人", finishExplanation: "结束解释并开始任务 2", answer: "系统解释", emptyExplanation: "本帧未能生成可靠解释，请重试或选择其他动作帧。",
    presetQuestions: "快捷问题（点击后直接提问）", presetWhyAction: "为什么所选机器人在这一帧执行了这个动作？", presetTaskEffect: "这个动作如何影响当前配送任务？", presetEnergy: "当前电量和充电需求如何影响了这个动作？", presetTeammate: "队友的位置或动作是否影响了这个决定？", presetCollision: "这一步是否存在冲突或碰撞风险？", presetWhyAssignedA1: "为什么任务 1 的 A 点由所选机器人去取？", presetWhyNotAssignedA1: "为什么所选机器人没有去取任务 1 的 A 点？",
    surveyTitle: "结束问卷", surveyHint: "请对以下陈述按 1（非常不同意）到 5（非常同意）评分。", comment: "可选意见", submitSurvey: "提交问卷",
    complete: "实验完成", saved: "记录已保存。", task1Score: "任务 1 得分", task2Score: "任务 2 得分", scoreDelta: "得分变化", restart: "开始新的参与者",
    interrupted: "本轮已中断", interruptedHint: "此实验已在另一页面继续，或服务恢复后旧运行被放弃。请重新开始。", desktopRequired: "请使用宽度至少 1024 像素的桌面或笔记本电脑。",
    participant: "参与者", ai: "AI", battery: "电量", cargo: "承运", none: "无", available: "可认领", carried: "运输中", carrier: "承运者",
    coordinationUnderstanding: "我理解如何与机器人 2 协调。", aiPredictability: "机器人 2 的行为对我而言是可预测的。", interfaceClarity: "界面与计分信息清晰易懂。",
    deliveryScore: "配送得分", collisionPenalty: "碰撞扣分", shutdownPenalty: "断电扣分", timePenalty: "步数扣分", detourPenalty: "绕路扣分",
    loading: "处理中…", requiredFields: "请填写参与者编号并确认已阅读说明。", requestFailed: "操作失败", taskLabel: "任务", roundScore: "本轮得分",
    action: "动作", requestedAction: "请求", executedAction: "实际", batteryChange: "电量",
    transitionActions: "所选帧动作", workingExplanation: "正在根据所选帧生成解释…", stillWorking: "仍在生成，请稍候…",
    eventPickup: "取货", eventDelivery: "交付", eventCharging: "充电", eventChargerQueue: "排队", eventYield: "让行", eventConflict: "冲突风险", eventCollision: "碰撞",
  },
  en: {
    appTitle: "Two-Robot Collaborative Delivery Study", workflowDemo: "Instructions & demo", task1: "Task 1", explanation: "Explanation", task2: "Task 2", survey: "Survey",
    warehouse: "10×11 warehouse", liveScene: "Collaborative delivery", step: "Steps", score: "Score", deliveries: "Deliveries", collisions: "Collisions", shutdowns: "Shutdowns", detours: "Detour units",
    shelf: "Shelf", pickup: "Pickup A", dropoff: "Drop-off B", charger: "Charger", robots: "Robots",
    participantSetup: "Participant setup", welcome: "Start the collaborative delivery study", overview: "You will always control robot 1 and complete two 120-step delivery rounds with AI-controlled robot 2.",
    participantId: "Participant ID", agreement: "I have read and understood the study instructions.", start: "Start study",
    requiredDemo: "AI–AI collaboration demonstration", demoText: "You may watch the complete standardized demonstration or finish it early and begin Task 1 at any time. It shows both robots claiming, delivering, yielding, and charging.",
    ruleJobs: "The map always contains two unassigned shared A-to-B jobs.", ruleControl: "Arrow keys, WASD, or a button submits one joint decision step; Space means wait.",
    ruleCharge: "A successful move costs 2 battery; waiting at the charger restores 10; shutdown ends the round.", ruleScore: "Score: +100 delivery, −200 robot collision, −50 shutdown, −1 per step, and −2 per human detour unit.",
    playDemo: "Play demonstration", pauseDemo: "Pause demonstration", beginTask1: "Begin Task 1", endDemoEarly: "Finish demo early and begin Task 1", roundInstruction: "Control robot 1 and collaborate with robot 2",
    roundHint: "After your action, the server obtains robot 2's deterministic action and advances one joint step.", up: "Up", down: "Down", left: "Left", right: "Right", wait: "Wait",
    task1Complete: "Task 1 complete", timeLeft: "Time left", aiAiExplanationTitle: "AI–AI explanation trajectory", aiAiExplanationScene: "Fixed AI–AI reference trajectory", referenceMetrics: "These are metrics from the fixed AI–AI reference trajectory, not your Task 1 score.", explanationTimelineLabel: "AI–AI reference trajectory frame", explanationHint: "This is the same fixed AI–AI reference trajectory shown in the opening demonstration, not the Task 1 you just completed. Select any executed frame, choose robot 1 or robot 2, and ask a free-form question. Every question uses a fresh generation seed; you may ask zero questions and finish early.",
    question: "Your question", questionPlaceholder: "Or type your own question here.", ask: "Ask about selected robot", finishExplanation: "Finish explanations and begin Task 2", answer: "System explanation", emptyExplanation: "No grounded explanation was produced for this frame. Please retry or select another action frame.",
    presetQuestions: "Quick questions (click to ask)", presetWhyAction: "Why did the selected robot execute this action at this frame?", presetTaskEffect: "How did this action affect the current delivery task?", presetEnergy: "How did the battery level and charging needs affect this action?", presetTeammate: "Did the teammate's position or action affect this decision?", presetCollision: "Was there a conflict or collision risk on this step?", presetWhyAssignedA1: "Why is the selected robot assigned to collect task 1 at point A?", presetWhyNotAssignedA1: "Why is the selected robot not collecting task 1 at point A?",
    surveyTitle: "Final survey", surveyHint: "Rate each statement from 1 (strongly disagree) to 5 (strongly agree).", comment: "Optional comment", submitSurvey: "Submit survey",
    complete: "Study complete", saved: "The record has been saved.", task1Score: "Task 1 score", task2Score: "Task 2 score", scoreDelta: "Score change", restart: "Start a new participant",
    interrupted: "Run interrupted", interruptedHint: "This run continued in another page or was abandoned during recovery. Please restart.", desktopRequired: "Use a desktop or laptop at least 1024 pixels wide.",
    participant: "Participant", ai: "AI", battery: "Battery", cargo: "Carrying", none: "None", available: "Available", carried: "In transit", carrier: "Carrier",
    coordinationUnderstanding: "I understand how to coordinate with robot 2.", aiPredictability: "Robot 2's behavior is predictable to me.", interfaceClarity: "The interface and scoring information are clear.",
    deliveryScore: "Delivery points", collisionPenalty: "Collision penalty", shutdownPenalty: "Shutdown penalty", timePenalty: "Step penalty", detourPenalty: "Detour penalty",
    loading: "Working…", requiredFields: "Enter a participant ID and confirm the instructions.", requestFailed: "Request failed", taskLabel: "Task", roundScore: "Round score",
    action: "Action", requestedAction: "Requested", executedAction: "Executed", batteryChange: "Battery",
    transitionActions: "Selected actions", workingExplanation: "Generating an explanation for the selected frame…", stillWorking: "Still generating—please wait…",
    eventPickup: "Pickup", eventDelivery: "Delivery", eventCharging: "Charging", eventChargerQueue: "Queue", eventYield: "Yield", eventConflict: "Conflict risk", eventCollision: "Collision",
  },
};

Object.assign(COPY.zh, {
  controlTransitionHint: "本实验流程在两轮任务之间不提供解释或提问环节。任务 2 将从新的机器人状态、电量 100 和不同任务序列开始。",
  beginTask2: "开始任务 2",
  ruleCharge: "成功移动耗电 2；在充电站等待恢复 10；断电会提前结束本轮。",
  testCondition: "开发测试条件",
  conditionAuto: "自动区组分配",
  conditionExplanation: "A 组（有解释）",
  conditionControl: "B 组（无解释）",
  testConditionHint: "仅用于界面测试；数据写入独立的 development 命名空间。",
  assignedTestCondition: "当前测试条件",
  groupATitle: "您已分配到 A 组（有解释）",
  groupADescription: "任务 1 结束后，您将进入解释与提问环节，然后再开始任务 2。",
  groupBTitle: "您已分配到 B 组（无解释）",
  groupBDescription: "任务 1 结束后不提供解释或提问环节；确认过渡说明后开始任务 2。",
  explanationHint: "这里显示的是与开场演示相同的固定 AI–AI 参考轨迹，不是您刚完成的任务 1。拖动时间轴选择任意实际动作帧，再选择机器人 1 或机器人 2。您可以点击快捷问题直接提问，也可以自行输入。每次提问都会使用新的生成种子；可以零次提问并随时结束。",
  questionTarget: "提问对象",
  robot1Option: "机器人 1（AI）",
  robot2Option: "机器人 2（AI）",
  questionPlaceholder: "也可以在这里输入自己的问题。",
  ask: "询问所选机器人",
  temporaryNetworkError: "临时网络连接中断，请重试；当前进度已保留。",
});
Object.assign(COPY.en, {
  controlTransitionHint: "This study flow does not provide an explanation or question period between the two rounds. Task 2 will start with a fresh robot state, 100 battery, and a different task sequence.",
  beginTask2: "Begin Task 2",
  ruleCharge: "A successful move costs 2 battery; waiting at the charger restores 10; shutdown ends the round.",
  testCondition: "Development test condition",
  conditionAuto: "Automatic block allocation",
  conditionExplanation: "Group A (explanations)",
  conditionControl: "Group B (no explanations)",
  testConditionHint: "For interface testing only; records use the isolated development namespace.",
  assignedTestCondition: "Current test condition",
  groupATitle: "You are assigned to Group A (explanations)",
  groupADescription: "After Task 1, you will enter the explanation and question period before starting Task 2.",
  groupBTitle: "You are assigned to Group B (no explanations)",
  groupBDescription: "There is no explanation or question period after Task 1; Task 2 begins after you acknowledge the transition notice.",
  explanationHint: "This is the same fixed AI–AI reference trajectory shown in the opening demonstration, not the Task 1 you just completed. Select any executed frame and robot. You can click a quick question to ask immediately or type your own. Every question uses a fresh generation seed; you may ask zero questions and finish early.",
  questionTarget: "Robot to ask about",
  robot1Option: "Robot 1 (AI)",
  robot2Option: "Robot 2 (AI)",
  questionPlaceholder: "Or type your own question here.",
  ask: "Ask about selected robot",
  temporaryNetworkError: "The temporary tunnel connection was interrupted. Please retry; your current progress is preserved.",
});

function tr(key) { return COPY[state.locale][key] || key; }
function localeCode() { return state.locale === "zh" ? "zh-CN" : "en"; }
function allowed(command) { return Boolean(state.view?.study?.allowed_commands?.includes(command)); }
function operationId() { return crypto.randomUUID ? crypto.randomUUID() : `op-${Date.now()}-${Math.random()}`; }

const TRANSIENT_HTTP_STATUSES = new Set([408, 425, 429, 500, 502, 503, 504]);
const API_MAX_ATTEMPTS = 4;

function delay(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

function isTransientHttpStatus(status) {
  return TRANSIENT_HTTP_STATUSES.has(status) || (status >= 520 && status <= 530);
}

async function api(path, options = {}) {
  const requestOptions = {
    ...options,
    headers: { "Content-Type": "application/json", "X-Warehouse-Page": PAGE_ID, ...(options.headers || {}) },
  };

  for (let attempt = 1; attempt <= API_MAX_ATTEMPTS; attempt += 1) {
    try {
      const response = await fetch(path, requestOptions);
      let payload = {};
      try { payload = await response.json(); } catch (_) { payload = {}; }
      if (response.ok) return payload;

      const error = new Error(payload.error || `${response.status} ${response.statusText}`);
      error.payload = payload;
      error.status = response.status;
      throw error;
    } catch (error) {
      const retryable = error.status == null || isTransientHttpStatus(error.status);
      if (!retryable) throw error;
      if (attempt === API_MAX_ATTEMPTS) {
        const friendlyError = new Error(tr("temporaryNetworkError"));
        friendlyError.cause = error;
        friendlyError.status = error.status;
        friendlyError.payload = error.payload || {};
        throw friendlyError;
      }
      await delay(350 * (2 ** (attempt - 1)));
    }
  }

  throw new Error(tr("temporaryNetworkError"));
}

async function ensureReferenceTrajectory() {
  if (state.referenceTrajectory) return state.referenceTrajectory;
  state.referenceTrajectory = await api("/api/study/reference-trajectory");
  const count = state.referenceTrajectory.frames?.length || 1;
  // The participant's Task 1 frame is unrelated to the fixed AI-AI reference
  // material.  Always start at its first executed transition.
  state.referenceIndex = Math.min(1, count - 1);
  state.referenceSettledIndex = state.referenceIndex;
  state.referenceSettledAt = performance.now();
  return state.referenceTrajectory;
}

async function synchronizeReferenceTrajectory() {
  const latest = await api("/api/study/reference-trajectory");
  if (state.referenceTrajectory?.trajectory_hash === latest.trajectory_hash) {
    return latest;
  }
  state.referenceTrajectory = latest;
  const count = latest.frames?.length || 1;
  state.referenceIndex = Math.min(Math.max(1, state.referenceIndex), count - 1);
  state.referenceSettledIndex = state.referenceIndex;
  state.referenceSettledAt = performance.now();
  await render(referenceView(state.view, state.referenceIndex), { skipReferenceLoad: true });
  return latest;
}

function referenceView(baseView, index = state.referenceIndex) {
  const trajectory = state.referenceTrajectory;
  const frames = trajectory?.frames || [];
  if (!frames.length) return baseView;
  const bounded = Math.min(Math.max(1, Number(index) || 1), frames.length - 1);
  const frame = frames[bounded];
  return {
    ...baseView,
    map: trajectory.map || baseView.map,
    state: frame.state,
    transition: frame.transition,
    timeline: {
      ...(baseView.timeline || {}),
      index: bounded,
      count: frames.length,
      trajectory_kind: trajectory.trajectory_kind,
      trajectory_seed: trajectory.trajectory_seed,
      trajectory_hash: trajectory.trajectory_hash,
      agent_control: trajectory.agent_control,
    },
  };
}

function queueSettledBrowseEvent() {
  if (!state.referenceTrajectory || !state.referenceSettledAt) return;
  const durationMs = Math.max(0, Math.round(performance.now() - state.referenceSettledAt));
  state.browseEvents.push({
    timeline_index: state.referenceSettledIndex,
    dwell_ms: durationMs,
    trajectory_hash: state.referenceTrajectory.trajectory_hash,
  });
  state.referenceSettledAt = 0;
}

async function flushTimelineEvents() {
  if (!state.browseEvents.length || !state.view?.study?.run_id) return;
  const events = state.browseEvents.splice(0, state.browseEvents.length);
  try {
    await api("/api/study/timeline-events", {
      method: "POST",
      body: JSON.stringify({
        operation_id: operationId(),
        run_id: state.view.study.run_id,
        trajectory_hash: state.referenceTrajectory?.trajectory_hash,
        events,
      }),
    });
  } catch (_) {
    state.browseEvents.unshift(...events);
  }
}

function renderReferenceEvents() {
  const eventContainer = $("timelineEvents");
  const events = {};
  for (const frame of state.referenceTrajectory?.frames || []) {
    for (const tag of frame.event_tags || []) {
      (events[tag] ||= []).push(frame.index);
    }
  }
  const labels = {
    pickup: "eventPickup", delivery: "eventDelivery", charging: "eventCharging",
    charger_queue: "eventChargerQueue", queue: "eventChargerQueue",
    coordination_yield: "eventYield", yield: "eventYield",
    collision_risk: "eventConflict", conflict: "eventConflict",
    robot_collision: "eventCollision", collision: "eventCollision",
  };
  eventContainer.replaceChildren(...Object.entries(labels).flatMap(([tag, labelKey]) => {
    const frames = events[tag] || [];
    if (!frames.length) return [];
    const button = document.createElement("button");
    button.type = "button";
    button.className = "timeline-event-button";
    button.textContent = `${tr(labelKey)} (${frames.length})`;
    button.addEventListener("click", () => {
      const next = frames.find((frame) => frame > state.referenceIndex) ?? frames[0];
      selectReferenceFrame(next, false);
    });
    return [button];
  }));
}

async function selectReferenceFrame(index, scrubbing = false) {
  if (!state.referenceTrajectory) await ensureReferenceTrajectory();
  const count = state.referenceTrajectory.frames?.length || 1;
  state.referenceIndex = Math.min(Math.max(1, Number(index) || 1), count - 1);
  state.scrubbing = scrubbing;
  clearTimeout(state.scrubTimer);
  cancelMotion();
  await render(referenceView(state.view, state.referenceIndex), { skipReferenceLoad: true });
  const settle = async () => {
    queueSettledBrowseEvent();
    state.referenceSettledIndex = state.referenceIndex;
    state.referenceSettledAt = performance.now();
    state.scrubbing = false;
    await render(referenceView(state.view, state.referenceIndex), { skipReferenceLoad: true });
    if (state.browseEvents.length >= 10) void flushTimelineEvents();
  };
  if (scrubbing) state.scrubTimer = setTimeout(settle, 150);
  else await settle();
}

function showError(error) {
  $("toastText").textContent = error instanceof Error ? error.message : String(error);
  $("toast").classList.remove("hidden");
}

function setBusy(value) {
  state.busy = value;
  document.querySelectorAll("button").forEach((button) => {
    if (button.id === "languageButton" || button.id === "toastClose") return;
    const tutorialActive = state.view?.study?.stage === "instructions";
    if (tutorialActive && button.id === "beginTask1Button") {
      button.disabled = !allowed("begin_task1") || state.pendingBeginTask1;
      return;
    }
    if (tutorialActive && button.id === "demoPlayButton" && state.demoPlaying) {
      button.disabled = false;
      return;
    }
    button.disabled = value || button.dataset.locked === "true";
  });
}

async function command(name, payload = {}) {
  if (state.busy) return null;
  const study = state.view?.study || { stage: "idle", state_version: 0, run_id: null };
  const envelope = {
    operation_id: operationId(),
    run_id: name === "start" ? null : study.run_id,
    expected_stage: name === "start" ? "idle" : study.stage,
    expected_state_version: name === "start" ? 0 : study.state_version,
    command: name,
    payload,
  };
  setBusy(true);
  try {
    const result = await api("/api/study/command", { method: "POST", body: JSON.stringify(envelope) });
    await render(result.view || result);
    if (result.report) renderAnswer(result.report);
    return result;
  } catch (error) {
    if (error.payload?.view?.state) await render(error.payload.view);
    showError(error);
    return null;
  } finally {
    setBusy(false);
    renderStage();
  }
}

function setLanguage(locale, notify = true, rerender = true) {
  state.locale = locale === "en" ? "en" : "zh";
  const requestedLocale = localeCode();
  document.documentElement.lang = requestedLocale;
  document.querySelectorAll("[data-i18n]").forEach((node) => { node.textContent = tr(node.dataset.i18n); });
  document.querySelectorAll("[data-i18n-placeholder]").forEach((node) => { node.placeholder = tr(node.dataset.i18nPlaceholder); });
  $("timelineRange").setAttribute("aria-label", tr("explanationTimelineLabel"));
  $("languageButtonLabel").textContent = state.locale === "zh" ? "EN" : "中";
  buildSurvey();
  if (state.view && rerender) {
    state.view = {
      ...state.view,
      study: { ...(state.view.study || {}), locale: requestedLocale },
    };
    void render(state.view);
  }
  if (notify && allowed("set_language")) command("set_language", { locale: requestedLocale });
}

function drawWarehouse(view, visualFrame = null) {
  const canvas = $("warehouseCanvas");
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(640, canvas.clientWidth);
  const height = Math.max(430, canvas.clientHeight);
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  const rows = view.map.rows, cols = view.map.cols;
  const size = Math.min((width - 36) / cols, (height - 36) / rows);
  const originX = (width - cols * size) / 2, originY = (height - rows * size) / 2;
  const cell = ([row, col]) => [originX + col * size, originY + row * size];
  ctx.fillStyle = "#f8fafc"; ctx.fillRect(originX, originY, cols * size, rows * size);
  ctx.strokeStyle = "#dce4ef"; ctx.lineWidth = 1;
  for (let row = 0; row <= rows; row++) { ctx.beginPath(); ctx.moveTo(originX, originY + row * size); ctx.lineTo(originX + cols * size, originY + row * size); ctx.stroke(); }
  for (let col = 0; col <= cols; col++) { ctx.beginPath(); ctx.moveTo(originX + col * size, originY); ctx.lineTo(originX + col * size, originY + rows * size); ctx.stroke(); }
  for (const position of view.map.shelves) { const [x,y] = cell(position); ctx.fillStyle = "#9aa8ba"; ctx.fillRect(x + 2, y + 2, size - 4, size - 4); }
  const [cx,cy] = cell(view.map.charger_position); ctx.fillStyle = "#6558e8"; ctx.fillRect(cx + 4, cy + 4, size - 8, size - 8); ctx.fillStyle = "#ffffff"; ctx.font = `bold ${size*.46}px sans-serif`; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText("⚡", cx + size/2, cy + size/2);
  const tasks = view.state.tasks || [];
  tasks.forEach((task, index) => {
    if (task.status === "available") {
      const [x,y] = cell(task.pickup_position); ctx.fillStyle = "#f4b740"; ctx.beginPath(); ctx.arc(x+size/2,y+size/2,size*.31,0,Math.PI*2); ctx.fill(); ctx.fillStyle="#26324a"; ctx.fillText(`A${index+1}`,x+size/2,y+size/2);
    }
    const [x,y] = cell(task.delivery_position); ctx.fillStyle = "#31b883"; ctx.beginPath(); ctx.arc(x+size/2,y+size/2,size*.31,0,Math.PI*2); ctx.fill(); ctx.fillStyle="#ffffff"; ctx.fillText(`B${index+1}`,x+size/2,y+size/2);
  });
  (view.state.agents || []).forEach((agent) => {
    const visual = visualFrame?.agents?.[agent.id] || {};
    const position = visual.position || agent.position;
    const [x,y] = cell(position); const isHuman = agent.id === "robot_1";
    ctx.save();
    ctx.globalAlpha = visual.opacity == null ? 1 : visual.opacity;
    if (visual.charging) {
      ctx.strokeStyle = "rgba(101,88,232,.48)";
      ctx.lineWidth = Math.max(2, size * .06);
      ctx.beginPath(); ctx.arc(x+size/2,y+size/2,size*(.41 + .05*(visual.pulse || 0)),0,Math.PI*2); ctx.stroke();
    }
    ctx.translate(x + size/2, y + size/2);
    const scale = visual.scale || 1;
    ctx.scale(scale, scale);
    ctx.translate(-(x + size/2), -(y + size/2));
    ctx.fillStyle = agent.active ? (isHuman ? "#4f6ff0" : "#f56b3d") : "#d9485f";
    ctx.beginPath(); ctx.roundRect(x+size*.14,y+size*.14,size*.72,size*.72,size*.15); ctx.fill();
    ctx.fillStyle="#ffffff"; ctx.font=`900 ${size*.34}px sans-serif`; ctx.fillText(isHuman?"1":"2",x+size/2,y+size/2);
    ctx.restore();
    const batteryText = `${Math.round(agent.battery)}%`;
    ctx.save();
    ctx.font = `800 ${Math.max(10, size * .2)}px sans-serif`;
    const pillWidth = Math.max(size * .62, ctx.measureText(batteryText).width + 12);
    const pillHeight = Math.max(17, size * .27);
    const pillX = x + size / 2 - pillWidth / 2;
    const pillY = Math.max(originY + 2, y - pillHeight * .72);
    ctx.fillStyle = "rgba(255,255,255,.96)";
    ctx.strokeStyle = agent.battery <= 20 ? "#d9485f" : "#cbd6e4";
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.roundRect(pillX,pillY,pillWidth,pillHeight,pillHeight/2); ctx.fill(); ctx.stroke();
    ctx.fillStyle = agent.battery <= 20 ? "#b4233b" : "#26324a";
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(batteryText, x + size/2, pillY + pillHeight/2);
    if (agent.carrying_label) {
      const cargoSize = Math.max(17, size * .3);
      const cargoX = x + size * .78;
      const cargoY = y + size * .05;
      ctx.fillStyle = "#f4b740";
      ctx.beginPath(); ctx.roundRect(cargoX,cargoY,cargoSize,cargoSize,cargoSize*.25); ctx.fill();
      ctx.fillStyle = "#26324a";
      ctx.font = `900 ${Math.max(9, cargoSize * .48)}px sans-serif`;
      ctx.fillText(agent.carrying_label, cargoX + cargoSize/2, cargoY + cargoSize/2);
    }
    ctx.restore();
  });
}

const MOVE_VECTOR = {
  UP: [-1, 0], DOWN: [1, 0], LEFT: [0, -1], RIGHT: [0, 1], WAIT: [0, 0],
};

function actionLabel(action) {
  const labels = state.locale === "zh"
    ? { UP: "↑ 上移", DOWN: "↓ 下移", LEFT: "← 左移", RIGHT: "→ 右移", WAIT: "等待" }
    : { UP: "↑ Up", DOWN: "↓ Down", LEFT: "← Left", RIGHT: "→ Right", WAIT: "Wait" };
  return labels[action] || String(action || "—");
}

function easeMotion(value) {
  const clamped = Math.max(0, Math.min(1, value));
  return .5 - Math.cos(Math.PI * clamped) / 2;
}

function interpolateTransition(transition, progress, opacity = 1) {
  const agents = {};
  for (const motion of transition.agents || []) {
    const from = motion.from_position;
    const to = motion.to_position;
    let row = from[0] + (to[0] - from[0]) * progress;
    let col = from[1] + (to[1] - from[1]) * progress;
    if (motion.blocked) {
      const [dr, dc] = MOVE_VECTOR[motion.proposed_action] || [0, 0];
      const nudge = Math.sin(Math.PI * progress) * .18;
      row += dr * nudge;
      col += dc * nudge;
    }
    if (motion.collision) {
      const shake = Math.sin(progress * Math.PI * 6) * .055;
      col += motion.id === "robot_1" ? shake : -shake;
    }
    const waiting = motion.executed_action === "WAIT";
    agents[motion.id] = {
      position: [row, col],
      opacity,
      scale: waiting ? 1 + Math.sin(Math.PI * progress) * .065 : 1,
      pulse: Math.sin(Math.PI * progress),
      charging: Boolean(motion.charging),
    };
  }
  return { agents };
}

function publishVisualPositions(canvas, visualFrame) {
  for (const [agentId, visual] of Object.entries(visualFrame?.agents || {})) {
    const key = agentId.replace(/_([a-z0-9])/g, (_, letter) => letter.toUpperCase());
    canvas.dataset[`${key}Row`] = Number(visual.position[0]).toFixed(3);
    canvas.dataset[`${key}Col`] = Number(visual.position[1]).toFixed(3);
  }
}

function cancelMotion() {
  state.animationToken += 1;
  if (state.animationFrame != null) cancelAnimationFrame(state.animationFrame);
  state.animationFrame = null;
  state.visualFrame = null;
  if ($("warehouseCanvas")) {
    $("warehouseCanvas").dataset.animationRunning = "false";
  }
}

function animateOnce(view, transition, duration = 400) {
  cancelMotion();
  const token = state.animationToken;
  const canvas = $("warehouseCanvas");
  canvas.dataset.animationRunning = "true";
  canvas.dataset.animationMode = "single";
  canvas.dataset.transitionFrame = String(transition.to_frame);
  return new Promise((resolve) => {
    const started = performance.now();
    const paint = (timestamp) => {
      if (token !== state.animationToken) { resolve(false); return; }
      const raw = Math.min(1, (timestamp - started) / duration);
      const progress = easeMotion(raw);
      state.visualFrame = interpolateTransition(transition, progress);
      publishVisualPositions(canvas, state.visualFrame);
      canvas.dataset.animationProgress = progress.toFixed(3);
      drawWarehouse(view, state.visualFrame);
      if (raw < 1) {
        state.animationFrame = requestAnimationFrame(paint);
      } else {
        state.animationFrame = null;
        state.visualFrame = null;
        canvas.dataset.animationRunning = "false";
        drawWarehouse(view);
        resolve(true);
      }
    };
    state.animationFrame = requestAnimationFrame(paint);
  });
}

function animateLoop(view, transition) {
  cancelMotion();
  const token = state.animationToken;
  const canvas = $("warehouseCanvas");
  const cycle = 1000;
  canvas.dataset.animationRunning = "true";
  canvas.dataset.animationMode = "loop";
  canvas.dataset.transitionFrame = String(transition.to_frame);
  const started = performance.now();
  const paint = (timestamp) => {
    if (token !== state.animationToken) return;
    const elapsed = (timestamp - started) % cycle;
    let progress = 0;
    let opacity = 1;
    if (elapsed < 600) {
      progress = easeMotion(elapsed / 600);
    } else if (elapsed < 850) {
      progress = 1;
    } else if (elapsed < 925) {
      progress = 1;
      opacity = 1 - (elapsed - 850) / 75;
    } else {
      progress = 0;
      opacity = (elapsed - 925) / 75;
    }
    state.visualFrame = interpolateTransition(transition, progress, opacity);
    publishVisualPositions(canvas, state.visualFrame);
    canvas.dataset.animationProgress = progress.toFixed(3);
    drawWarehouse(view, state.visualFrame);
    state.animationFrame = requestAnimationFrame(paint);
  };
  state.animationFrame = requestAnimationFrame(paint);
}

function renderRobots(agents, transition = null, showActions = false, agentControl = {}) {
  const motions = Object.fromEntries((transition?.agents || []).map((item) => [item.id, item]));
  $("robotCards").replaceChildren(...agents.map((agent) => {
    const article = document.createElement("article");
    const controller = agentControl[agent.id]
      || (agent.id === "robot_1" ? "human" : "ai");
    article.className = `robot-card ${controller === "human" ? "participant" : "ai"}`;
    const motion = motions[agent.id];
    let motionMarkup = "";
    if (showActions && motion) {
      const proposed = motion.proposed_action;
      const executed = motion.executed_action;
      const actionText = proposed && proposed !== executed
        ? `<span><b>${tr("requestedAction")}:</b> ${actionLabel(proposed)} · <b>${tr("executedAction")}:</b> ${actionLabel(executed)}</span>`
        : `<span><b>${tr("action")}:</b> ${actionLabel(executed)}</span>`;
      const delta = Number(motion.battery_delta || 0);
      const deltaText = delta === 0 ? "" : `<em>${tr("batteryChange")} ${delta > 0 ? "+" : ""}${delta.toFixed(0)}</em>`;
      motionMarkup = `<div class="motion-status ${motion.blocked ? "blocked" : ""} ${motion.charging ? "charging" : ""}">${actionText}${deltaText}</div>`;
    }
    const robotNumber = agent.id === "robot_1" ? "1" : "2";
    const controllerLabel = controller === "human" ? tr("participant") : tr("ai");
    article.innerHTML = `<header><strong>${controllerLabel} · R${robotNumber}</strong></header>${motionMarkup || `<small>${tr("action")}: —</small>`}`;
    return article;
  }));
}

function renderScores(snapshot) {
  const breakdown = snapshot.score_breakdown || {};
  $("stepValue").textContent = `${snapshot.frame || 0} / 120`;
  $("scoreValue").textContent = Math.round(snapshot.user_score || 0);
  $("deliveryValue").textContent = snapshot.total_deliveries || 0;
  $("collisionValue").textContent = snapshot.robot_collision_events || 0;
  $("shutdownValue").textContent = snapshot.shutdown_count || 0;
  $("detourValue").textContent = Number(snapshot.human_route_regret_units || 0).toFixed(1);
  const labels = { delivery: tr("deliveryScore"), robot_collision: tr("collisionPenalty"), shutdown: tr("shutdownPenalty"), time: tr("timePenalty"), human_detour: tr("detourPenalty") };
  $("scoreBreakdown").replaceChildren(...Object.entries(labels).map(([key,label]) => {
    const row = document.createElement("div"); row.innerHTML = `<span>${label}</span><strong>${Math.round(breakdown[key] || 0)}</strong>`; return row;
  }));
}

function renderWorkflow(stage, condition) {
  const order = ["instructions", "task1", "explanation", "task2", "survey"];
  const normalized = stage === "completed" ? "survey" : stage;
  const current = stage === "task1_complete" ? -1 : Math.max(0, order.indexOf(normalized));
  ["workflowInstructions","workflowTask1","workflowExplanation","workflowTask2","workflowSurvey"].forEach((id,index) => {
    const skipped = condition === "control" && index === 2 && ["task1_complete", "task2", "survey", "completed"].includes(stage);
    $(id).classList.toggle("active", index === current);
    $(id).classList.toggle("done", !skipped && (stage === "task1_complete" ? index <= 1 : index < current || stage === "completed"));
    $(id).classList.toggle("skipped", skipped);
  });
}

function renderStage() {
  if (!state.view) return;
  const study = state.view.study || {};
  const stage = study.stage || "idle";
  const aiAiReference = stage === "explanation"
    && state.view.timeline?.trajectory_kind === "ai_ai_reference";
  $("sceneTitle").textContent = aiAiReference ? tr("aiAiExplanationScene") : tr("liveScene");
  $("referenceMetricsLabel").classList.toggle("hidden", !aiAiReference);
  const panels = { idle: "setupPanel", instructions: "instructionsPanel", task1: "roundPanel", task1_complete: "task1CompletePanel", task2: "roundPanel", explanation: "explanationPanel", survey: "surveyPanel", completed: "completePanel", abandoned: "interruptedPanel" };
  ["setupPanel","instructionsPanel","roundPanel","task1CompletePanel","explanationPanel","surveyPanel","completePanel","interruptedPanel"].forEach((id) => $(id).classList.toggle("hidden", panels[stage] !== id));
  renderWorkflow(stage, study.condition);
  const tutorial = study.tutorial || {};
  $("testConditionField").classList.toggle("hidden", !study.test_condition_selector);
  const groupVisible = Boolean(stage !== "idle" && study.group_code);
  $("assignmentGroupBanner").classList.toggle("hidden", !groupVisible);
  $("assignmentGroupBanner").classList.toggle("group-a", study.group_code === "A");
  $("assignmentGroupBanner").classList.toggle("group-b", study.group_code === "B");
  if (groupVisible) {
    const prefix = study.group_code === "A" ? "groupA" : "groupB";
    $("assignmentGroupTitle").textContent = tr(`${prefix}Title`);
    $("assignmentGroupDescription").textContent = tr(`${prefix}Description`);
  }
  const testStatusVisible = Boolean(
    study.test_condition_selector && stage !== "idle" && study.condition
  );
  $("testConditionStatus").classList.toggle("hidden", !testStatusVisible);
  if (testStatusVisible) {
    const conditionKey = study.condition === "explanation"
      ? "conditionExplanation"
      : "conditionControl";
    $("testConditionStatus").textContent = `${tr("assignedTestCondition")}: ${tr(conditionKey)}`;
  }
  if (stage === "instructions") {
    const total = Math.max(1, tutorial.total_frames || 1), played = Math.min(total, (tutorial.max_played_index || 0) + 1);
    $("demoStatus").textContent = `${played} / ${total}`; $("demoProgressBar").style.width = `${100*played/total}%`;
    const canBeginTask1 = allowed("begin_task1");
    const beginTask1Button = $("beginTask1Button");
    const beginTask1Key = tutorial.complete ? "beginTask1" : "endDemoEarly";
    beginTask1Button.disabled = !canBeginTask1 || state.pendingBeginTask1;
    beginTask1Button.dataset.locked = canBeginTask1 ? "false" : "true";
    beginTask1Button.dataset.i18n = beginTask1Key;
    beginTask1Button.textContent = tr(beginTask1Key);
    $("demoPlayButton").textContent = state.demoPlaying ? tr("pauseDemo") : tr("playDemo");
  }
  if (stage === "task1" || stage === "task2") {
    $("roundBadge").textContent = stage.toUpperCase();
    document.querySelectorAll("#actionPad button").forEach((button) => { button.disabled = state.busy; });
  }
  if (stage === "task1_complete") {
    const summary = study.round_summaries?.task1;
    $("controlTask1Score").textContent = Math.round(summary?.score ?? 0);
    $("beginTask2Button").disabled = state.busy || !allowed("begin_task2");
    $("beginTask2Button").dataset.locked = allowed("begin_task2") ? "false" : "true";
  }
  const timelineVisible = stage === "explanation";
  $("timelinePanel").classList.toggle("hidden", !timelineVisible);
  $("robotDetailPanel").classList.toggle("hidden", !timelineVisible);
  if (timelineVisible) {
    const count = state.referenceTrajectory?.frames?.length || state.view.timeline?.count || 1;
    const index = state.referenceTrajectory ? state.referenceIndex : (state.view.timeline?.index || 0);
    $("timelineRange").min = count > 1 ? "1" : "0"; $("timelineRange").max = String(Math.max(0,count-1)); $("timelineRange").value = String(index);
    $("timelineLabel").textContent = `${index} / ${Math.max(0,count-1)}`;
    $("timelineBack").disabled = state.busy || index <= 1;
    $("timelineBack").dataset.locked = index <= 1 ? "true" : "false";
    $("timelineForward").disabled = state.busy || index >= count - 1;
    $("timelineForward").dataset.locked = index >= count - 1 ? "true" : "false";
    renderReferenceEvents();
    const summary = study.round_summaries?.task1; $("task1Result").textContent = summary ? `${tr("roundScore")}: ${Math.round(summary.score)}` : "";
    const targets = study.explanation_target_agents || ["robot_1", "robot_2"];
    const requestedTarget = study.explanation_target_agent || "robot_2";
    $("questionTarget").value = targets.includes(requestedTarget)
      ? requestedTarget
      : "robot_2";
  }
  if (stage === "completed") {
    $("finalTask1").textContent = Math.round(study.round_summaries?.task1?.score ?? 0);
    $("finalTask2").textContent = Math.round(study.round_summaries?.task2?.score ?? 0);
    $("finalDelta").textContent = Math.round(study.score_delta ?? 0);
  }
  updateTimer(study.explanation_seconds_remaining);
}

function updateTimer(seconds) {
  clearInterval(state.timer); state.timer = null;
  if (seconds == null || !$("explanationTimer")) return;
  let remaining = Math.max(0, Math.floor(seconds));
  const paint = () => { $("explanationTimer").textContent = `${String(Math.floor(remaining/60)).padStart(2,"0")}:${String(remaining%60).padStart(2,"0")}`; };
  paint(); state.timer = setInterval(() => { remaining = Math.max(0,remaining-1); paint(); },1000);
}

function renderAnswer(report) {
  const text = report?.explanation_document?.text || report?.explanation || "";
  if (!text.trim()) {
    showError(new Error(tr("emptyExplanation")));
    return;
  }
  $("answerText").textContent = text; $("answerPanel").classList.remove("hidden");
}

async function render(view, options = {}) {
  const previousStage = state.view?.study?.stage;
  let renderedView = view;
  const requestedStage = view.study?.stage || "idle";
  if (requestedStage === "explanation" && !options.skipReferenceLoad) {
    state.view = view;
    await ensureReferenceTrajectory();
    renderedView = referenceView(view, state.referenceIndex);
  }
  if (previousStage === "explanation" && requestedStage !== "explanation") {
    clearTimeout(state.scrubTimer);
    queueSettledBrowseEvent();
    void flushTimelineEvents();
    state.referenceTrajectory = null;
    state.referenceIndex = 1;
    state.referenceSettledIndex = 1;
    state.scrubbing = false;
    cancelMotion();
  }
  state.view = renderedView;
  view = renderedView;
  document.body.dataset.studyStage = view.study?.stage || "idle";
  document.body.dataset.stateVersion = String(view.study?.state_version ?? 0);
  if (requestedStage !== "idle" && view.study?.locale) {
    const requestedLocale = view.study.locale === "en" ? "en" : "zh";
    if (requestedLocale !== state.locale) setLanguage(requestedLocale, false, false);
  }
  const stage = view.study?.stage || "idle";
  const showActions = ["instructions", "explanation"].includes(stage);
  renderRobots(
    view.state?.agents || [],
    view.transition,
    showActions,
    view.timeline?.agent_control || {},
  );
  renderScores(view.state || {});
  renderStage();
  if (view.transition?.loop && stage === "explanation" && !state.scrubbing) {
    animateLoop(view, view.transition);
  } else if (view.transition) {
    await animateOnce(view, view.transition, 400);
  } else {
    cancelMotion();
    drawWarehouse(view);
  }
  if (view.last_explanation) renderAnswer(view.last_explanation);
}

function buildSurvey() {
  const questions = [
    ["coordination_understanding", tr("coordinationUnderstanding")],
    ["ai_predictability", tr("aiPredictability")],
    ["interface_clarity", tr("interfaceClarity")],
  ];
  $("surveyQuestions").replaceChildren(...questions.map(([name,label]) => {
    const section = document.createElement("section"); section.className = "survey-question";
    section.innerHTML = `<p>${label}</p><div class="scale">${[1,2,3,4,5].map((value) => `<label><input type="radio" name="${name}" value="${value}" required><span>${value}</span></label>`).join("")}</div>`;
    return section;
  }));
}

async function playDemo() {
  if (state.demoPlaying) { state.demoPlaying = false; renderStage(); return; }
  state.pendingBeginTask1 = false;
  state.demoPlaying = true; renderStage();
  while (state.demoPlaying && state.view?.study?.stage === "instructions" && !state.view.study.tutorial?.complete) {
    const result = await command("tutorial_advance");
    if (state.pendingBeginTask1) {
      state.pendingBeginTask1 = false;
      if (state.view?.study?.stage === "instructions") {
        await command("begin_task1");
      }
      break;
    }
    if (!result) break;
  }
  state.demoPlaying = false; renderStage();
}

function restartPayload() {
  const payload = {
    participant_id: $("participantInput").value.trim(),
    locale: localeCode(),
    viewport_width: window.innerWidth,
  };
  // A condition override is a development-only capability.  The selector is
  // still present in the shared HTML so development and formal deployments
  // use the same page, but a hidden control must never influence a formal
  // enrollment request.
  if (state.view?.study?.test_condition_selector === true) {
    payload.condition_override = $("testConditionSelector").value;
  }
  return payload;
}

$("languageButton").addEventListener("click", () => setLanguage(state.locale === "zh" ? "en" : "zh"));
$("toastClose").addEventListener("click", () => $("toast").classList.add("hidden"));
$("startButton").addEventListener("click", () => {
  if (!$("participantInput").value.trim() || !$("rulesAgreement").checked) { showError(tr("requiredFields")); return; }
  command("start", restartPayload());
});
$("demoPlayButton").addEventListener("click", playDemo);
$("beginTask1Button").addEventListener("click", async () => {
  if (state.view?.study?.stage !== "instructions") return;
  state.demoPlaying = false;
  if (state.busy) {
    state.pendingBeginTask1 = true;
    renderStage();
    return;
  }
  cancelMotion();
  await command("begin_task1");
});
$("beginTask2Button").addEventListener("click", () => command("begin_task2"));
document.querySelectorAll("#actionPad button").forEach((button) => button.addEventListener("click", () => command("human_action", { action: button.dataset.action })));
$("timelineBack").addEventListener("click", () => selectReferenceFrame(state.referenceIndex - 1, false));
$("timelineForward").addEventListener("click", () => selectReferenceFrame(state.referenceIndex + 1, false));
$("timelineRange").addEventListener("input", (event) => selectReferenceFrame(Number(event.target.value), true));
async function submitExplanationQuestion(question, questionKind = null) {
  const prompt = String(question || "").trim();
  if (!prompt) return;
  clearTimeout(state.scrubTimer);
  if (state.scrubbing || state.referenceSettledIndex !== state.referenceIndex) {
    queueSettledBrowseEvent();
    state.referenceSettledIndex = state.referenceIndex;
    state.referenceSettledAt = performance.now();
    state.scrubbing = false;
  }
  $("questionInput").value = prompt;
  $("questionStatus").textContent = tr("workingExplanation");
  $("questionStatus").classList.remove("hidden");
  clearTimeout(state.questionTimer);
  state.questionTimer = setTimeout(() => {
    $("questionStatus").textContent = tr("stillWorking");
  }, 5000);
  await flushTimelineEvents();
  await synchronizeReferenceTrajectory();
  const payload = {
    question: prompt,
    target_agent: $("questionTarget").value,
    trajectory_hash: state.referenceTrajectory?.trajectory_hash,
    selected_frame: state.referenceIndex,
  };
  if (questionKind) payload.question_kind = questionKind;
  await command("ask_explanation", payload);
  clearTimeout(state.questionTimer);
  $("questionStatus").classList.add("hidden");
}
document.querySelectorAll("#presetQuestions button").forEach((button) => {
  button.addEventListener("click", () => {
    submitExplanationQuestion(tr(button.dataset.questionKey), button.dataset.questionKind);
  });
});
$("askButton").addEventListener("click", () => {
  submitExplanationQuestion($("questionInput").value);
});
$("finishExplanationButton").addEventListener("click", async () => {
  queueSettledBrowseEvent();
  await flushTimelineEvents();
  await command("finish_explanation");
});
$("surveyPanel").addEventListener("submit", (event) => {
  event.preventDefault(); const form = new FormData(event.currentTarget);
  command("submit_survey", { coordination_understanding: Number(form.get("coordination_understanding")), ai_predictability: Number(form.get("ai_predictability")), interface_clarity: Number(form.get("interface_clarity")), comment: $("surveyComment").value.trim() });
});
[$("restartButton"), $("interruptedRestartButton")].forEach((button) => button.addEventListener("click", () => command("restart", restartPayload())));

window.addEventListener("keydown", (event) => {
  if (state.busy || !["task1","task2"].includes(state.view?.study?.stage) || ["INPUT","TEXTAREA"].includes(document.activeElement?.tagName)) return;
  const action = { ArrowUp:"UP", w:"UP", W:"UP", ArrowDown:"DOWN", s:"DOWN", S:"DOWN", ArrowLeft:"LEFT", a:"LEFT", A:"LEFT", ArrowRight:"RIGHT", d:"RIGHT", D:"RIGHT", " ":"WAIT" }[event.key];
  if (action) { event.preventDefault(); command("human_action", { action }); }
});
window.addEventListener("resize", () => { if (state.view) drawWarehouse(state.view, state.visualFrame); });

async function bootstrap() {
  buildSurvey(); setLanguage(DEFAULT_LOCALE, false);
  try { await render(await api("/api/view")); } catch (error) { showError(error); }
}

bootstrap();
