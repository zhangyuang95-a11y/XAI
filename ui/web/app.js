"use strict";

const $ = (id) => document.getElementById(id);
const PAGE_ID = (() => {
  const key = 'warehouse.threeTask.pageId';
  const existing = sessionStorage.getItem(key);
  if (existing) return existing;
  const created = crypto.randomUUID ? crypto.randomUUID() : `page-${Date.now()}`;
  sessionStorage.setItem(key, created);
  return created;
})();
const DEFAULT_LOCALE = "zh";
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
  questionTimer: null,
  lastPenaltyFrame: null,
  expandedBubble: null,
  bubbleAnswers: new Map(),
  pendingBubbleRequest: null,
  surveyFor: null,
};

const COPY = {
  zh: {
    appTitle: "双机器人协作配送实验", workflowDemo: "说明与演示", task1: "任务 1", task2: "任务 2", task3: "任务 3", survey: "问卷",
    warehouse: "6×7 仓库", liveScene: "协作配送现场", step: "步数", score: "总分", deliveries: "配送", collisions: "碰撞", shutdowns: "断电", detours: "绕路单位",
    shelf: "货架", pickup: "取货点", dropoff: "交付点", charger: "充电站", robots: "机器人",
    participantSetup: "参与者登记", welcome: "开始协作配送实验", overview: "你将固定控制机器人 1，与 AI 控制的机器人 2 完成三轮 120 步配送任务。",
    participantId: "参与者编号", agreement: "我已阅读并理解实验说明。", start: "开始实验",
    requiredDemo: "AI–AI 协作演示（可提前结束）", demoText: "您可以完整观看演示，也可以随时提前结束并开始任务 1。演示展示两台机器人认领、交付、协调让路和充电。",
    ruleJobs: "地图始终有两个未预分配的 A→B 共享任务。", ruleControl: "方向键、WASD 或按钮每次提交一个联合决策步；空格表示等待。",
    ruleCharge: "成功移动耗电 2；在充电站等待恢复 10；断电会提前结束本轮。", ruleScore: "计分：配送 +100、机器人碰撞 −200、断电 −50、每步 −1、参与者绕路每单位 −2；协作资源使用可能触发额外扣分。",
    playDemo: "播放演示", pauseDemo: "暂停演示", beginTask1: "开始任务 1", endDemoEarly: "提前结束演示并开始任务 1", roundInstruction: "控制机器人 1，与机器人 2 协作配送",
    roundHint: "你和机器人 2 都只根据同一移动前状态独立选动作，两个动作随后同时执行。", up: "上", down: "下", left: "左", right: "右", wait: "等待", spaceKey: "空格",
    task1Complete: "任务 1 已完成", task2Complete: "任务 2 已完成",
    askRobot2Title: "询问机器人2", askSystemTitle: "系统问答", liveExplanationHint: "询问机器人2刚才的行为；生成回答时任务会暂停。", question: "你的问题", questionPlaceholder: "询问机器人2最近几步的行为。", ask: "询问机器人2", answer: "机器人2", emptyExplanation: "暂时无法生成可靠回答，请重试。",
    presetWhyAction: "机器人2刚才为什么这样做？", presetWhyWait: "机器人2为什么等待？", presetCollision: "我们刚才为什么碰撞？", presetHumanInfluence: "我的动作影响机器人2了吗？", presetGoal: "机器人2当前想做什么？", presetEnergy: "机器人2需要充电吗？", presetChargerPenalty: "刚才为什么扣了50分？",
    surveyTitle: "结束问卷", surveyHint: "请对以下陈述按 1（非常不同意）到 5（非常同意）评分。", comment: "可选意见", submitSurvey: "提交问卷",
    complete: "实验完成", saved: "记录已保存。", task1Score: "任务 1 得分", task2Score: "任务 2 得分", task3Score: "任务 3 得分", restart: "开始新的参与者",
    interrupted: "本轮已中断", interruptedHint: "此实验已在另一页面继续，或服务恢复后旧运行被放弃。请重新开始。", desktopRequired: "请使用宽度至少 1024 像素的桌面或笔记本电脑。",
    participant: "参与者", ai: "AI", battery: "电量", cargo: "承运", none: "无", available: "可认领", carried: "运输中", carrier: "承运者",
    coordinationUnderstanding: "我理解如何与机器人 2 协调。", aiPredictability: "机器人 2 的行为对我而言是可预测的。", interfaceClarity: "界面与计分信息清晰易懂。",
    explanationClarity: "Task 2 的解释让我理解机器人 2 当时的行为。", explanationUsefulness: "Task 2 的解释帮助我决定如何配合。", questionHelpfulness: "Task 2 的问答帮助我理解所选动作。", notUsed: "未使用／不适用",
    deliveryScore: "配送得分", collisionPenalty: "碰撞扣分（每次 −200）", shutdownPenalty: "断电扣分", timePenalty: "步数扣分", detourPenalty: "绕路扣分", chargerOccupancyPenalty: "占桩扣分（每次 −50）",
    loading: "处理中…", requiredFields: "请填写参与者编号并确认已阅读说明。", requestFailed: "操作失败", taskLabel: "任务", roundScore: "本轮得分",
    action: "动作", requestedAction: "请求", executedAction: "实际", batteryChange: "电量",
    transitionActions: "动作", causalFrameNote: "双方从同一移动前状态决策并同步执行。", workingExplanation: "正在根据最近的人机交互生成回答…", stillWorking: "仍在生成，请稍候…",
    eventPickup: "取货", eventDelivery: "交付", eventCharging: "充电", eventChargerQueue: "排队", eventYield: "让行", eventConflict: "冲突风险", eventCollision: "碰撞",
    askWhy: "为什么？", hideWhy: "收起", answerFrame: "所问步数",
  },
  en: {
    appTitle: "Two-Robot Collaborative Delivery Study", workflowDemo: "Instructions & demo", task1: "Task 1", task2: "Task 2", task3: "Task 3", survey: "Survey",
    warehouse: "6×7 warehouse", liveScene: "Collaborative delivery", step: "Steps", score: "Score", deliveries: "Deliveries", collisions: "Collisions", shutdowns: "Shutdowns", detours: "Detour units",
    shelf: "Shelf", pickup: "Pickup A", dropoff: "Drop-off B", charger: "Charger", robots: "Robots",
    participantSetup: "Participant setup", welcome: "Start the collaborative delivery study", overview: "You will always control robot 1 and complete three 120-step delivery rounds with AI-controlled robot 2.",
    participantId: "Participant ID", agreement: "I have read and understood the study instructions.", start: "Start study",
    requiredDemo: "AI–AI collaboration demonstration", demoText: "You may watch the complete standardized demonstration or finish it early and begin Task 1 at any time. It shows both robots claiming, delivering, yielding, and charging.",
    ruleJobs: "The map always contains two unassigned shared A-to-B jobs.", ruleControl: "Arrow keys, WASD, or a button submits one joint decision step; Space means wait.",
    ruleCharge: "A successful move costs 2 battery; waiting at the charger restores 10; shutdown ends the round.", ruleScore: "Score: +100 delivery, −200 robot collision, −50 shutdown, −1 per step, and −2 per human detour unit; shared-resource use may incur an additional penalty.",
    playDemo: "Play demonstration", pauseDemo: "Pause demonstration", beginTask1: "Begin Task 1", endDemoEarly: "Finish demo early and begin Task 1", roundInstruction: "Control robot 1 and collaborate with robot 2",
    roundHint: "You and robot 2 choose independently from the same pre-move state; both actions then execute simultaneously.", up: "Up", down: "Down", left: "Left", right: "Right", wait: "Wait", spaceKey: "Space",
    task1Complete: "Task 1 complete", task2Complete: "Task 2 complete",
    askRobot2Title: "Ask Robot 2", askSystemTitle: "System questions", liveExplanationHint: "Ask about what Robot 2 just did. The task pauses while the answer is prepared.", question: "Your question", questionPlaceholder: "Ask Robot 2 about the last few steps.", ask: "Ask Robot 2", answer: "Robot 2", emptyExplanation: "No grounded answer was available. Please try again.",
    presetWhyAction: "Why did Robot 2 do that?", presetWhyWait: "Why did Robot 2 wait?", presetCollision: "Why did we just collide?", presetHumanInfluence: "Did my action affect Robot 2?", presetGoal: "What is Robot 2 trying to do?", presetEnergy: "Does Robot 2 need to charge?", presetChargerPenalty: "Why did I lose 50 points just now?",
    surveyTitle: "Final survey", surveyHint: "Rate each statement from 1 (strongly disagree) to 5 (strongly agree).", comment: "Optional comment", submitSurvey: "Submit survey",
    complete: "Study complete", saved: "The record has been saved.", task1Score: "Task 1 score", task2Score: "Task 2 score", task3Score: "Task 3 score", restart: "Start a new participant",
    interrupted: "Run interrupted", interruptedHint: "This run continued in another page or was abandoned during recovery. Please restart.", desktopRequired: "Use a desktop or laptop at least 1024 pixels wide.",
    participant: "Participant", ai: "AI", battery: "Battery", cargo: "Carrying", none: "None", available: "Available", carried: "In transit", carrier: "Carrier",
    coordinationUnderstanding: "I understand how to coordinate with robot 2.", aiPredictability: "Robot 2's behavior is predictable to me.", interfaceClarity: "The interface and scoring information are clear.",
    explanationClarity: "The Task 2 explanations helped me understand Robot 2's actions.", explanationUsefulness: "The Task 2 explanations helped me decide how to cooperate.", questionHelpfulness: "The Task 2 answers helped me understand the selected action.", notUsed: "Not used / not applicable",
    deliveryScore: "Delivery points", collisionPenalty: "Collision penalty (−200 each)", shutdownPenalty: "Shutdown penalty", timePenalty: "Step penalty", detourPenalty: "Detour penalty", chargerOccupancyPenalty: "Charger occupancy penalty (−50 each)",
    loading: "Working…", requiredFields: "Enter a participant ID and confirm the instructions.", requestFailed: "Request failed", taskLabel: "Task", roundScore: "Round score",
    action: "Action", requestedAction: "Requested", executedAction: "Executed", batteryChange: "Battery",
    transitionActions: "Actions", causalFrameNote: "Both agents decide from the same pre-move state and execute simultaneously.", workingExplanation: "Answering from your recent Human–AI interaction…", stillWorking: "Still generating—please wait…",
    eventPickup: "Pickup", eventDelivery: "Delivery", eventCharging: "Charging", eventChargerQueue: "Queue", eventYield: "Yield", eventConflict: "Conflict risk", eventCollision: "Collision",
    askWhy: "Ask why", hideWhy: "Hide", answerFrame: "Answered step",
  },
};

Object.assign(COPY.zh, {
  task2TransitionHint: "任务 2 将从新的机器人状态开始；只有 A 组可以询问机器人 2。",
  task3TransitionHint: "任务 3 从新的机器人状态开始，两组均不提供解释。",
  beginTask2: "开始任务 2",
  beginTask3: "开始任务 3",
  ruleCharge: "成功移动耗电 2；在充电站等待恢复 10；断电会提前结束本轮。",
  testCondition: "开发测试条件",
  conditionAuto: "自动区组分配",
  conditionExplanation: "A 组（有解释）",
  conditionControl: "B 组（无解释）",
  testConditionHint: "仅用于界面测试；数据写入独立的 development 命名空间。",
  assignedTestCondition: "当前测试条件",
  groupATitle: "您已分配到 A 组（有解释）",
  groupADescription: "任务 1 和任务 3 无解释；任务 2 可以点击“为什么？”并询问机器人 2。",
  groupBTitle: "您已分配到 B 组（无解释）",
  groupBDescription: "三轮任务均不显示解释面板。",
  questionTarget: "提问对象",
  robot1Option: "机器人 1（AI）",
  robot2Option: "机器人 2（AI）",
  questionPlaceholder: "询问机器人2最近几步的行为。",
  ask: "询问机器人2",
  temporaryNetworkError: "临时网络连接中断，请重试；当前进度已保留。",
});
Object.assign(COPY.en, {
  task2TransitionHint: "Task 2 begins with a fresh robot state. Only Group A can ask Robot 2 questions.",
  task3TransitionHint: "Task 3 begins with a fresh robot state. Neither group receives explanations.",
  beginTask2: "Begin Task 2",
  beginTask3: "Begin Task 3",
  ruleCharge: "A successful move costs 2 battery; waiting at the charger restores 10; shutdown ends the round.",
  testCondition: "Development test condition",
  conditionAuto: "Automatic block allocation",
  conditionExplanation: "Group A (explanations)",
  conditionControl: "Group B (no explanations)",
  testConditionHint: "For interface testing only; records use the isolated development namespace.",
  assignedTestCondition: "Current test condition",
  groupATitle: "You are assigned to Group A (explanations)",
  groupADescription: "Tasks 1 and 3 have no explanations. In Task 2, you can click Ask why and question Robot 2.",
  groupBTitle: "You are assigned to Group B (no explanations)",
  groupBDescription: "Explanations are not shown in any of the three tasks.",
  questionTarget: "Robot to ask about",
  robot1Option: "Robot 1 (AI)",
  robot2Option: "Robot 2 (AI)",
  questionPlaceholder: "Ask Robot 2 about the last few steps.",
  ask: "Ask Robot 2",
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

function showError(error) {
  $("toastText").textContent = error instanceof Error ? error.message : String(error);
  $("toast").classList.remove("hidden");
}

function setBusy(value) {
  state.busy = value;
  document.querySelectorAll("button").forEach((button) => {
    if (button.id === "toastClose") return;
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
  if (name === 'human_action') {
    state.expandedBubble = null;
    state.pendingBubbleRequest = null;
    $('aiActionBubble').classList.add('hidden');
  }
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
  $("languageButtonLabel").textContent = state.locale === "zh" ? "English" : "中文";
  const surveyAnswers = Object.fromEntries(
    [...document.querySelectorAll('#surveyQuestions input:checked')].map((input) => [input.name, input.value])
  );
  buildSurvey();
  Object.entries(surveyAnswers).forEach(([name, value]) => {
    const selected = [...document.querySelectorAll('#surveyQuestions input')]
      .find((input) => input.name === name && input.value === value);
    if (selected) selected.checked = true;
  });
  if (state.view && rerender) {
    state.view = {
      ...state.view,
      study: { ...(state.view.study || {}), locale: requestedLocale },
    };
    void render(state.view, { skipAnimation: true });
  }
  if (notify && allowed("set_language")) command("set_language", { locale: requestedLocale });
}

// Okabe-Ito-inspired task colors, deliberately separated from the blue and
// orange robot colors.  Task IDs map deterministically so A/B retain the same
// color after task refreshes and in both languages.
const TASK_PALETTE = ["#009E73", "#CC79A7", "#B79F00", "#7A5AF8", "#5D6B7A"];
function taskColor(taskId, fallbackIndex = 0) {
  const match = String(taskId || "").match(/(\d+)$/);
  const index = match ? Math.max(0, Number(match[1]) - 1) : fallbackIndex;
  return TASK_PALETTE[index % TASK_PALETTE.length];
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
  canvas.dataset.taskColors = JSON.stringify(
    Object.fromEntries(tasks.map((task, index) => [task.task_id, taskColor(task.task_id, index)]))
  );
  tasks.forEach((task, index) => {
    const color = taskColor(task.task_id, index);
    if (task.status === "available") {
      const [x,y] = cell(task.pickup_position); ctx.fillStyle = color; ctx.beginPath(); ctx.arc(x+size/2,y+size/2,size*.31,0,Math.PI*2); ctx.fill(); ctx.fillStyle="#ffffff"; ctx.fillText(`A${index+1}`,x+size/2,y+size/2);
    }
    const [x,y] = cell(task.delivery_position); ctx.fillStyle = color; ctx.beginPath(); ctx.arc(x+size/2,y+size/2,size*.31,0,Math.PI*2); ctx.fill(); ctx.fillStyle="#ffffff"; ctx.fillText(`B${index+1}`,x+size/2,y+size/2);
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
      const cargoIndex = Math.max(0, Number(String(agent.carrying_label).replace(/\D/g, "")) - 1);
      ctx.fillStyle = TASK_PALETTE[cargoIndex % TASK_PALETTE.length];
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

function renderRobots(agents, transition = null, showActions = false, agentControl = {}, revealOutcome = false) {
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
      const executed = revealOutcome ? motion.executed_action : proposed;
      const actionText = revealOutcome && proposed && proposed !== executed
        ? `<span><b>${tr("requestedAction")}:</b> ${actionLabel(proposed)} · <b>${tr("executedAction")}:</b> ${actionLabel(executed)}</span>`
        : `<span><b>${tr("action")}:</b> ${actionLabel(executed)}</span>`;
      const delta = revealOutcome ? Number(motion.battery_delta || 0) : 0;
      const deltaText = delta === 0 ? "" : `<em>${tr("batteryChange")} ${delta > 0 ? "+" : ""}${delta.toFixed(0)}</em>`;
      motionMarkup = `<div class="motion-status ${motion.blocked ? "blocked" : ""} ${motion.charging ? "charging" : ""}">${actionText}${deltaText}</div>`;
    }
    const robotNumber = agent.id === "robot_1" ? "1" : "2";
    const controllerLabel = controller === "human" ? tr("participant") : tr("ai");
    article.innerHTML = `<header><strong>${controllerLabel} · R${robotNumber}</strong></header>${motionMarkup || `<small>${tr("action")}: —</small>`}`;
    return article;
  }));
}

function renderActionBubble(view, revealOutcome) {
  const bubble = $("aiActionBubble");
  const payload = view.study?.action_bubble;
  const enabled = Boolean(
    revealOutcome
    && view.study?.stage === "task2"
    && view.study?.condition === "explanation"
    && payload?.target_agent === "robot_2"
    && payload?.run_id === view.study?.task_run_id
    && Number(payload?.frame) > 0
  );
  bubble.classList.toggle("hidden", !enabled);
  if (!enabled) return;
  const agent = (view.state?.agents || []).find((item) => item.id === "robot_2");
  if (!agent) { bubble.classList.add("hidden"); return; }
  const canvas = $("warehouseCanvas");
  const cols = Math.max(1, Number(view.map?.cols || 1));
  const rows = Math.max(1, Number(view.map?.rows || 1));
  const width = Math.max(1, canvas.clientWidth);
  const height = Math.max(1, canvas.clientHeight);
  const size = Math.min((width - 36) / cols, (height - 36) / rows);
  const originX = (width - cols * size) / 2;
  const originY = (height - rows * size) / 2;
  const left = originX + (Number(agent.position[1]) + .5) * size;
  const top = originY + (Number(agent.position[0]) + .5) * size;
  const key = `${view.study.protocol_version}:warehouse:${view.study.run_id}:task2:${payload.run_id}:${payload.frame}`;
  const expanded = state.expandedBubble === key;
  const answer = state.bubbleAnswers.get(key);
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'bubble-toggle';
  button.textContent = tr(expanded ? 'hideWhy' : 'askWhy');
  button.disabled = state.busy;
  button.setAttribute('aria-expanded', String(expanded));
  button.setAttribute('aria-label', `${button.textContent} · ${tr('step')} ${payload.frame}`);
  button.addEventListener('click', async (event) => {
    event.stopPropagation();
    if (state.busy) return;
    if (state.expandedBubble === key) {
      state.expandedBubble = null;
      renderActionBubble(state.view, true);
      return;
    }
    if (answer) {
      state.expandedBubble = key;
      renderAnswer(answer);
      renderActionBubble(state.view, true);
      return;
    }
    const requestId = operationId();
    state.pendingBubbleRequest = requestId;
    const result = await command('ask_explanation', {
      question: tr('presetWhyAction'), question_kind: 'action', target_agent: 'robot_2',
      action_run_id: payload.run_id, action_stage: 'task2', action_frame: Number(payload.frame),
      requested_language: localeCode(), request_id: requestId,
    });
    const report = result?.view?.last_explanation;
    if (state.pendingBubbleRequest === requestId && report?.request_id === requestId
      && report?.run_id === payload.run_id && Number(report?.requested_action_frame) === Number(payload.frame)
      && state.view?.study?.stage === 'task2' && state.view?.study?.condition === 'explanation'
      && state.view?.study?.task_run_id === payload.run_id
      && Number(state.view?.study?.action_bubble?.frame) === Number(payload.frame)) {
      state.bubbleAnswers.set(key, report);
      state.expandedBubble = key;
      renderActionBubble(state.view, true);
    }
    if (state.pendingBubbleRequest === requestId) state.pendingBubbleRequest = null;
  });
  bubble.replaceChildren(button);
  if (expanded && answer) {
    const explanation = document.createElement('p');
    explanation.textContent = state.locale === 'zh' ? answer.answer_zh : answer.answer_en;
    bubble.appendChild(explanation);
  }
  bubble.classList.toggle('compact', !expanded);
  bubble.style.left = `${Math.max(92, Math.min(width - 92, left))}px`;
  bubble.style.top = `${top}px`;
  bubble.classList.toggle("below", top < 86);
}

function renderScores(snapshot) {
  const breakdown = snapshot.score_breakdown || {};
  $("stepValue").textContent = `${snapshot.frame || 0} / 120`;
  $("scoreValue").textContent = Math.round(snapshot.user_score || 0);
  $("deliveryValue").textContent = snapshot.total_deliveries || 0;
  $("collisionValue").textContent = snapshot.robot_collision_events || 0;
  $("shutdownValue").textContent = snapshot.shutdown_count || 0;
  $("detourValue").textContent = Number(snapshot.human_route_regret_units || 0).toFixed(1);
  const labels = { delivery: tr("deliveryScore"), robot_collision: tr("collisionPenalty"), shutdown: tr("shutdownPenalty"), time: tr("timePenalty"), human_detour: tr("detourPenalty"), shared_charger_occupancy: tr("chargerOccupancyPenalty") };
  $("scoreBreakdown").replaceChildren(...Object.entries(labels).map(([key,label]) => {
    const row = document.createElement("div"); row.innerHTML = `<span>${label}</span><strong>${Math.round(breakdown[key] || 0)}</strong>`; return row;
  }));
  const penalty = (snapshot.rule_events || []).find((event) => event?.event === "shared_charger_occupancy");
  const flash = $("penaltyFlash");
  if (penalty && Number(snapshot.frame || 0) !== state.lastPenaltyFrame) {
    state.lastPenaltyFrame = Number(snapshot.frame || 0);
    flash.textContent = "−50";
    flash.classList.remove("hidden");
    document.body.classList.add("charger-penalty-flash");
    window.setTimeout(() => {
      flash.classList.add("hidden");
      document.body.classList.remove("charger-penalty-flash");
    }, 900);
  }
}

function renderWorkflow(stage, condition) {
  const order = ["instructions", "task1", "task2", "task3", "survey"];
  const normalized = stage === "completed" ? "survey" : stage;
  const current = stage === "task1_complete" ? 2 : stage === "task2_complete" ? 3 : Math.max(0, order.indexOf(normalized));
  ["workflowInstructions","workflowTask1","workflowTask2","workflowTask3","workflowSurvey"].forEach((id,index) => {
    $(id).classList.toggle("active", index === current);
    $(id).classList.toggle("done", index < current || stage === "completed");
  });
}

function renderStage() {
  if (!state.view) return;
  const study = state.view.study || {};
  const stage = study.stage || "idle";
  $("sceneTitle").textContent = tr("liveScene");
  const panels = { idle: "setupPanel", instructions: "instructionsPanel", task1: "roundPanel", task1_complete: "task1CompletePanel", task2: "roundPanel", task2_complete: "task2CompletePanel", task3: "roundPanel", survey: "surveyPanel", completed: "completePanel", abandoned: "interruptedPanel" };
  ["setupPanel","instructionsPanel","roundPanel","task1CompletePanel","task2CompletePanel","surveyPanel","completePanel","interruptedPanel"].forEach((id) => $(id).classList.toggle("hidden", panels[stage] !== id));
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
  if (["task1", "task2", "task3"].includes(stage)) {
    $("roundBadge").textContent = stage.toUpperCase();
    const permitted = new Set(study.allowed_human_actions || []);
    document.querySelectorAll("#actionPad button").forEach((button) => {
      button.disabled = state.busy || (permitted.size > 0 && !permitted.has(button.dataset.action));
    });
  }
  const liveExplanationVisible = Boolean(
    stage === "task2"
    && study.condition === "explanation"
    && study.live_explanation_available
  );
  $("liveExplanationPanel").classList.toggle("hidden", !liveExplanationVisible);
  const penaltyQuestion = $("chargerPenaltyQuestionButton");
  const latestPenalty = study.latest_charger_penalty;
  const penaltyVisible = Boolean(liveExplanationVisible && latestPenalty?.event_id);
  penaltyQuestion.classList.toggle("hidden", !penaltyVisible);
  if (penaltyVisible) {
    penaltyQuestion.textContent = `${tr("presetChargerPenalty")} · ${tr("step")} ${latestPenalty.frame}`;
    penaltyQuestion.dataset.eventId = String(latestPenalty.event_id);
    penaltyQuestion.dataset.penaltyFrame = String(latestPenalty.frame);
  } else {
    penaltyQuestion.dataset.eventId = "";
    penaltyQuestion.dataset.penaltyFrame = "";
  }
  if (stage === "task1_complete") {
    const summary = study.round_summaries?.task1;
    $("controlTask1Score").textContent = Math.round(summary?.score ?? 0);
    $("beginTask2Button").disabled = state.busy || !allowed("begin_task2");
    $("beginTask2Button").dataset.locked = allowed("begin_task2") ? "false" : "true";
  }
  if (stage === "task2_complete") {
    const summary = study.round_summaries?.task2;
    $("controlTask2Score").textContent = Math.round(summary?.score ?? 0);
    $("beginTask3Button").disabled = state.busy || !allowed("begin_task3");
    $("beginTask3Button").dataset.locked = allowed("begin_task3") ? "false" : "true";
  }
  if (stage === "survey" && state.surveyFor !== study.group_code) buildSurvey();
  if (stage === "completed") {
    $("finalTask1").textContent = Math.round(study.round_summaries?.task1?.score ?? 0);
    $("finalTask2").textContent = Math.round(study.round_summaries?.task2?.score ?? 0);
    $("finalTask3").textContent = Math.round(study.round_summaries?.task3?.score ?? 0);
  }
}

function renderAnswer(report) {
  if (state.view?.study?.stage !== 'task2' || state.view?.study?.condition !== 'explanation') return;
  const text = (state.locale === 'zh' ? report?.answer_zh : report?.answer_en)
    || report?.explanation_document?.text || report?.explanation || "";
  if (!text.trim()) {
    showError(new Error(tr("emptyExplanation")));
    return;
  }
  $("answerText").textContent = text;
  $("answerFrame").textContent = `${tr('answerFrame')} ${report.anchor_frame ?? report.selected_timeline_frame ?? '—'}`;
  $("answerPanel").classList.remove("hidden");
}

function transitionBeforeView(view) {
  const transition = view.transition;
  if (!transition?.before_state) return view;
  return {
    ...view,
    state: transition.before_state,
    study: {
      ...(view.study || {}),
      stage: transition.loop
        ? view.study?.stage
        : (transition.before_stage || view.study?.stage),
      progress: Number(transition.from_frame || 0),
    },
  };
}

function paintView(view, revealOutcome = false) {
  state.view = view;
  document.body.dataset.studyStage = view.study?.stage || "idle";
  document.body.dataset.stateVersion = String(view.study?.state_version ?? 0);
  const stage = view.study?.stage || "idle";
  const showActions = stage === "instructions";
  renderRobots(
    view.state?.agents || [],
    view.transition,
    showActions,
    view.timeline?.agent_control || {},
    revealOutcome,
  );
  renderScores(view.state || {});
  renderActionBubble(view, revealOutcome);
  renderStage();
}

async function render(view, options = {}) {
  const previous = state.view?.study;
  const requestedStage = view.study?.stage || "idle";
  if (requestedStage !== "idle" && view.study?.locale) {
    const requestedLocale = view.study.locale === "en" ? "en" : "zh";
    if (requestedLocale !== state.locale) setLanguage(requestedLocale, false, false);
  }
  const stage = view.study?.stage || "idle";
  if (stage !== 'task2' || view.study?.condition !== 'explanation'
      || view.study?.task_run_id !== previous?.task_run_id) {
    state.expandedBubble = null;
    state.pendingBubbleRequest = null;
    if (stage !== 'task2' || view.study?.condition !== 'explanation') state.bubbleAnswers.clear();
  } else if (Number(view.study?.action_bubble?.frame) !== Number(previous?.action_bubble?.frame)) {
    state.expandedBubble = null;
    state.pendingBubbleRequest = null;
  }
  const beforeView = transitionBeforeView(view);
  const newTransition = Boolean(view.transition && previous
    && Number(view.study?.progress) !== Number(previous.progress) && !options.skipAnimation);
  if (newTransition) {
    paintView(beforeView);
    await animateOnce(beforeView, view.transition, 400);
    paintView(view, true);
    drawWarehouse(view);
  } else {
    cancelMotion();
    paintView(view, true);
    drawWarehouse(view);
  }
  if (stage === 'task2' && view.study?.condition === 'explanation' && view.last_explanation) {
    renderAnswer(view.last_explanation);
  } else if (stage !== 'task2' || view.study?.condition !== 'explanation') {
    $("answerPanel").classList.add("hidden");
    $("answerText").textContent = '';
    $("questionStatus").classList.add("hidden");
  }
}

function buildSurvey() {
  const group = state.view?.study?.group_code;
  const questions = [
    ["coordination_understanding", tr("coordinationUnderstanding")],
    ["ai_predictability", tr("aiPredictability")],
    ["interface_clarity", tr("interfaceClarity")],
  ];
  if (group === 'A') questions.push(
    ["explanation_clarity", tr("explanationClarity")],
    ["explanation_usefulness", tr("explanationUsefulness")],
    ["question_helpfulness", tr("questionHelpfulness")],
  );
  $("surveyQuestions").replaceChildren(...questions.map(([name,label]) => {
    const section = document.createElement("section"); section.className = "survey-question";
    const optional = group === 'A' && (name.startsWith('explanation_') || name === 'question_helpfulness');
    section.innerHTML = `<p>${label}</p><div class="scale">${[1,2,3,4,5].map((value) => `<label><input type="radio" name="${name}" value="${value}" required><span>${value}</span></label>`).join("")}${optional ? `<label><input type="radio" name="${name}" value="na"><span>${tr('notUsed')}</span></label>` : ''}</div>`;
    return section;
  }));
  state.surveyFor = group || 'none';
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
$("beginTask3Button").addEventListener("click", () => command("begin_task3"));
document.querySelectorAll("#actionPad button").forEach((button) => button.addEventListener("click", () => command("human_action", { action: button.dataset.action })));
async function submitExplanationQuestion(question, questionKind = null, anchor = null) {
  const prompt = String(question || "").trim();
  if (!prompt || state.busy || !allowed("ask_explanation")) return;
  $("questionInput").value = prompt;
  $("questionStatus").textContent = tr("workingExplanation");
  $("questionStatus").classList.remove("hidden");
  clearTimeout(state.questionTimer);
  state.questionTimer = setTimeout(() => {
    $("questionStatus").textContent = tr("stillWorking");
  }, 5000);
  const payload = {
    question: prompt,
    target_agent: "robot_2",
  };
  if (questionKind) payload.question_kind = questionKind;
  if (anchor?.event_id) payload.penalty_event_id = String(anchor.event_id);
  if (anchor?.frame != null) payload.penalty_frame = Number(anchor.frame);
  await command("ask_explanation", payload);
  clearTimeout(state.questionTimer);
  $("questionStatus").classList.add("hidden");
}
document.querySelectorAll("#presetQuestions button").forEach((button) => {
  if (button.id === "chargerPenaltyQuestionButton") return;
  button.addEventListener("click", () => {
    submitExplanationQuestion(tr(button.dataset.questionKey), button.dataset.questionKind);
  });
});
$("chargerPenaltyQuestionButton").addEventListener("click", () => {
  const penalty = state.view?.study?.latest_charger_penalty;
  submitExplanationQuestion(
    tr("presetChargerPenalty"),
    "charger_penalty",
    penalty,
  );
});
$("askButton").addEventListener("click", () => {
  submitExplanationQuestion($("questionInput").value);
});
$("surveyPanel").addEventListener("submit", (event) => {
  event.preventDefault(); const form = new FormData(event.currentTarget);
  const value = (name) => form.get(name) === 'na' ? 'na' : form.has(name) ? Number(form.get(name)) : null;
  command("submit_survey", {
    coordination_understanding: value('coordination_understanding'),
    ai_predictability: value('ai_predictability'), interface_clarity: value('interface_clarity'),
    explanation_clarity: value('explanation_clarity'),
    explanation_usefulness: value('explanation_usefulness'),
    question_helpfulness: value('question_helpfulness'),
    comment: $("surveyComment").value.trim(),
  });
});
[$("restartButton"), $("interruptedRestartButton")].forEach((button) => button.addEventListener("click", () => command("restart", restartPayload())));

window.addEventListener("keydown", (event) => {
  if (state.busy || !["task1","task2","task3"].includes(state.view?.study?.stage) || ["INPUT","TEXTAREA"].includes(document.activeElement?.tagName)) return;
  const action = { ArrowUp:"UP", w:"UP", W:"UP", ArrowDown:"DOWN", s:"DOWN", S:"DOWN", ArrowLeft:"LEFT", a:"LEFT", A:"LEFT", ArrowRight:"RIGHT", d:"RIGHT", D:"RIGHT", " ":"WAIT" }[event.key];
  if (action) { event.preventDefault(); command("human_action", { action }); }
});
window.addEventListener("resize", () => { if (state.view) drawWarehouse(state.view, state.visualFrame); });

async function bootstrap() {
  buildSurvey(); setLanguage(DEFAULT_LOCALE, false);
  try { await render(await api("/api/view")); } catch (error) { showError(error); }
}

bootstrap();
