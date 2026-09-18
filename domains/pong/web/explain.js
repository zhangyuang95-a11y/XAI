/* Frame-bound, question-aware language for the actual submitted controller plan. */
(() => {
  const label = id => id ? `${id.startsWith('B') ? '合作大球' : '小球'}${id}` : '来球';
  const direction = action => ({ left: '向左移动', right: '向右移动', stay: '停留' }[action] || '行动');
  const distance = value => value !== null && value !== undefined && Number.isFinite(Number(value))
    ? `${Math.round(Math.max(0, Number(value)))}格` : '距离未知';
  const selectedBall = (question, fallback) => (String(question).toUpperCase().match(/(?:A[123]|B[12])/) || [fallback])[0];
  const eventId = event => event.encounter_id || `${event.ball_id}:${event.frame}`;
  function encounterText(event) {
    if (!event) return '这一帧没有接球结算。';
    const name = label(event.ball_id);
    if (event.outcome === 'caught') {
      return event.kind === 'large' ? `${name}由两块球拍分别覆盖左右接触点，实际接住了。`
        : `${name}由至少一块球拍覆盖，实际接住了。`;
    }
    if (event.kind !== 'large') return `${name}经过接球线时，两块球拍都没有覆盖接触点，因此漏接，计1次。`;
    const p = event.player_coverage || [];
    const a = event.ai_coverage || [];
    const left = p[0] && a[1]; const right = p[1] && a[0];
    if (left || right) return `${name}的覆盖记录与漏接结果不一致，需要检查该次回放。`;
    const player = p.some(Boolean) ? '你覆盖了一个接触点' : '你没有覆盖接触点';
    const ai = a.some(Boolean) ? '机器人2覆盖了一个接触点' : '机器人2没有覆盖接触点';
    return `${name}经过接球线时，${player}，${ai}，但没有分别覆盖左右两侧，因此漏接，计3次。`;
  }
  function relevantEncounter(history, index, ballId, opportunityId, recent) {
    const lower = recent ? Math.max(0, index - 180) : index;
    for (let i = index; i >= lower; i -= 1) {
      const events = history[i]?.events || [];
      const found = events.find(event => event.event === 'encounter'
        && (!ballId || event.ball_id === ballId)
        && (!opportunityId || eventId(event) === opportunityId));
      if (found) return { ...found, _review_frame: history[i].frame };
    }
    return null;
  }
  function assignment(evidence) {
    const id = evidence?.target_ball_id;
    if (!id) return '当前没有可核验的接球分工。';
    if (evidence.requires_partner) {
      const own = evidence.contact_side === 'right' ? '右侧' : '左侧';
      const other = own === '左侧' ? '右侧' : '左侧';
      const status = evidence.partner_status === 'covered' ? '你已经覆盖了另一侧。'
        : evidence.partner_status === 'reachable' ? `你目前距${other}约${distance(evidence.partner_distance)}，还需要实际到位。`
          : '你目前无法及时覆盖另一侧，这次合作接球没有把握。';
      return `机器人2负责${label(id)}的${own}，需要你守${other}。${status}`;
    }
    const remaining = Number(evidence.self_distance);
    const position = evidence.holding ? '已经守住预计接球位置'
      : Number.isFinite(remaining) && remaining < 1 ? '距预计接球位置不足1格'
        : `距预计接球位置还需移动约${distance(evidence.self_distance)}`;
    return `机器人2负责${label(id)}的接球位置，${position}。`;
  }
  function bubble(decision) {
    const evidence = decision?.planEvidence;
    if (!evidence) return { target_ball_id: null, text: '我正在确认下一次接球位置。', detail: '' };
    const id = evidence.target_ball_id;
    if (!id) return { target_ball_id: null, text: '当前没有可核验的接球任务，我先观察下一次机会。', detail: '' };
    const main = evidence.requires_partner
      ? `我${evidence.holding ? '已守住' : '去守'}${label(id)}的${evidence.contact_side === 'right' ? '右侧' : '左侧'}；请你接另一侧。`
      : `我${evidence.holding ? '已守住' : '去接'}${label(id)}；你可以关注另一颗球。`;
    const next = (evidence.planned_sequence || [])[1];
    const forgoneSmall = evidence.requires_partner ? (evidence.alternatives || []).find(item =>
      item.kind === 'small' && item.viable && item.in_planning_window
      && item.plan_score_if_first < evidence.candidate_scores?.baseline) : null;
    const detail = evidence.requires_partner
      ? (evidence.partner_status === 'covered' ? '你已覆盖另一侧，我会守到接球结束。'
        : evidence.partner_status === 'reachable' ? `你距另一侧约${distance(evidence.partner_distance)}，请及时到位；漏掉大球计3次。`
          : '你目前赶不到另一侧；我会继续比较其他可救的球。')
      : `我距预计接球位置约${distance(evidence.self_distance)}。${next?.startsWith('B')
        ? `接住后当前预计仍能赶到${label(next)}，你可先准备合作侧。` : ''}`;
    const usefulDetail = forgoneSmall ? `${detail}先接${label(forgoneSmall.ball_id)}会少保住加权机会。` : detail;
    return { target_ball_id: id, text: main, detail: usefulDetail };
  }
  function answer(question, frame, history, index, fallbackBallId, counterfactual) {
    const q = String(question || '').trim();
    if (!q) return '请问一个具体问题，例如“为什么接B1”或“我该去哪里？”';
    const decision = frame?.decision || {};
    const evidence = decision.planEvidence;
    const mentioned = [...q.toUpperCase().matchAll(/(?:A[123]|B[12])/g)].map(match => match[0]);
    const target = selectedBall(q, fallbackBallId || evidence?.target_ball_id || null);
    const targetRow = evidence?.alternatives?.find(item => item.ball_id === target);
    const chosen = evidence?.target_ball_id;
    const parts = [];
    if (/如果|假如|要是/.test(q)) {
      parts.push(counterfactual || '这帧没有足够的可恢复状态，无法可靠模拟这个假设。');
    }
    if (/漏|没接住|没有接住|接住了吗|结果|发生/.test(q)) {
      const ownOpportunity = target === chosen ? evidence?.opportunity_id : null;
      const recent = /刚才|之前|上次|最近/.test(q);
      const event = relevantEncounter(history, index, target, recent ? null : ownOpportunity, recent);
      parts.push(event ? `${recent ? `最近一次在第${event._review_frame}帧：` : ''}${encounterText(event)}`
        : `所选第${frame?.frame ?? '当前'}帧没有${target ? label(target) : '目标球'}的接球结算；请选对应事件${recent ? '或更靠近它的回放帧' : ''}。`);
    }
    if (/为什么不|为何不|怎么不|放弃|没去接/.test(q)) {
      if (!target) parts.push('请指出你问的是A1、A2、A3、B1还是B2。');
      else if (target === chosen) parts.push(`这次计划实际选择了${label(target)}；${assignment(evidence)}`);
      else if (!targetRow) parts.push(`当时没有${label(target)}即将到来的接球机会，不能把它说成已被放弃。`);
      else if (!targetRow.viable) parts.push(`${label(target)}在当时的落点预测中不能及时覆盖${targetRow.kind === 'large' ? '双方接触点' : '接球位置'}；${chosen ? `机器人2改守${label(chosen)}。` : '当前没有可行的替代分工。'}`);
      else if (!targetRow.in_planning_window) parts.push(`${label(target)}当时可达，但不在这次最多三个机会的短期比较中；${chosen ? `机器人2先守${label(chosen)}。` : '当前尚未分配目标。'}`);
      else if (targetRow.plan_score_if_first < evidence.candidate_scores?.baseline)
        parts.push(`先接${label(target)}会比当前计划少保住加权接球机会；${chosen ? `机器人2先守${label(chosen)}。` : '当前没有可行的目标。'}`);
      else parts.push(`${label(target)}当时也可达；机器人2沿当前承诺继续守${label(chosen)}，后续仍会重新比较。`);
    } else if (/为什么|为何|原因|干嘛|怎么会/.test(q) && !parts.length) {
      if (/停|不动|等待/.test(q)) parts.push(evidence?.holding
        ? `机器人2已经覆盖${label(chosen)}的分配接触点，留在这里等待本次接球结算。`
        : chosen ? `当前动作是${direction(decision.action)}，对应${label(chosen)}的接球计划。`
          : '当前没有可核验的接球分工，机器人2暂时停留。');
      else if (/换|改变|来回|左右/.test(q)) {
        const previous = history[index - 1]?.decision?.planEvidence?.target_ball_id;
        const settled = [history[index - 1], frame].flatMap(item => item?.events || [])
          .find(event => event.event === 'encounter' && event.ball_id === previous);
        parts.push(previous && previous !== chosen
          ? settled ? `${label(previous)}已经${settled.outcome === 'caught' ? '接住' : '漏接'}，机器人2这次改守${label(chosen)}。`
            : evidence?.reason === 'sequence_gain' ? `重新比较连续接球后，改守${label(chosen)}可保住更多加权机会。`
              : `此前负责${label(previous)}，这一帧改守${label(chosen)}；记录未给出足够证据证明更细的原因。`
          : '这一帧没有记录到目标切换；若问持续左右移动，请选一段时间。');
      }
      else parts.push(evidence ? `${assignment(evidence)}机器人2实际${direction(decision.action)}。` : '当前帧没有可核验的决策证据。');
    }
    if (/先接|来得及|赶得上|再接/.test(q) && !/如果|假如|要是/.test(q)) {
      const order = evidence?.planned_sequence || [];
      parts.push(order.length >= 2 ? `根据当前预测，计划顺序为${order.map(label).join('、')}；实际接球仍取决于后续位置与玩家行动。`
        : '当前没有经过验证的连续两球计划，不能保证先接另一颗球仍赶得上。');
    }
    if (mentioned.length >= 2 && /还是|比较|哪个|先.*再/.test(q)) {
      const [first, second] = mentioned;
      const rows = [first, second].map(id => evidence?.alternatives?.find(item => item.ball_id === id));
      if (rows.every(row => row && row.in_planning_window && row.plan_score_if_first >= 0)) {
        const preferred = rows[0].plan_score_if_first === rows[1].plan_score_if_first ? null
          : rows[0].plan_score_if_first > rows[1].plan_score_if_first ? first : second;
        parts.push(preferred ? `按当时最多三个接球机会的预测，先守${label(preferred)}能保住更多加权机会；实际顺序还要随球位更新。`
          : `${label(first)}与${label(second)}先处理的加权机会相同；当前承诺是${label(chosen)}。`);
      } else parts.push(`当时没有同时验证${label(first)}和${label(second)}的短期接球方案，不能可靠比较先后。`);
    }
    if (/我该|我应该|我去|怎么配合|帮你|哪边/.test(q)) parts.push(assignment(evidence));
    if (/哪个球|什么球|接什么|目标|正在接/.test(q) && !parts.length) parts.push(assignment(evidence));
    if (/NN|神经|规则|谁决定|谁控制|改选/.test(q)) parts.push(decision.intervened
      ? `NN建议${direction(decision.nnProposedAction)}，规则按当前接球计划改为${direction(decision.action)}。`
      : `NN建议${direction(decision.nnProposedAction)}，实际提交${direction(decision.action)}。`);
    if (/大球.*几|计分|扣几|漏.*算/.test(q)) parts.push('小球漏接计1次，合作大球漏接计3次。');
    if (!parts.length) return `可以根据这帧回答机器人2的目标、分工、漏接结果和动作原因。你是想问${label(chosen)}，还是另一颗球？`;
    return [...new Set(parts)].join(' ');
  }
  function answerRange(question, frames) {
    if (!frames?.length) return '这段时间没有可用回放。';
    const q = String(question || '').trim();
    const goals = [];
    for (const frame of frames) {
      const id = frame.decision?.planEvidence?.target_ball_id || null;
      if (id !== goals.at(-1)?.id) goals.push({ id, frame: frame.frame,
        reason: frame.decision?.planEvidence?.reason || null,
        evidence: frame.decision?.planEvidence || null });
    }
    const encounters = new Map();
    for (const frame of frames) for (const event of frame.events || []) {
      if (event.event === 'encounter') encounters.set(eventId(event), event);
    }
    const caught = [...encounters.values()].filter(event => event.outcome === 'caught');
    const missed = [...encounters.values()].filter(event => event.outcome === 'missed');
    const answers = [];
    if (/漏|接住|结果|发生/.test(q)) answers.push(`这段时间接住${caught.length}次、漏接${missed.length}次接球机会。${missed.slice(0, 3).map(encounterText).join(' ')}`);
    if (/等|停|不动/.test(q)) {
      const still = frames.filter(frame => frame.decision?.action === 'stay');
      const holding = still.filter(frame => frame.decision?.planEvidence?.holding);
      const unassigned = still.filter(frame => !frame.decision?.planEvidence?.target_ball_id);
      const parts = [`这段${frames.length}帧里，机器人2提交停留${still.length}帧。`];
      if (holding.length) parts.push(`其中${holding.length}帧已覆盖分配的接球点，需要守到球经过。`);
      if (unassigned.length) parts.push(`另有${unassigned.length}帧没有可核验的短期接球分工。`);
      if (still.length > holding.length + unassigned.length)
        parts.push('其余停留帧不能仅凭这段记录归因于守位，请查看相应单帧。');
      answers.push(parts.join(''));
    }
    if (/换|改变|来回|左右|为什么/.test(q) && !/为什么漏|为什么等|为什么停/.test(q)) {
      const named = goals.filter(item => item.id);
      if (named.length < 2) answers.push(`这段时间没有确认的目标切换；主要负责${label(named[0]?.id)}。`);
      else {
        const changes = named.slice(1, 4).map((item, index) => {
          const previous = named[index];
          const settled = [...encounters.values()].find(event => event.ball_id === previous.id
            && event.frame >= previous.frame && event.frame <= item.frame);
          const why = settled ? `${label(previous.id)}已${settled.outcome === 'caught' ? '接住' : '漏接'}，本次机会结束`
            : item.reason === 'sequence_gain' ? '重新比较后，连续接球方案能减少加权漏接'
              : `原来负责的${label(previous.id)}不再是当时选定的短期方案`;
          return `${why}，于是转向${label(item.id)}`;
        });
        answers.push(`开始负责${label(named[0].id)}；${changes.join('；')}。`);
      }
    }
    if (/NN|神经|规则|谁决定|谁控制|改选/.test(q)) {
      const decisions = new Map(frames.filter(frame => frame.decision?.decisionId)
        .map(frame => [frame.decision.decisionId, frame.decision]));
      const interventions = [...decisions.values()].filter(decision => decision.intervened).length;
      answers.push(`这段有${decisions.size}次独立决策，其中${interventions}次规则改选了冻结NN的建议；这些动作来自规则协调的混合控制器。`);
    }
    if (/我该|我应该|怎么配合|哪边/.test(q)) answers.push(assignment(frames.at(-1)?.decision?.planEvidence));
    if (/先接|来得及|赶得上|再接/.test(q)) {
      const sequence = frames.at(-1)?.decision?.planEvidence?.planned_sequence || [];
      answers.push(sequence.length >= 2 ? `这段末尾的预测顺序是${sequence.map(label).join('、')}；后续实际行动仍需重新判断。`
        : '这段末尾没有可核验的连续两球接球计划。');
    }
    if (/如果|假如|要是/.test(q)) answers.push('请选择具体一帧，再说明假设你向左、向右或停留；时间段不能当作单一的行动前快照。');
    if (/大球.*几|计分|扣几|漏.*算/.test(q)) answers.push('小球漏接计1次，合作大球漏接计3次。');
    return answers.length ? answers.join(' ') : `这段回放包含${goals.length}段接球分工、${caught.length}次接住和${missed.length}次漏接。请指出你想问的球或目标变化。`;
  }
  globalThis.PongExplanations = { bubble, answer, answerRange };
})();
