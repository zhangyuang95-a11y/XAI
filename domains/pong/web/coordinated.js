/* Decision-only counterpart of policies/coordinated.py. Physics remains in app.js. */
(() => {
  const VERSION = 'pong-coordinated.v2.3';
  const EPSILON = 1e-6;

  class PongCoordinator {
    constructor({ horizonSeconds = 10, maxOpportunities = 3, switchGain = 1.5 } = {}) {
      this.horizonSeconds = horizonSeconds;
      this.maxOpportunities = maxOpportunities;
      this.switchGain = switchGain;
      this.smallCommitment = null;
    }

    plan(game, candidates, position, first = null) {
      const visit = (index, at, x, chosen) => {
        if (index === candidates.length) return { score: chosen.reduce((sum, item) => sum + item.penalty, 0), chosen };
        const candidate = candidates[index];
        const skipped = index === 0 && first !== null ? { score: -1, chosen: [] }
          : visit(index + 1, at, x, chosen);
        const arrival = at + Math.abs(candidate.targetX - x) / game.constructorSpec.paddleSpeed;
        const contact = candidate.prediction.time_until_contact;
        if (arrival > contact - .02 + EPSILON) return skipped;
        const taken = visit(index + 1, contact, candidate.targetX, [...chosen, candidate]);
        return taken.score > skipped.score ? taken : skipped;
      };
      return visit(0, 0, position, []);
    }

    choose(game, proposal) {
      const own = game.aiX;
      const other = game.playerX;
      let oldLarge = game.commitment ? { ...game.commitment } : null;
      const base = game.chooseRuleAI();
      const all = game.balls.map(ball => game.candidateFor(ball)).filter(Boolean);
      let available = all.filter(item => item.viable && !item.handoff
        && item.prediction.time_until_contact <= this.horizonSeconds)
        .sort((a, b) => a.prediction.time_until_contact - b.prediction.time_until_contact
          || b.penalty - a.penalty || a.prediction.ball_id.localeCompare(b.prediction.ball_id))
        .slice(0, this.maxOpportunities);
      let selected = all.find(item => item.prediction.opportunity_id === base.opportunityId
        && item.contactSide === base.contactSide) || null;
      if (selected?.requiresPartner && !selected.viable) {
        game.commitment = null;
        oldLarge = null;
        selected = null;
      }
      const priorSmall = available.find(item => item.prediction.opportunity_id === this.smallCommitment);
      if (priorSmall) selected = priorSmall;
      else if (this.smallCommitment) this.smallCommitment = null;

      const best = this.plan(game, available, own);
      let baseScore = 0;
      let baseSequence = [];
      if (selected) {
        const baseline = [selected, ...available.filter(item => item !== selected
          && item.prediction.time_until_contact > selected.prediction.time_until_contact)];
        const baselinePlan = this.plan(game, baseline, own, selected.prediction.opportunity_id);
        baseScore = baselinePlan.score;
        baseSequence = baselinePlan.chosen;
      }
      const planned = best.chosen[0] || null;
      let switched = false;
      if (planned && selected && !oldLarge && planned.prediction.opportunity_id !== selected.prediction.opportunity_id
          && best.score >= baseScore + this.switchGain) {
        selected = planned;
        switched = true;
      } else if (!selected && planned && !oldLarge) selected = planned;
      const actualSequence = switched || (selected === planned && !baseSequence.length) ? best.chosen : baseSequence;

      let action = base.action;
      let reason = 'no_feasible_assignment';
      let holding = false;
      let partnerStatus = 'unknown';
      if (selected) {
        if (selected.requiresPartner) {
          game.commitment = { ball_id: selected.prediction.ball_id,
            opportunity_id: selected.prediction.opportunity_id, contactSide: selected.contactSide };
          this.smallCommitment = null;
        } else {
          game.commitment = null;
          this.smallCommitment = selected.prediction.opportunity_id;
        }
        const contact = selected.prediction.contact_cells;
        const targetCell = selected.contactSide === 'right' ? contact.at(-1) : contact[0];
        const difference = selected.targetX - own;
        const coveredNow = game.paddleCoversAt(own, targetCell);
        action = !coveredNow
          ? (difference > 0 ? 'right' : difference < 0 ? 'left' : 'stay')
          : (difference > .025 ? 'right' : difference < -.025 ? 'left' : 'stay');
        const partnerCell = selected.requiresPartner
          ? (selected.contactSide === 'left' ? contact.at(-1) : contact[0]) : targetCell;
        holding = coveredNow && action === 'stay';
        partnerStatus = game.paddleCoversAt(other, partnerCell)
          ? 'covered' : selected.viable ? 'reachable' : 'unreachable';
        reason = holding && selected.requiresPartner ? 'keep_large_side'
          : holding ? 'keep_small_coverage' : switched ? 'sequence_gain'
            : selected.requiresPartner ? 'prepare_large_side' : 'approach_small';
      }
      const maximum = game.constructorSpec.width - game.constructorSpec.paddleWidth;
      const boundary = (own <= EPSILON && proposal === 'left')
        || (own >= maximum - EPSILON && proposal === 'right');
      if (!selected && boundary) { action = 'stay'; reason = 'boundary_no_motion'; }
      if (!selected && !boundary) action = proposal;

      const evidence = {
        version: VERSION, agent: 'ai', decision_frame: game.frame,
        target_ball_id: selected?.prediction.ball_id || null,
        opportunity_id: selected?.prediction.opportunity_id || null,
        target_kind: selected?.prediction.kind || null,
        contact_side: selected?.contactSide || null,
        target_x: selected?.targetX ?? null,
        self_distance: selected?.aiDistance ?? null,
        partner_distance: selected?.playerDistance ?? null,
        self_reachable: selected?.aiViable || false,
        partner_status: partnerStatus,
        requires_partner: selected?.requiresPartner || false,
        holding, reason,
        candidate_scores: { planned: best.score, baseline: baseScore },
        alternatives: all.map(item => ({ ball_id: item.prediction.ball_id,
          opportunity_id: item.prediction.opportunity_id, kind: item.prediction.kind,
          target_x: item.targetX, contact_side: item.contactSide,
          self_distance: item.aiDistance, partner_distance: item.playerDistance,
          time_until_contact: item.prediction.time_until_contact,
          viable: item.viable, weight: item.penalty,
          in_planning_window: available.includes(item),
          plan_score_if_first: available.includes(item) ? this.plan(game,
            [item, ...available.filter(later => later !== item && later.prediction.time_until_contact > item.prediction.time_until_contact)],
            own, item.prediction.opportunity_id).score : null })),
        planned_sequence: actualSequence.map(item => item.prediction.ball_id),
        discarded_neural_action: action !== proposal ? proposal : null,
        other_current_position: other, excluded_future_player_action: true,
      };
      return { action, reason: action !== proposal ? reason : null, evidence, ruleVersion: VERSION,
        base, selected };
    }
  }
  globalThis.PongCoordinator = PongCoordinator;
})();
