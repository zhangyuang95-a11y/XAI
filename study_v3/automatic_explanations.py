"""Deterministic explanation triggers, grounded in the executed game controller.

These records are separate from voluntary questions. No provider request, hidden
future schedule, or group-dependent controller action is introduced.
"""

import json

GUIDED_VERSION = 'first-node-question-guide-v4'
GUIDED_VERSIONS = frozenset({'guided-question-choice-v3', GUIDED_VERSION})
VERSION = GUIDED_VERSION
CONFIRM_VERSIONS = frozenset({'event-nodes-confirm-v2', *GUIDED_VERSIONS})
SUPPORTED_VERSIONS = frozenset({'event-nodes-v1', *CONFIRM_VERSIONS})
QUESTION = {'en': 'Why are you making this decision?', 'zh': '你为什么做这个决定？'}


def detouring(decision):
    selected = decision.get('controller_trace', {}).get('selected_ai_action', {})
    return (decision.get('action') not in (None, 'wait')
            and selected.get('distance_after', -1) >= selected.get('distance_before', 0))


def charge_phase(decision):
    if decision.get('reason_code') == 'charging':
        return 'charging'
    if decision.get('goal_details', {}).get('label') == 'charger':
        return 'travel'
    return None


def candidate(domain, state, decision, previous=None):
    """One card for the displayed state; reasons describe its NEXT decision.

    A collision refers explicitly to the incoming transition instead. Previous
    is the stored decision that actually preceded this state, never recomputed.
    """
    if state['terminal']:
        return None
    turn = state['turn']
    previous = previous or {}
    triggers, keys, facts = [], [], []
    if domain == 'warehouse':
        collision = next((e for e in state.get('events', []) if e['type'] == 'collision'), None)
        if collision:
            triggers.append('collision')
            actions = state.get('last_actions') or {}
            facts.append({
                'en': f"The completed step {turn - 1} → {turn}: you chose {actions.get('human', 'unknown')}; I chose {actions.get('ai', 'unknown')}. {collision['en']}",
                'zh': f"刚完成的第 {turn - 1} → {turn} 步：你的指令为 {actions.get('human', 'unknown')}，我的指令为 {actions.get('ai', 'unknown')}。{collision['zh']}",
            })
        if detouring(decision) and (not detouring(previous) or decision.get('goal') != previous.get('goal')):
            triggers.append('detour')
        phase = charge_phase(decision)
        if phase and phase != charge_phase(previous):
            triggers.append('charge_' + phase)
    elif domain == 'pong':
        target = decision.get('explanation_target') or {}
        # Explain the current team catch, including both contact assignments.
        # A continuing commitment does not emit another card each move.
        if target.get('team_ids'):
            triggers.append('team_ball')
            keys = [f"team:{ball}:{target['lane']}:{target['human_lane']}" for ball in sorted(target['team_ids'])]
    elif domain == 'kitchen' and turn > 0 and turn % 5 == 0:
        triggers.append('five_steps')
    if not triggers:
        return None
    facts.append({'en': decision['reason_en'], 'zh': decision['reason_zh']})
    return {'trigger_key': '|'.join(keys) if keys else f'turn:{turn}',
            'trigger_types': triggers, 'turn': turn,
            'body': {lang: '\n\n'.join(f[lang] for f in facts) for lang in ('en', 'zh')},
            'reason_code': decision.get('reason_code'), 'version': VERSION}


def public_card(row, language):
    content = json.loads(row['content_json'])
    guided = content.get('version') in GUIDED_VERSIONS
    requested = content.get('requested_at') is not None
    return {'id': row['id'], 'turn': row['turn'],
            'body': content['body'][language] if not guided or requested else '',
            'trigger_types': content['trigger_types'],
            'displayed': row['displayed'] is not None,
            'requires_confirmation': content.get('version') in CONFIRM_VERSIONS,
            'confirmed': row['confirmed'] is not None,
            'in_question_panel': content.get('version') == GUIDED_VERSION,
            'guided': guided, 'onboarding': bool(content.get('onboarding')),
            'requested': requested, 'question': QUESTION[language] if guided else '',
            'response': content.get('response')}
