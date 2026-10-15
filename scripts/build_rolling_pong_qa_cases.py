#!/usr/bin/env python3
"""Build 80 bilingual v7 Pong evidence-composition fixtures.

These injected plans check binding and factual rendering, not language-model
understanding or an explanation effect on participants.
"""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from domains.pong import turnbased as pong
ROOT = Path(__file__).resolve().parents[1]


def build():
    state = pong.initial_state(730100, 2)
    snapshots = []
    for turn in range(6):
        if turn in (0, 1, 3, 5):
            snapshots.append(deepcopy(state))
        state = pong.step(state, pong.human_advisor(state))
    cases = []
    for snapshot in snapshots:
        turn = snapshot['turn']
        small = next(ball for ball in snapshot['balls'] if ball['kind'] == 'ordinary')
        team = next(ball for ball in snapshot['balls'] if ball['kind'] == 'cooperative')
        specs = [
            ('geometry', 'How large is the court at turn {turn}?', '第{turn}回合时球场有多大？', ['public_rule:0'], 'shared', 'rule', None, None),
            ('small', 'Where will {small} arrive after turn {turn}?', '第{turn}回合，{small}会到达哪一道？', ['ball:' + small['id']], 'shared', 'observation', None, None),
            ('team', 'What contact lanes does {team} require at turn {turn}?', '第{turn}回合，{team}需要覆盖哪两道？', ['ball:' + team['id']], 'shared', 'observation', None, None),
            ('ai-action', 'At turn {turn}, what will you do next?', '第{turn}回合你下一步会做什么？', ['system:ai_action'], 'ai', 'action', None, None),
            ('ai-reason', 'At turn {turn}, why this next action?', '第{turn}回合为什么选择下一步动作？', ['system:ai_reason'], 'ai', 'reason', None, None),
            ('human-advice', 'At turn {turn}, what move might I choose?', '第{turn}回合我可以选择什么动作？', ['system:human_advice'], 'human', 'advice', None, None),
            ('nearest-human', 'At turn {turn}, what visible ball is nearest to me?', '第{turn}回合哪个可见球离我最近？', ['human_nearest_ball'], 'human', 'observation', None, None),
            ('ambiguous', 'At turn {turn}, why did you ignore that one?', '第{turn}回合你为什么忽略那一个？', [], 'shared', 'observation', None, None),
            ('team-requirement', 'At turn {turn}, did you assign us sides for {team}?', '第{turn}回合你为{team}给我们分配两侧了吗？', ['ball_plan:' + team['id']], 'shared', 'assignment', team['id'], None),
            ('wait-branch', 'If I wait at turn {turn}, what happens next?', '如果我在第{turn}回合等待，下一步会怎样？', [], 'human', 'comparison', None, ['wait']),
        ]
        for name, english, chinese, ids, subject, purpose, object_id, actions in specs:
            substitutions = {'turn': turn, 'small': small['id'], 'team': team['id']}
            for language, question in (('en', english), ('zh', chinese)):
                kind = 'clarification' if name == 'ambiguous' else 'counterfactual' if actions else 'facts'
                intent = {'kind': kind, 'subject': subject, 'purpose': purpose,
                          'evidence_ids': list(ids)}
                if object_id:
                    intent['object_id'] = object_id
                if actions:
                    intent.update(actions=list(actions), horizon=1)
                cases.append({'case_id': f'pong-neural-{turn}-{name}-{language}',
                              'domain': 'pong', 'question': question.format(**substitutions),
                              'language': language, 'state': deepcopy(snapshot),
                              'expected_kind': kind, 'expected_fact_ids': list(ids),
                              'expected_claims': [{'path': 'turn', 'equals': turn}],
                              'expected_plan': {'language': language,
                                                'binding': {'task': snapshot['task'], 'turn': turn},
                                                'premise': 'unclear' if kind == 'clarification' else 'supported',
                                                'clarification': 'ambiguous_object' if kind == 'clarification' else None,
                                                'intents': [] if kind == 'clarification' else [intent]}})
    assert len(cases) == 80 and len({case['question'] for case in cases}) == 80
    return cases


if __name__ == '__main__':
    path = ROOT / 'configs/study_v3_qa_cases.json'
    existing = json.loads(path.read_text())
    existing['cases'] = [case for case in existing['cases'] if case.get('domain', case['state']['domain']) != 'pong'] + build()
    existing['version'] = 'three-domain-qa-cases.v4.0'
    existing['evaluation_status'] = 'Injected composition fixtures only; they do not establish real-provider question understanding.'
    path.write_text(json.dumps(existing, ensure_ascii=False, indent=2) + '\n')
    print('Rebuilt 80 frozen-actor Pong composition fixtures.')
