"""Candidate public-mechanics questionnaire; not a validated understanding scale.

This fixed bank is identical across A/B/C and does not depend on play, answers,
NN outputs or treatment. It measures public rules and control comprehension,
not knowledge of an NN's motives, explanation quality or human efficacy.
Translations share item/option IDs; their equivalence still needs pilot review.

SERVER ONLY: never serve this module or its private answer key as a web asset.
``bank_for_participant`` returns a detached key-free view. The caller must check
authoritative Task 3 completion before delivering the bank AND before calling
``score``. This pure scorer has no session/database access and cannot perform
that authorization. Do not offer correctness feedback while any measured task
is active. Version/hash identify content, not scientific validation/admission.
"""
from copy import deepcopy
from hashlib import sha256
import json


VERSION = 'study-understanding-mechanics-20261029.v1'
INSTRUMENT_ID = VERSION
STATUS = 'candidate_instrument_requires_pilot_validation'
CONSTRUCT = 'public_mechanics_and_control_comprehension'
_DOMAINS = ('warehouse', 'pong', 'kitchen')
_LANGUAGES = ('en', 'zh')


class UnderstandingError(ValueError):
    pass


def _item(identifier, en, zh, options, correct):
    return dict(id=identifier, prompt=dict(en=en, zh=zh),
                options=[dict(id=key, text=dict(en=e, zh=z)) for key, e, z in options],
                correct=correct)


def _control_items(domain):
    return [
        _item(domain + '_frozen_teammate',
              'During a human–AI task, what determines the AI teammate’s action?',
              '在真人与 AI 合作任务中，队友 AI 的动作由什么决定？', [
                  ('a', 'A new policy is trained from each human key press.', '每次真人按键后重新训练的新策略。'),
                  ('b', 'A task-allocation rule replaces the network’s chosen action.', '任务分配规则替换网络选出的动作。'),
                  ('c', 'The bound frozen NN, with the environment’s physical legal-action mask.',
                   '绑定的冻结神经网络，以及环境的物理合法动作掩码。'),
              ], 'c'),
        _item(domain + '_past_actions',
              'A record displays an AI action from a completed turn. What does it tell you?',
              '记录中显示了一个已经完成回合的 AI 动作。这能说明什么？', [
                  ('a', 'It records the executed past action; it does not reveal the next unexecuted action.',
                   '它记录已经执行的过去动作，并不提前暴露下一个尚未执行的动作。'),
                  ('b', 'It guarantees the AI will repeat that action next turn.', '它保证 AI 下一回合会重复这个动作。'),
                  ('c', 'It is a command that the human can use to replace the AI’s next action.',
                   '它是可由真人用来替换 AI 下一动作的指令。'),
              ], 'a'),
    ]


_BANK = {
    'warehouse': [
        _item('warehouse_charging',
              'Where can a robot replenish its battery by waiting?',
              '机器人在哪里等待可以补充电量？', [
                  ('a', 'At any empty aisle cell.', '任何空走廊格。'),
                  ('b', 'At the marked charging cell.', '标出的充电格。'),
                  ('c', 'At any pickup cell.', '任何取货格。'),
              ], 'b'),
        _item('warehouse_endpoints',
              'How does a robot collect and deliver an assigned order?',
              '机器人如何领取并交付对应订单？', [
                  ('a', 'Press E anywhere on the map.', '在地图任意位置按 E。'),
                  ('b', 'Wait at its starting position.', '在起点等待。'),
                  ('c', 'Enter the marked pickup cell, then enter that order’s delivery cell while carrying it.',
                   '进入标出的取货格取货，再携货进入该订单对应的送货格。'),
              ], 'c'),
    ] + _control_items('warehouse'),
    'pong': [
        _item('pong_ordinary_ball',
              'If both paddles cover an ordinary ball’s contact lane when it arrives, how is that ball counted?',
              '普通球到达接球线时，如果两块挡板都覆盖它的接触列，这个球如何计分？', [
                  ('a', 'Once; overlapping paddles do not double its points.', '计一次；挡板重叠不会让这个球得双倍分。'),
                  ('b', 'Twice, once for each paddle.', '计两次，每块挡板各计一次。'),
                  ('c', 'It is automatically missed.', '自动判为漏接。'),
              ], 'a'),
        _item('pong_cooperative_ball',
              'What is required to catch a cooperative ball with two different contact lanes?',
              '接住有两个不同接触列的合作球，需要满足什么条件？', [
                  ('a', 'One paddle visits both lanes at any earlier time.', '一块挡板曾在更早时间到过这两列即可。'),
                  ('b', 'The two paddles cover its two distinct contact lanes at the arrival step.',
                   '球到达时，两块挡板同时覆盖它的两个不同接触列。'),
                  ('c', 'Both paddles overlap on only one of the two lanes.', '两块挡板只需重叠在其中一列。'),
              ], 'b'),
    ] + _control_items('pong'),
    'kitchen': [
        _item('kitchen_interaction',
              'When using E to interact with a kitchen station, which location matters?',
              '按 E 与厨房工作台交互时，哪个位置决定交互目标？', [
                  ('a', 'The nearest station anywhere on the map.', '地图上距离最近的任意工作台。'),
                  ('b', 'The station directly in front of the character, when that interaction is available.',
                   '角色正面相邻的工作台，并且该交互当前可用。'),
                  ('c', 'The station behind the character.', '角色背后的工作台。'),
              ], 'b'),
        _item('kitchen_wait',
              'What does Space (wait) do during the kitchen task?',
              '厨房任务中，空格键（等待）的作用是什么？', [
                  ('a', 'It submits all remaining orders automatically.', '自动交付所有剩余订单。'),
                  ('b', 'It freezes both characters and the environment until another key is pressed.',
                   '冻结双方角色和环境，直到按下其他键。'),
                  ('c', 'It submits a waiting action and advances one joint turn.', '提交等待动作，并推进一个联合回合。'),
              ], 'c'),
    ] + _control_items('kitchen'),
}


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


# Includes private keys so changing scoring requires a new content identity.
INSTRUMENT_SHA256 = sha256(_canonical(dict(version=VERSION, construct=CONSTRUCT,
                                         status=STATUS, bank=_BANK)).encode('utf-8')).hexdigest()


def instrument_sha256():
    return INSTRUMENT_SHA256


def _domain(domain):
    if type(domain) is not str or domain not in _DOMAINS:
        raise UnderstandingError('unregistered questionnaire domain')
    return _BANK[domain]


def bank_for_participant(domain, language):
    """Return key-free content only after the caller authorizes Task 3 completion."""
    items = _domain(domain)
    if type(language) is not str or language not in _LANGUAGES:
        raise UnderstandingError('unregistered questionnaire language')
    return deepcopy(dict(
        version=VERSION, sha256=INSTRUMENT_SHA256, domain=domain, language=language,
        status=STATUS, construct=CONSTRUCT, administration='after_task3_only',
        questions=[dict(id=item['id'], prompt=item['prompt'][language],
                        options=[dict(id=option['id'], text=option['text'][language])
                                 for option in item['options']]) for item in items],
    ))


def score(domain, answers):
    """Server-only aggregate scoring after an authoritative Task 3 completion check.

    ``answers`` is exactly {item_id: option_id}; missing, extra and malformed
    responses are rejected rather than silently scored as incorrect. There is
    no pass threshold and no feedback identifying an item's correct option.
    """
    items = _domain(domain)
    if type(answers) is not dict or set(answers) != {item['id'] for item in items}:
        raise UnderstandingError('answers must contain exactly this domain’s item IDs')
    for item in items:
        answer = answers[item['id']]
        if type(answer) is not str or answer not in {option['id'] for option in item['options']}:
            raise UnderstandingError('answer is not a registered option ID')
    correct = sum(answers[item['id']] == item['correct'] for item in items)
    return dict(version=VERSION, sha256=INSTRUMENT_SHA256, domain=domain,
                status=STATUS, construct=CONSTRUCT, administration='after_task3_only',
                item_count=len(items), correct_count=correct,
                proportion_correct=correct / len(items), validated_scale=False,
                efficacy_evidence=False)
