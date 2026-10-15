"""Repeated self-reported AI understanding; never an objective comprehension score."""
VERSION = 'task2-understanding-20pct-v1'
TASKS = (2,)
GROUPS = ('A', 'B')
CHECKPOINTS = (20, 40, 60, 80, 100)
QUESTION = {
    'en': 'How well do you currently understand why your AI teammate acts the way it does?',
    'zh': '你目前有多理解 AI 队友为什么这样行动？',
}
LABELS = {
    'en': ['Not at all', 'Slightly', 'Moderately', 'Well', 'Completely'],
    'zh': ['完全不理解', '不太理解', '一般', '比较理解', '完全理解'],
}

def checkpoint(state):
    # Reaching the end early produces an end rating, not several retrospective ratings.
    if state['terminal']:
        return 100
    turn, maximum = state['turn'], state['max_turns']
    return next((p for p in CHECKPOINTS if turn == (maximum * p + 99) // 100), None)

def public(row, language):
    if row is None:
        return None
    return {**row, 'question': QUESTION[language], 'labels': LABELS[language]}
