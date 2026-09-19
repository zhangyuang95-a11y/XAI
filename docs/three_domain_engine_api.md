# Shared implementation contract

Worktree: `/Users/zhangyuang/Desktop/ICLR/XAI-study-v3`.
Full brief: `/Users/zhangyuang/Desktop/ICLR/analysis/THREE_DOMAIN_HUMAN_STUDY_CODEX_BRIEF_20260919.md`.
All functions are pure and JSON serializable. Never accept group/participant credentials in an engine.

Engine modules: `domains/kitchen/engine.py`, `domains/pong/turnbased.py`, `domains/warehouse/turnbased.py`.

Required exports:

- `VERSION: str`, `DOMAIN: str`
- `initial_state(seed: int, task: int) -> dict`
- `legal_actions(state, actor='human') -> list[str]`; include `wait`; movement uses `up/down/left/right`. Kitchen interactions use descriptive IDs e.g. `take_tomato`, `interact_handoff`, `chop`, `plate`, `serve`, `discard`; expose only currently allowed interactions.
- `decide(state) -> dict`: `action`, `reason_code`, `reason_en`, `reason_zh`, `goal` (str), `memory` (dict), `alternatives` (list of actual comparisons with `action`, `en`, `zh`), optional `facts` list. Must ignore hidden future schedules and human unsubmitted action.
- `step(state, human_action, decision=None) -> dict`: deep copied next state, default decide before human movement. Terminal input rejected. Illegal human input rejected. Update policy memory only during actual step, never decide/facts. New state contains the step's public factual `events`.
- `public_state(state) -> dict`: whitelist no policy memory/reasons/future schedule/seed. Common fields below.
- `score(state) -> dict`: `task_score` 0–100, `raw_score`, `metrics` dict.
- `rules(language='en') -> list[str]`: exact public mechanics, no AI hidden priorities.
- `facts(state, decision=None) -> list[dict]`: each `{id, en, zh}`, verified public state or actual decision facts. Never future hidden schedule. Include reasons/alternatives as supported evidence for language selection.
- `human_advisor(state) -> str`: feasibility proxy controls human only, same fixed AI, no hidden future access. Used in development and neutral tutorial, never in participant UI/API.
- `demonstration() -> dict`: `{frames: [public_state...], captions: [{index: int, en: str, zh: str}]}`. Produce real deterministic execution, 4–6 public-mechanics captions, at least successful cooperation and appropriate public failure/repair example. No AI strategy explanations.
- `comprehension(language='en') -> list[dict]`: 3 items `{id, text, options:[str], answer:int}` from independently verified small scenarios. Server strips answers.

Common internal state keys: `domain`, `version`, `task` (1/2/3), `turn` (0 initially), `max_turns`, `terminal` (bool), `events` (list of `{type,en,zh,...public facts}`), `policy_memory` (dict). Domain may add internal fields.
Common public keys: `domain`, `version`, `task`, `turn`, `max_turns`, `terminal`, `events`, `score` (score result), `human` and `ai` (`x,y` where grid; `x` where Pong), domain visuals. Grid public keys `width,height,walls:[[x,y]],stations:[{id,x,y,label_en,label_zh,kind}]`; kitchen actor `holding`, `pots`, `orders`, `handoff`, `buffers`; Warehouse `orders`, `chargers`, actor `battery`, `carrying`; Pong `lanes`, `wave`, `wave_count`, `balls:[{id,kind,contacts:[int],remaining,initial_remaining}]`.

Each engine owns its domain tests `tests/test_study_v3_{domain}.py`, configuration `configs/study_v3_{domain}.json`, and controller documentation `docs/study_v3_{domain}.md`. Do not edit shared server, frontend, requirements, deployment, registry or other engines. Root integrates these.

Develop and held-out seed lists each contain 24 entries, deterministic and disjoint. Feasibility tests must show actual fixed-AI success, not replacing AI. Scenario values may be calibrated before release with recorded rationale. No claim that simulated proxies establish human A/B effectiveness.
