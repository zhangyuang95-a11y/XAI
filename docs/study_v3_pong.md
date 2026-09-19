# Turn-based cooperative Pong v3.0

The new engine is `domains/pong/turnbased.py`. The older continuous Pong files remain unchanged. Shared study code controls participant assignment, consent, demonstrations, Task 1 → Task 2 → Task 3, question permissions, and storage. The engine never accepts a group or participant argument.

## Public mechanics and scenarios

Nine lanes store positions 0–8 and display lane numbers 1–9. Human and AI each submit one left/right/wait action. Both move simultaneously, then all visible countdowns decrease by one, then arrivals score exactly once. Paddles may overlap without a penalty. An ordinary ball needs one covering paddle and earns 1 point; a cooperative ball needs distinct paddles on its two contact lanes and earns 3 points. Misses earn zero. All current balls, contacts and countdowns are public. No bounce or real-time reaction is required.

Tasks contain 6/8/8 waves and last 30/41/41 simulated turns. Each wave contains one cooperative ball and at most one ordinary ball. Waves last 4–6 turns. Position carries over at wave boundaries. Seeds pre-sample the entire schedule independently of actions, scores, group or questions. Task 1 includes two ordinary-ball distractions; Tasks 2 and 3 each include six, with both safe and costly detours. Task 3 mirrors contact combinations and reverses initial player positions while retaining exactly the same AI rules.

Task score is `100 × raw points / all scheduled points`. Total scheduled point value is public, while future contacts and arrival timings are not. Some ordinary balls are incompatible with cooperation; a maximum score below 100 is legitimate and is quantified by an offline exact search. Public score metrics omit internal commitment-change counters; the researcher score retains them.

## Fixed AI and evidence

The AI first enumerates both assignments of the visible cooperative ball. An assignment is feasible only if the AI can reach its side and the human can reach the opposite side within the remaining turns. The AI retains its existing assignment while feasible. A new assignment minimizes total required travel, with the AI taking the right-hand side on a tie. An infeasible prior assignment may change to the other feasible assignment; otherwise the AI may take a reachable ordinary ball. It never reads the human's unsubmitted action.

An ordinary-ball detour is permitted only if the AI can reach that ball and still reach its assigned cooperative contact by its arrival. A small human movement does not by itself reverse sides. With no verified reachable job, the AI waits. There is no neural or random fallback. Assignment changes and releases are recorded separately from public catch events. `decide` returns a proposed memory update; only `step` updates real policy memory.

English/Chinese facts are generated from the same verified decision, exact lane distances/countdowns, comparison records, public mechanics and recorded events. Statements that require future human movement are conditional. `facts` rejects mismatched decision evidence. Language selection and question interpretation belong to the shared semantic question service; this domain engine does not claim to implement general question understanding.

## Information and simulation boundaries

`_observation` whitelists human position, AI position, visible balls, terminal status and AI memory. `decide`, `facts` and the human advisory proxy cannot read the seed or hidden schedule. Changing unannounced waves produces identical decisions, visible evidence and advisory actions.

`human_advisor` performs an exact finite-state search over the current visible wave. It controls only the human, and every simulated AI action comes from the actual fixed AI. It has no future-wave access and never appears in the participant API/UI. `wave_feasibility` returns its current-wave optimum and replayable human actions.

`global_reachable_upper_bound` is explicitly an offline researcher verifier. It knows the fixed full schedule to compute an exact whole-task upper bound, retains the fixed AI at every branch, and merges equivalent states by best earned score. It is not used by the AI, participant advisor, demo, facts or participant answers. Participant forward counterfactuals must stop at a wave boundary before exposing new balls. Shared permissions must also prevent explanations outside active A/Task 2.

## Verification on 2026-09-20

`python3 -m pytest tests/test_study_v3_pong.py -q`: 28 tests passed. The suite covers simultaneous catch timing; ordinary overlap counted once; two paddles on the same cooperative side failing; bounds and stale decision rejection; stable and necessary changed assignments; safe and costly detours; waiting without a fallback; non-mutating evidence/search; hidden-future isolation; deterministic replay; finite termination; exact search checked against independently enumerated full-engine paths; real demonstration outcomes; and independently instantiated comprehension scenarios.

The frozen configuration has 24 development seeds (730100–730123) and 24 disjoint reserved seeds (731100–731123). All 48 seeds were run through all three tasks. The planning human proxy caught every cooperative ball in every task. Scores below 100 came from actual competing ordinary opportunities. The held-out set is reserved from scenario tuning, not a substitute for an independent human study.

Mean task scores from deterministic synthetic proxies:

| Seed set | Task | Current-wave planner | Nearest-arrival greedy | Public-history learner | Random human | Exact global upper bound |
|---|---:|---:|---:|---:|---:|---:|
| Development | 1 | 97.50 | 75.63 | 97.50 | 22.08 | 97.50 |
| Development | 2 | 95.56 | 69.58 | 92.22 | 22.50 | 95.56 |
| Development | 3 | 96.81 | 70.56 | 86.81 | 27.92 | 96.81 |
| Reserved | 1 | 98.13 | 85.00 | 98.13 | 24.38 | 98.13 |
| Reserved | 2 | 96.25 | 76.53 | 94.58 | 25.69 | 96.25 |
| Reserved | 3 | 97.64 | 67.92 | 88.75 | 25.83 | 97.64 |

These are human-role proxy scores with the identical fixed AI. The history learner estimates AI side preference from publicly observed past contacts and pursues safe detours; it never sees policy memory. Random movement uses a separate `Random(seed + 991)` generator. Each table cell averages 24 complete task runs. The exact full-schedule upper bounds were evaluated for all 144 task instances. The current-wave planner happened to attain those upper bounds on every instance.

The history learner approaches the planner in Task 2, demonstrating that the AI habits are learnable from observation. These results establish technical feasibility and some coordination space; they neither establish nor guarantee a 50% human A/B improvement. No human outcomes were collected or invented. No scenario tuning was performed after viewing the reserved results.

## Real demonstration and comprehension

The demonstration is an actual 12-turn, three-wave execution with all 13 public snapshots. Six bilingual captions reference exact snapshots: start, first simultaneous move, ordinary catch (+1), cooperative catch (+3), an uncovered contact and miss (+0), and recovered cooperation (+3). It uses the same controller and physical transition function as gameplay; captions explain only public mechanics.

Three bilingual comprehension questions use separate fixed scenarios: next movement toward a reachable cooperative side, waiting while already covering a contact, and choosing an ordinary ball when cooperative coverage is impossible. Their answer indices are checked against real decisions and must be stripped by the shared participant-facing questionnaire serializer.
