from copy import deepcopy

from backend.warehouse_r43_runtime import R43WarehouseEnv
from env.warehouse.domain import collaborative_study_config
from env.warehouse_native.r43_charger import R43_OBSERVATION_FEATURE_NAMES


def _env(occupant="robot_2", *, occupant_battery=60, teammate_battery=20,
         teammate_position=(5, 2)):
    env = R43WarehouseEnv(collaborative_study_config(move_battery_cost=3.0))
    env.reset(seed=43)
    state = env.get_state()
    other = "robot_1" if occupant == "robot_2" else "robot_2"
    state.by_id(occupant).position = env.layout.charger_position
    state.by_id(occupant).battery = float(occupant_battery)
    state.by_id(occupant).active = True
    state.by_id(other).position = teammate_position
    state.by_id(other).battery = float(teammate_battery)
    state.by_id(other).active = True
    env.set_state(state)
    return env, other


def _wait(env, occupant, other):
    actions = {occupant: "WAIT", other: "WAIT"}
    return env.step(actions)[-1]


def _penalties(info):
    return [event for event in info["events"]
            if event["event"] == "charger_occupancy_penalty"]


def test_three_percent_movement_and_ten_percent_charge():
    env, other = _env(occupant_battery=60)
    before = env.state.by_id("robot_2").battery
    env.step({"robot_1": "WAIT", "robot_2": "UP"})
    assert env.state.by_id("robot_2").battery == before - 3
    state = env.get_state()
    state.by_id("robot_2").position = env.layout.charger_position
    state.by_id("robot_2").battery = 42
    env.set_state(state)
    _wait(env, "robot_2", other)
    assert env.state.by_id("robot_2").battery == 52


def test_penalty_threshold_grace_once_and_reoffence_after_departure():
    env, other = _env()
    assert not _penalties(_wait(env, "robot_2", other))
    assert not _penalties(_wait(env, "robot_2", other))
    third = _wait(env, "robot_2", other)
    assert _penalties(third)[0]["amount"] == -50
    score = env.state.user_score
    assert not _penalties(_wait(env, "robot_2", other))
    assert env.state.user_score == score - 1
    env.step({"robot_1": "WAIT", "robot_2": "UP"})
    env.step({"robot_1": "WAIT", "robot_2": "DOWN"})
    for _ in range(2):
        assert not _penalties(_wait(env, "robot_2", other))
    assert _penalties(_wait(env, "robot_2", other))


def test_rule_boundaries_and_collision_exemption():
    low, other = _env(occupant_battery=59)
    assert not _penalties(_wait(low, "robot_2", other))
    assert low._r43_streaks["robot_2"] == 0
    high_other, other = _env(teammate_battery=21)
    assert all(not _penalties(_wait(high_other, "robot_2", other)) for _ in range(4))
    exact, other = _env(teammate_position=(4, 2))
    assert _penalties([_wait(exact, "robot_2", other) for _ in range(3)][-1])
    blocked, _ = _env()
    for _ in range(2):
        _wait(blocked, "robot_2", "robot_1")
    collision = blocked.step({"robot_1": "RIGHT", "robot_2": "WAIT"})[-1]
    assert collision["robot_collision"] is True
    assert not _penalties(collision)


def test_snapshot_preserves_counter_and_event_id_is_deterministic():
    env, other = _env()
    _wait(env, "robot_2", other)
    snapshot = deepcopy(env.snapshot())
    left = R43WarehouseEnv(collaborative_study_config(move_battery_cost=3.0))
    right = R43WarehouseEnv(collaborative_study_config(move_battery_cost=3.0))
    left.restore(deepcopy(snapshot)); right.restore(deepcopy(snapshot))
    _wait(left, "robot_2", other); a = _wait(left, "robot_2", other)
    _wait(right, "robot_2", other); b = _wait(right, "robot_2", other)
    assert _penalties(a) == _penalties(b)
    assert left.snapshot() == right.snapshot()


def test_penalty_is_symmetric_and_has_no_total_cap():
    env, other = _env(occupant="robot_1")
    for _ in range(2):
        assert not _penalties(_wait(env, "robot_1", other))
    first = _penalties(_wait(env, "robot_1", other))[0]
    assert first["agent_id"] == "robot_1"
    first_score = env.state.user_score
    env.step({"robot_1": "UP", "robot_2": "WAIT"})
    env.step({"robot_1": "DOWN", "robot_2": "WAIT"})
    for _ in range(2):
        assert not _penalties(_wait(env, "robot_1", other))
    assert _penalties(_wait(env, "robot_1", other))[0]["amount"] == -50
    # The second independent occupation loses another 50 points; there is no
    # per-round total cap.
    assert env.state.user_score == first_score - 2 - 3 - 50


def test_no_safe_departure_is_exempt():
    env, other = _env()
    # Exercise the authoritative no-exit branch independently of the fixed
    # study map, whose charger normally has two exits.
    env._safe_departures = lambda _state, _agent_id: ()
    for _ in range(4):
        assert not _penalties(_wait(env, "robot_2", other))


def test_rule_counter_and_flags_are_in_the_actor_observation():
    env, other = _env()
    assert env.observation_size == 207
    assert tuple(env.feature_names[-10:]) == R43_OBSERVATION_FEATURE_NAMES
    start = len(env.feature_names) - len(R43_OBSERVATION_FEATURE_NAMES)
    first = env.observations()["robot_2"][start:]
    assert first.tolist()[:8] == [0.0, 0.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0]
    _wait(env, "robot_2", other)
    assert env.observations()["robot_2"][start] == .5
    _wait(env, "robot_2", other)
    assert env.observations()["robot_2"][start] == 1.0
    _wait(env, "robot_2", other)
    assert env.observations()["robot_2"][start + 1] == 1.0
