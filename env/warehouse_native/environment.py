"""Native warehouse physics and public-only reward, isolated from legacy control.

The original collision resolver, task sampler and terminal/score finalizer are
reused unchanged. No coordination, assignment, route-goal or old credit-assignment
routine runs here. Legacy dataclasses are storage containers, not control inputs.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from itertools import permutations
import json
import math
import numpy as np

from env.warehouse.domain import AgentState, DeliveryTask, WarehouseState, collaborative_study_config
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.navigation import ACTIONS, MOVE_DELTAS, shortest_path_distance
from env.warehouse.transition_outcome import finalize_transition_outcome
from .observations import observation_names, observation_size, public_observations, task_order

ENVIRONMENT_VERSION = "warehouse-native-physics-v1"
REWARD_VERSION = "warehouse-native-public-pbrs-v1"
GAMMA = .99


def physical_state(state):
    """Public dynamical content, excluding audit goals, rewards and RNG labels."""
    return {
        "frame": state.frame,
        "agents": [{key: deepcopy(getattr(a,key)) for key in ("agent_id","position","battery","active","carrying_task_id","heading","last_action","last_executed_action","last_battery_delta","steps_since_charging","charger_wait_streak","deliveries_completed")} for a in state.agents],
        "tasks": [{key: deepcopy(getattr(t,key)) for key in ("task_id","pickup_position","delivery_position","status","carrier_agent_id","created_frame","claimed_frame","delivered_frame")} for t in state.tasks],
        "completed_tasks": [{key: deepcopy(getattr(t,key)) for key in ("task_id","pickup_position","delivery_position","status","carrier_agent_id","created_frame","claimed_frame","delivered_frame")} for t in state.completed_tasks],
        **{key: getattr(state,key) for key in ("next_task_index","total_deliveries","robot_collision_events","invalid_move_count","shutdown_count","terminated","truncated","terminal_reason")},
    }


class NativeWarehouseEnv(WarehouseMultiAgentEnv):
    environment_name = ENVIRONMENT_VERSION
    actions = ACTIONS

    def __init__(self, config=None):
        super().__init__(config or collaborative_study_config())

    @property
    def observation_size(self):
        return observation_size(self.config)

    @property
    def feature_names(self):
        return observation_names(self.config)

    @property
    def done(self):
        return self.state is not None and (self.state.terminated or self.state.truncated)

    @property
    def native_score(self):
        return 0. if self.state is None else self.state.user_score

    @property
    def participant_score(self):
        return self.native_score

    def reset(self, *, seed=None):
        if seed is not None:
            self._rng.seed(seed)
        self._episode_counter += 1
        excluded = {self.layout.charger_position, *self.layout.robot_start_positions, *self.layout.task_endpoint_exclusions}
        tasks = []
        for index in range(1,self.config.active_task_count+1):
            task = self._sample_delivery_job(task_index=index,created_frame=0,excluded_positions=excluded)
            tasks.append(task)
            excluded.update((task.pickup_position,task.delivery_position))
        agents = [AgentState(agent_id=agent_id,position=self.layout.robot_start_positions[index],heading=self._rng.choice(tuple(MOVE_DELTAS))) for index,agent_id in enumerate(self.agent_ids)]
        self.state = WarehouseState(episode_id=self._episode_counter,frame=0,agents=agents,tasks=tasks,next_task_index=len(tasks)+1)
        return self.observations(), self._info()

    def observations(self):
        self._require_state()
        return public_observations(self.state,self.config)

    def action_masks(self):
        """All five actions remain available, even at a wall or another robot."""
        return {key:(1.,)*len(ACTIONS) for key in self.agent_ids}

    def global_state(self):
        return np.concatenate([self.observations()[key] for key in self.agent_ids]).astype(np.float32)

    def potential(self, state=None):
        """Bounded negative public remaining-work potential, never a teacher label.

        Enumerates both physical task allocations symmetrically. A task may only
        be allocated to its actual carrier or an empty robot. No allocation is
        persisted, exposed to the Actor or used to choose an action. Public energy
        debt accounts for finishing the route and returning to the shared charger.
        The discounted potential telescopes, including zero at either terminal.
        """
        state = state or self.state
        if state.terminated or state.truncated:
            return 0.
        def distance(a,b):
            d = shortest_path_distance(a,b,self.config.map_layout_id)
            return float(d) if math.isfinite(d) else float(self.config.rows*self.config.cols)
        available = [t for t in task_order(state) if t.status=="available"]
        empty = [a for a in state.agents if a.carrying_task_id is None]
        fixed_cost = 0.
        for a in state.agents:
            if a.carrying_task_id:
                task = state.task_by_id(a.carrying_task_id)
                route = distance(a.position,task.delivery_position)
                energy = self.config.move_battery_cost*(route+distance(task.delivery_position,self.layout.charger_position))
                fixed_cost += route + max(0.,energy-a.battery)/self.config.charge_per_wait
        costs = []
        for allocation in permutations(empty,len(available)):
            cost = fixed_cost
            for a,t in zip(allocation,available):
                route = distance(a.position,t.pickup_position)+distance(t.pickup_position,t.delivery_position)
                energy = self.config.move_battery_cost*(route+distance(t.delivery_position,self.layout.charger_position))
                cost += route+max(0.,energy-a.battery)/self.config.charge_per_wait
            costs.append(cost)
        work = min(costs) if costs else fixed_cost
        return -2.*math.tanh(work/30.)

    def step(self, actions, *, decision_metadata=None):
        self._require_state()
        if self.done:
            raise RuntimeError("Cannot step a completed warehouse episode.")
        if any(key not in self.agent_ids for key in actions):
            raise ValueError("Unknown warehouse agent")
        previous = self.state
        raw = {key:str(actions.get(key,"WAIT")) for key in self.agent_ids}
        targets,executed,invalid,collision,collision_kind,intended = self._resolve_motion(previous,raw)
        phi_before = self.potential(previous)
        state = deepcopy(previous)
        state.frame += 1
        state.active_coordination_plan = None
        state.last_coordination_events = ()
        state.last_robot_collision_event = collision
        state.last_robot_collision_kind = collision_kind
        state.robot_collision_events += int(collision)
        state.collision_count += int(collision)
        state.invalid_move_count += len(invalid)
        events = []
        energy_gained = 0.
        for agent in state.agents:
            old = previous.by_id(agent.agent_id)
            agent.position = targets[agent.agent_id]
            agent.last_action = raw[agent.agent_id]
            agent.last_executed_action = executed[agent.agent_id]
            if executed[agent.agent_id] in MOVE_DELTAS:
                agent.battery = max(0.,agent.battery-self.config.move_battery_cost)
                agent.heading = executed[agent.agent_id]
            elif agent.position == self.layout.charger_position:
                agent.battery = min(100.,agent.battery+self.config.charge_per_wait)
            agent.last_battery_delta = agent.battery-old.battery
            if agent.last_battery_delta > 0:
                agent.steps_since_charging = 0
                agent.charger_wait_streak = old.charger_wait_streak+1
                energy_gained += agent.last_battery_delta
                events.append({"event":"charge","agent_id":agent.agent_id,"amount":agent.last_battery_delta})
            else:
                agent.steps_since_charging = min(self.config.horizon,old.steps_since_charging+1)
                agent.charger_wait_streak = 0
        delivered,claimed = [],[]
        for agent in state.agents:
            if agent.carrying_task_id:
                task = state.task_by_id(agent.carrying_task_id)
                if agent.position == task.delivery_position:
                    task.status,task.delivered_frame = "delivered",state.frame
                    delivered.append(task)
                    agent.carrying_task_id = None
                    agent.deliveries_completed += 1
                    state.total_deliveries += 1
                    events.append({"event":"delivery","agent_id":agent.agent_id,"task_id":task.task_id})
            if agent.carrying_task_id is None:
                here = sorted([t for t in state.tasks if t.status=="available" and t.pickup_position==agent.position],key=lambda t:t.task_id)
                if here:
                    task = here[0]
                    task.status,task.carrier_agent_id,task.claimed_frame = "carried",agent.agent_id,state.frame
                    task.claimed_battery = float(agent.battery)
                    agent.carrying_task_id = task.task_id
                    claimed.append(task)
                    events.append({"event":"pickup","agent_id":agent.agent_id,"task_id":task.task_id})
        if delivered:
            ids = {t.task_id for t in delivered}
            state.tasks = [t for t in state.tasks if t.task_id not in ids]
            state.completed_tasks.extend(delivered)
            for _ in delivered:
                excluded = {self.layout.charger_position,*self.layout.robot_start_positions,*self.layout.task_endpoint_exclusions,*(a.position for a in state.agents),*(p for t in state.tasks for p in (t.pickup_position,t.delivery_position))}
                task = self._sample_delivery_job(task_index=state.next_task_index,created_frame=state.frame,excluded_positions=excluded)
                state.next_task_index += 1
                state.tasks.append(task)
                events.append({"event":"task_created","task_id":task.task_id})
        shutdown,components,delta,terminated,truncated,reason = finalize_transition_outcome(self.config,self.layout,state,delivered_count=len(delivered),robot_collision=collision,route_regret=0.)
        state.user_score += delta
        for name,value in components.items():
            state.score_breakdown[name] += value
        state.terminated,state.truncated,state.terminal_reason = terminated,truncated,reason
        for key in shutdown:
            events.append({"event":"shutdown","agent_id":key})
        if collision:
            events.append({"event":"collision","kind":collision_kind})
        shaping = GAMMA*self.potential(state)-phi_before
        reward_parts = {"delivery":10.*len(delivered),"pickup":.25*len(claimed),"time":-.01,"collision":-.25*int(collision),"shutdown":-5.*len(shutdown),"potential":shaping}
        reward = float(sum(reward_parts.values()))
        rewards = {key:reward for key in self.agent_ids}
        state.last_rewards = rewards
        self.state = state
        info = self._info()
        info.update({"requested_actions":raw,"executed_actions":executed,"intended_targets":intended,"invalid_moves":sorted(invalid),"robot_collision":collision,"collision_kind":collision_kind,"shutdowns":list(shutdown),"events":events,"reward_components":reward_parts,"potential_before":phi_before,"potential_after":self.potential(),"energy_gained":energy_gained})
        # Metadata is stored as provenance only; it never affects dynamics/reward.
        if decision_metadata is not None:
            info["decision_metadata"] = deepcopy(dict(decision_metadata))
        return self.observations(),rewards,terminated,truncated,info

    def _info(self):
        return {"environment_version":ENVIRONMENT_VERSION,"reward_version":REWARD_VERSION,"frame":self.state.frame,"native_score":self.native_score,"participant_score":self.native_score,"legacy_score":None,"legacy_score_status":"not_computed_offline_shadow_only","score_breakdown":dict(self.state.score_breakdown),"original_score_formula":{"delivery":self.config.delivery_points,"collision":self.config.robot_collision_points,"shutdown":self.config.shutdown_points,"step":self.config.step_points,"human_detour_per_unit":self.config.human_detour_points_per_unit},"native_detour_penalty":0.}

    def public_view(self):
        state = physical_state(self.state)
        state.update({"native_score":self.native_score,"participant_score":self.native_score,"max_steps":self.config.horizon,"map":{"layout_id":self.layout.layout_id,"tiles":list(self.layout.tiles),"charger_position":self.layout.charger_position},"legacy_score":None})
        return state

    def set_state(self,state):
        candidate = deepcopy(state)
        if [a.agent_id for a in candidate.agents] != list(self.agent_ids):
            raise ValueError("Invalid robot identities")
        if len({a.position for a in candidate.agents}) != 2 or any(not self.layout.is_passable(a.position) or not 0<=a.battery<=100 for a in candidate.agents):
            raise ValueError("Invalid robot physical state")
        if len(candidate.tasks)!=self.config.active_task_count:
            raise ValueError("Invalid task count")
        self.state = candidate

    def snapshot(self):
        return {"version":ENVIRONMENT_VERSION,"configuration":asdict(self.config),"state":asdict(self.get_state()),"rng":self.get_rng_state(),"episode_counter":self._episode_counter}

    def restore(self,payload):
        if payload.get("version") != ENVIRONMENT_VERSION:
            raise ValueError("Native snapshot version mismatch")
        if payload.get("configuration") != asdict(self.config):
            raise ValueError("Native snapshot configuration mismatch")
        data = deepcopy(payload["state"])
        def task(value):
            for key in ("pickup_position","delivery_position"):
                value[key]=tuple(value[key])
            return DeliveryTask(**value)
        agents=[]
        for value in data["agents"]:
            for key in ("position","navigation_goal_position","recent_goal_types"):
                value[key]=tuple(value[key])
            value["recent_positions"]=tuple(tuple(p) for p in value["recent_positions"])
            agents.append(AgentState(**value))
        data["agents"]=agents
        data["tasks"]=[task(value) for value in data["tasks"]]
        data["completed_tasks"]=[task(value) for value in data["completed_tasks"]]
        data["last_coordination_events"]=tuple(data["last_coordination_events"])
        self.set_state(WarehouseState(**data))
        def tuples(value):
            return tuple(tuples(x) for x in value) if isinstance(value,(tuple,list)) else value
        self.set_rng_state(tuples(payload["rng"]))
        self._episode_counter=int(payload["episode_counter"])

    def branch(self):
        branch=type(self)(self.config)
        branch.restore(self.snapshot())
        return branch

    def fingerprint(self):
        content={"map":self.layout.tiles,"charger":self.layout.charger_position,"state":physical_state(self.state)}
        return sha256(json.dumps(content,sort_keys=True,separators=(",",":")).encode()).hexdigest()
