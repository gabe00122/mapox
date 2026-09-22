"""A single active environment driven one agent action at a time.

The session owns a rust env (`mapox._core.Env`) with numpy buffers, a seeded
rng for the uncontrolled agents, and the ASCII rendering tables. Uncontrolled
agents sample uniformly from their own `action_mask` row each step — the same
semantics as `mapox_core::policy::RandomPolicy`.
"""

import threading

import numpy as np

from mapox._core import Env as CoreEnv
from mapox.server.ascii import LEGEND, build_char_table, render_grid


class AgentIdError(ValueError): ...
class ActionSymbolError(ValueError): ...
class IllegalActionError(ValueError):
    def __init__(self, action: str, agent_id: int, legal_symbols: list[str]):
        super().__init__(
            f"action {action!r} is illegal for agent {agent_id}; "
            f"legal actions: {legal_symbols}"
        )
        self.legal_symbols = legal_symbols


class PlaySession:
    def __init__(self, config_json: str, length: int, seed: int | None):
        self._env = CoreEnv(config_json, length)
        self._length = length
        self._rng = np.random.default_rng(seed)

        (
            self._num_agents,
            self._view_width,
            self._view_height,
            self._obs_channels,
        ) = self._env.observation_shape
        self._num_actions = self._env.num_actions
        self._obs_symbols = list(self._env.obs_symbols)
        self._action_symbols = list(self._env.action_symbols)
        self._symbol_to_action = {s: i for i, s in enumerate(self._action_symbols)}

        self._obs = np.zeros(
            (self._num_agents, self._view_width, self._view_height, self._obs_channels),
            np.uint16,
        )
        self._time = np.zeros((self._num_agents,), np.int32)
        self._terminated = np.zeros((self._num_agents,), np.bool_)
        self._last_action = np.zeros((self._num_agents,), np.uint16)
        self._reward = np.zeros((self._num_agents,), np.float32)
        self._action_mask = np.zeros((self._num_agents, self._num_actions), np.bool_)
        self._task_ids = np.zeros((self._num_agents,), np.int32)

        self._char_table = build_char_table(self._obs_symbols)

        self._env.reset(0 if seed is None else seed, self._obs, self._time,
                        self._terminated, self._last_action, self._reward,
                        self._action_mask, self._task_ids)

    def info(self) -> dict:
        return {
            "num_agents": self._num_agents,
            "obs_shape": [self._view_width, self._view_height],
            "actions": self._action_symbols,
            "ascii_legend": dict(LEGEND),
            "time": int(self._time[0]) if self._num_agents else 0,
        }

    def obs_ascii(self, agent_id: int) -> list[str]:
        self._check_agent(agent_id)
        return render_grid(self._obs[agent_id, :, :, 0], self._char_table)

    def act(self, agent_id: int, action: str) -> dict:
        self._check_agent(agent_id)
        action_id = self._resolve_action(agent_id, action)

        actions = self._random_actions()
        actions[agent_id] = action_id
        self._env.step(actions, self._obs, self._time, self._terminated,
                        self._last_action, self._reward, self._action_mask,
                        self._task_ids)

        return self.agent_report(agent_id)

    def _random_actions(self) -> np.ndarray:
        """Uniform over each uncontrolled agent's legal actions."""

        # The argmax of i.i.d. uniforms restricted to a row's legal entries
        # (illegal entries set to -1) is uniform over those entries.
        noise = self._rng.random(self._action_mask.shape)
        return np.argmax(
            np.where(self._action_mask, noise, -1.0), axis=1
        ).astype(np.uint16)

    def agent_report(self, agent_id: int) -> dict:
        return {
            "agent_id": agent_id,
            "last_action": self._action_symbols[int(self._last_action[agent_id])],
            "reward": float(self._reward[agent_id]),
            "time": int(self._time[agent_id]),
            "terminated": bool(self._terminated[agent_id]),
            "obs_ascii": self.obs_ascii(agent_id),
            "legal_actions": [
                self._action_symbols[i]
                for i in np.flatnonzero(self._action_mask[agent_id])
            ],
        }

    def _check_agent(self, agent_id: int) -> None:
        if not 0 <= agent_id < self._num_agents:
            raise AgentIdError(
                f"agent_id {agent_id} out of range for {self._num_agents} agents"
            )

    def _resolve_action(self, agent_id: int, action: str) -> int:
        action_id = self._symbol_to_action.get(action)
        if action_id is None:
            raise ActionSymbolError(
                f"unknown action {action!r}; "
                f"known: {self._action_symbols}"
            )

        if not self._action_mask[agent_id, action_id]:
            raise IllegalActionError(
                str(action), agent_id,
                [self._action_symbols[i]
                 for i in np.flatnonzero(self._action_mask[agent_id])],
            )
        return action_id
