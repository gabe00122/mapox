from typing import Any

import numpy as np

class Env:
    """Compiled gridworld environment; see crates/mapox-py."""

    def __init__(self, config_json: str, length: int, num_envs: int) -> None: ...

    num_agents: int
    num_actions: int
    observation_shape: tuple[int, int, int, int]
    obs_symbols: list[str]
    action_symbols: list[str]

    def reset(
        self,
        seed: int,
        obs: np.ndarray,
        time: np.ndarray,
        terminated: np.ndarray,
        last_action: np.ndarray,
        reward: np.ndarray,
        action_mask: np.ndarray,
        task_ids: np.ndarray,
    ) -> None: ...

    def step(
        self,
        actions: np.ndarray,
        obs: np.ndarray,
        time: np.ndarray,
        terminated: np.ndarray,
        last_action: np.ndarray,
        reward: np.ndarray,
        action_mask: np.ndarray,
        task_ids: np.ndarray,
    ) -> None: ...


def enjoy(
    env: Env,
    length: int,
    seed: int,
    policy: Any | None = None,
) -> None: ...
