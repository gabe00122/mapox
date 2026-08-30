from typing import NamedTuple

import jax
import numpy as np


class TimeStep(NamedTuple):
    """The observation that the agent sees.

    Leaves are jax arrays under jit; the rust env's host callbacks fill
    shared numpy buffers in place, so leaves may be ndarrays outside jit.

    obs: the agent's view of the environment.
    action_mask: boolean array specifying, for each agent, which action is legal.
    time: the number of steps elapsed since the beginning of the episode.
    """

    obs: jax.Array | np.ndarray  # (num_agents, num_obs_features)
    time: jax.Array | np.ndarray
    terminated: jax.Array | np.ndarray
    last_action: jax.Array | np.ndarray
    reward: jax.Array | np.ndarray
    action_mask: jax.Array | np.ndarray  # (num_agents, num_actions)
    task_ids: jax.Array | np.ndarray | None = None
