import jax
from typing import NamedTuple

Metrics = dict[str, jax.Array]


class TimeStep(NamedTuple):
    """The observation that the agent sees.

    agents_view: the agent's view of the environment.
    action_mask: boolean array specifying, for each agent, which action is legal.
    step_count: the number of steps elapsed since the beginning of the episode.
    """

    obs: jax.Array  # (num_agents, num_obs_features)
    time: jax.Array
    terminated: jax.Array
    last_action: jax.Array
    reward: jax.Array
    action_mask: jax.Array | None  # (num_agents, num_actions)
    task_ids: jax.Array | None = None
