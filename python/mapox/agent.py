import jax
from jax import numpy as jnp
import numpy as np

from mapox.timestep import TimeStep
from mapox.specs import ActionSpec
from typing import Protocol

class Agent(Protocol):
    def act(self, timestep: TimeStep) -> jax.Array: ...

    def reset(self, num_agents: int, seed: int) -> None:
        ...


class RandomAgent(Agent):
    def __init__(self, action_spec: ActionSpec):
        self._action_spec = action_spec
        self._rng_key = jax.random.key(0)

    def act(self, timestep: TimeStep) -> jax.Array:
        num_agents = timestep.obs.shape[0]

        action_rng, self._rng_key = jax.random.split(self._rng_key)
        logits = jax.random.uniform(action_rng, (num_agents, self._action_spec.n))
        actions = jnp.argmax(
            jnp.where(timestep.action_mask, logits, -jnp.inf), axis=-1
        ).astype(self._action_spec.dtype)

        return actions

    def reset(self, num_agents: int, seed: int) -> None:
        self._rng_key = jax.random.key(np.uint64(seed).astype(np.int64))
