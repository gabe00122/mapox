import jax

from mapox import TimeStep, ActionSpec
from typing import Protocol, TypeVar, Generic

AgentState = TypeVar("AgentState")

class Agent(Protocol, Generic[AgentState]):
    def sample_actions(self, agent_state: AgentState, timestep: TimeStep, rng_key: jax.Array) -> tuple[AgentState, jax.Array, jax.Array]:
        ...


class RandomAgent(Agent[None]):
    def __init__(self, action_spec: ActionSpec):
        self._action_spec = action_spec

    def sample_actions(self, agent_state: None, timestep: TimeStep, rng_key: jax.Array) -> tuple[None, jax.Array, jax.Array]:
        action_shape = timestep.obs.shape[:1]

        action_rng, rng_key = jax.random.split(rng_key)
        action = jax.random.randint(action_rng, shape=action_shape, minval=0, maxval=self._action_spec.n)

        return None, action, rng_key
