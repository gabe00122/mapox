import jax
from jax import numpy as jnp

from mapox.timestep import TimeStep
from mapox.specs import ActionSpec
from typing import Protocol, TypeVar, Generic

AgentState = TypeVar("AgentState")

class Agent(Protocol, Generic[AgentState]):
    def sample_actions(self, agent_state: AgentState, timestep: TimeStep, rng_key: jax.Array) -> tuple[AgentState, jax.Array, jax.Array]:
        ...


class RandomAgent(Agent[None]):
    def __init__(self, action_spec: ActionSpec):
        self._action_spec = action_spec

    def sample_actions(self, agent_state: None, timestep: TimeStep, rng_key: jax.Array) -> tuple[None, jax.Array, jax.Array]:
        num_agents = timestep.obs.shape[0]

        action_rng, rng_key = jax.random.split(rng_key)
        logits = jax.random.uniform(action_rng, (num_agents, self._action_spec.n))
        actions = jnp.argmax(jnp.where(timestep.action_mask, logits, -jnp.inf), axis=-1)

        return None, actions, rng_key
