from mapox.renderer import GridRenderSettings, GridRenderState
from typing import Any
from functools import cached_property
import jax
from jax import numpy as jnp
from einops import rearrange

from mapox.environment import Environment, EnvState
from mapox.specs import ActionSpec, ObservationSpec
from mapox.timestep import TimeStep


class VectorWrapper(Environment[EnvState]):
    def __init__(self, env: Environment[EnvState], vec_count: int):
        self._env = env
        self._vec_count = vec_count

    def reset(self, rng_key: jax.Array) -> tuple[EnvState, TimeStep]:
        rng_keys = jax.random.split(rng_key, self._vec_count)
        state, timestep = jax.vmap(self._env.reset)(rng_keys)
        return state, self._flatten_timestep(timestep)

    def step(
        self, state: EnvState, action: jax.Array, rng_key: jax.Array
    ) -> tuple[EnvState, TimeStep]:
        rng_keys = jax.random.split(rng_key, self._vec_count)

        action = action.reshape(self._vec_count, self._env.num_agents)
        state, timestep = jax.vmap(
            self._env.step, in_axes=(0, 0, 0), out_axes=(0, 0)
        )(state, action, rng_keys)
        return state, self._flatten_timestep(timestep)

    @property
    def num_agents(self) -> int:
        return self._vec_count * self._env.num_agents

    @property
    def teams(self) -> jax.Array | None:
        if self._env.teams is None:
            return None
        else:
            return jnp.tile(self._env.teams, self._vec_count)

    def _flatten_timestep(self, timestep: TimeStep) -> TimeStep:
        return jax.tree.map(
            lambda x: rearrange(x, "b a ... -> (b a) ...") if x is not None else None,
            timestep,
        )

    @cached_property
    def observation_spec(self) -> ObservationSpec:
        return self._env.observation_spec

    @cached_property
    def action_spec(self) -> ActionSpec:
        return self._env.action_spec

    def create_logs(self, state):
        log_updates = jax.vmap(self._env.create_logs)(state)
        log_updates = jax.tree.map(lambda xs: jnp.mean(xs, axis=0), log_updates)

        return log_updates

    def create_placeholder_logs(self) -> dict[str, Any]:
        return self._env.create_placeholder_logs()

    @property
    def is_jittable(self) -> bool:
        return self._env.is_jittable

    def get_render_settings(self) -> GridRenderSettings:
        return self._env.get_render_settings()

    def get_render_state(self, state: EnvState) -> GridRenderState:
        return self._env.get_render_state(state)
