"""JAX view of the rust env.

Wraps `rust_env_numpy.RustEnvNumpy`: the rust side steps on the host, so each
jax reset/step crosses the boundary once through `io_callback`, which marshals
the jax seed/action to numpy for the host call and reads the env's shared
buffers back out as jax arrays.
"""

from functools import cached_property
from typing import Any, cast

import jax
import numpy as np
from jax import Array
from jax import numpy as jnp
from jax.experimental import io_callback
from pydantic import BaseModel

from mapox.environment import Environment
from mapox.envs.rust_env_numpy import RustEnvNumpy
from mapox.renderer import GridRenderSettings, GridRenderState
from mapox.specs import DiscreteActionSpec, ObservationSpec
from mapox.timestep import TimeStep
from mapox.vocab import Vocabulary


def _shape_placeholder(buffer: np.ndarray) -> jax.Array:
    # io_callback result-spec leaf: a shape/dtype placeholder standing in for
    # a shared buffer. The machinery only reads .shape/.dtype, never the value,
    # but the spec must be a TimeStep for io_callback to return one.
    return cast(jax.Array, jax.ShapeDtypeStruct(buffer.shape, buffer.dtype))


class RustEnvJax(Environment[None]):
    def __init__(self, config: BaseModel, length: int):
        self._env = RustEnvNumpy(config, length)
        self._result_shapes = jax.tree.map(_shape_placeholder, self._env.buffers)

    # jit static-arg caching keys on (hash, eq); identity semantics so two
    # wrappers never share a cache entry — each owns a distinct rust env
    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)

    def reset(self, rng_key: Array) -> tuple[None, TimeStep]:
        seed = jax.random.bits(rng_key, dtype=jnp.uint32)
        timestep = io_callback(
            self._env.reset, self._result_shapes, seed, ordered=True
        )
        return None, timestep

    def step(self, state: None, action: Array, rng_key: Array) -> tuple[None, TimeStep]:
        # the rust env owns its rng (seeded at reset), so rng_key is unused
        del state, rng_key
        timestep = io_callback(
            self._env.step, self._result_shapes, action, ordered=True
        )
        return None, timestep

    def create_placeholder_logs(self) -> dict[str, Any]:
        # TODO: logging isn't wired through the rust env yet
        return {}

    def create_logs(self, state: None) -> dict[str, Any]:
        return {}

    @cached_property
    def observation_spec(self) -> ObservationSpec:
        return self._env.observation_spec

    @cached_property
    def action_spec(self) -> DiscreteActionSpec:
        return self._env.action_spec

    @property
    def num_agents(self) -> int:
        return self._env.num_agents

    @property
    def num_tasks(self) -> int:
        return self._env.num_tasks

    def get_render_settings(self) -> GridRenderSettings:
        raise NotImplementedError(
            "the rust env does not expose render state; use mapox.run_demo for "
            "the native viewer"
        )

    def get_render_state(self, state: None) -> GridRenderState:
        raise NotImplementedError(
            "the rust env does not expose render state; use mapox.run_demo for "
            "the native viewer"
        )

    @property
    def obs_vocab(self) -> Vocabulary:
        return self._env.obs_vocab

    @property
    def action_vocab(self) -> Vocabulary:
        return self._env.action_vocab

    def set_enjoy_mode(self, task_id: int | None) -> None:
        self._env.set_enjoy_mode(task_id)

    def consume_metrics(self) -> dict[str, Any]:
        return self._env.consume_metrics()

    @property
    def inner(self):
        """The raw pyo3 env, for host-side drivers like `mapox._core.enjoy`."""
        return self._env.inner
