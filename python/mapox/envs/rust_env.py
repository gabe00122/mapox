from functools import cached_property
from typing import Any

import jax
import numpy as np
from jax import Array
from jax import numpy as jnp
from jax.experimental import io_callback
from pydantic import BaseModel

from mapox._core import Env as _CoreEnv
from mapox.environment import Environment
from mapox.envs.common import make_obs_spec
from mapox.renderer import GridRenderSettings, GridRenderState
from mapox.specs import DiscreteActionSpec, ObservationSpec
from mapox.timestep import TimeStep
from mapox.vocab import Vocabulary


class RustEnv(Environment[None]):
    """Exposes the compiled `mapox._core.Env` through the jax `Environment`
    interface via ordered `io_callback`s.

    The episode state lives inside the rust env, so the python-side EnvState
    is always ``None`` and the wrapper is stateful: callbacks are ordered so
    reset/step cannot be reordered or elided by jit, and one wrapper instance
    must not be shared across traces that interleave (in particular, ordered
    callbacks do not support vmap — vectorize on the rust side instead).

    Each instance owns persistent numpy buffers that the rust env writes
    timesteps into; io_callback copies them into device arrays on return.
    """

    def __init__(self, config: str | BaseModel, num_envs: int = 1):
        # accepts either a serialized rust EnvConfig or a pydantic config
        # (e.g. mapox.envs.find_return.FindReturnConfig) that mirrors one
        config_json = (
            config.model_dump_json() if isinstance(config, BaseModel) else config
        )
        self._env = _CoreEnv(config_json, num_envs)

        num_agents, view_width, view_height, channels = self._env.observation_shape
        num_actions = self._env.num_actions
        self._view_width = view_width
        self._view_height = view_height

        self._obs_vocab = Vocabulary(self._env.obs_symbols).freeze()
        self._action_vocab = Vocabulary(self._env.action_symbols).freeze()

        self._obs = np.zeros((num_agents, view_width, view_height, channels), np.uint16)
        self._time = np.zeros((num_agents,), np.int32)
        self._terminated = np.zeros((num_agents,), np.bool_)
        self._last_action = np.zeros((num_agents,), np.uint16)
        self._reward = np.zeros((num_agents,), np.float32)
        self._action_mask = np.zeros((num_agents, num_actions), np.bool_)
        self._task_ids = np.zeros((num_agents,), np.int32)

        self._result_shapes = TimeStep(
            obs=jax.ShapeDtypeStruct(self._obs.shape, jnp.uint16),
            time=jax.ShapeDtypeStruct(self._time.shape, jnp.int32),
            terminated=jax.ShapeDtypeStruct(self._terminated.shape, jnp.bool_),
            last_action=jax.ShapeDtypeStruct(self._last_action.shape, jnp.uint16),
            reward=jax.ShapeDtypeStruct(self._reward.shape, jnp.float32),
            action_mask=jax.ShapeDtypeStruct(self._action_mask.shape, jnp.bool_),
            task_ids=jax.ShapeDtypeStruct(self._task_ids.shape, jnp.int32),
        )

    # jit static-arg caching keys on (hash, eq); identity semantics so two
    # wrappers never share a cache entry — each owns a distinct rust env
    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)

    def _timestep(self) -> TimeStep:
        return TimeStep(
            # Rust and JAX share uint16 buffers for observations and actions.
            obs=self._obs,
            time=self._time,
            terminated=self._terminated,
            last_action=self._last_action,
            reward=self._reward,
            action_mask=self._action_mask,
            task_ids=self._task_ids,
        )

    def _reset_callback(self, seed: np.ndarray) -> TimeStep:
        self._env.reset(
            int(seed),
            self._obs,
            self._time,
            self._terminated,
            self._last_action,
            self._reward,
            self._action_mask,
            self._task_ids,
        )
        return self._timestep()

    def _step_callback(self, action: np.ndarray) -> TimeStep:
        if np.any(action < 0) or np.any(action > np.iinfo(np.uint16).max):
            raise ValueError("action id outside the Rust VocabId range")
        self._env.step(
            np.ascontiguousarray(action, dtype=np.uint16),
            self._obs,
            self._time,
            self._terminated,
            self._last_action,
            self._reward,
            self._action_mask,
            self._task_ids,
        )
        return self._timestep()

    def reset(self, rng_key: Array) -> tuple[None, TimeStep]:
        seed = jax.random.bits(rng_key, dtype=jnp.uint32)
        timestep = io_callback(
            self._reset_callback, self._result_shapes, seed, ordered=True
        )
        return None, timestep

    def step(
        self, state: None, action: Array, rng_key: Array
    ) -> tuple[None, TimeStep]:
        # the rust env owns its rng (seeded at reset), so rng_key is unused
        del state, rng_key
        timestep = io_callback(
            self._step_callback, self._result_shapes, action, ordered=True
        )
        return None, timestep

    def create_placeholder_logs(self) -> dict[str, Any]:
        # TODO: logging isn't wired through the rust env yet
        return {}

    def create_logs(self, state: None) -> dict[str, Any]:
        return {}

    @cached_property
    def observation_spec(self) -> ObservationSpec:
        # rust emits the leading OBS_CHANNELS of the canonical 4-channel
        # layout (currently just tile ids), so truncate the canonical spec
        # to the channel count the env actually produces
        full = make_obs_spec(self._view_width, self._view_height, len(self._obs_vocab))
        channels = self._obs.shape[-1]
        if channels > full.shape[-1]:
            raise ValueError(
                f"rust env produces {channels} obs channels, more than the "
                f"{full.shape[-1]} the canonical layout defines"
            )
        assert isinstance(full.max_value, tuple)
        return full._replace(
            shape=(*full.shape[:-1], channels),
            max_value=full.max_value[:channels],
        )

    @cached_property
    def action_spec(self) -> DiscreteActionSpec:
        return DiscreteActionSpec(n=self._env.num_actions)

    @property
    def num_agents(self) -> int:
        return self._env.num_agents

    def get_render_settings(self) -> GridRenderSettings:
        raise NotImplementedError(
            "RustEnv does not expose render state; use mapox.run_demo for the "
            "native viewer"
        )

    def get_render_state(self, state: None) -> GridRenderState:
        raise NotImplementedError(
            "RustEnv does not expose render state; use mapox.run_demo for the "
            "native viewer"
        )

    @property
    def obs_vocab(self) -> Vocabulary:
        return self._obs_vocab

    @property
    def action_vocab(self) -> Vocabulary:
        return self._action_vocab
