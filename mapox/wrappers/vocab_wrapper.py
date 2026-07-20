from typing import Any
from functools import cached_property

import jax
from jax import numpy as jnp

from mapox.environment import Environment, EnvState
from mapox.envs.common import make_obs_spec
from mapox.renderer import GridRenderState, GridRenderSettings
from mapox.specs import ActionSpec, DiscreteActionSpec, ObservationSpec
from mapox.timestep import TimeStep
from mapox.vocab import Vocabulary


class VocabWrapper(Environment):
    """Presents a local-vocab env in a global vocabulary.

    Global action ids in, global obs/action ids out; translation happens at
    the boundary. Global symbols the inner env lacks map to local id 0 (they
    are masked out, so the agent never legally selects them).
    """

    def __init__(
        self,
        env: Environment,
        action_vocab: Vocabulary,
        obs_vocab: Vocabulary,
    ):
        self._env = env
        self._action_vocab = action_vocab
        self._obs_vocab = obs_vocab

        self._global_to_local_action = action_vocab.lut_to(
            env.action_vocab, default=0
        )
        self._local_to_global_action = env.action_vocab.lut_to(
            action_vocab, default=0
        )
        # int8 so tile-channel scatter matches the obs dtype; make_obs_spec
        # guarantees the vocab fits.
        self._local_to_global_obs = env.obs_vocab.lut_to(
            obs_vocab, default=0, dtype=jnp.int8
        )

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _encode_timestep(self, ts: TimeStep) -> TimeStep:
        mask = jnp.zeros(
            (*ts.action_mask.shape[:-1], len(self._action_vocab)), jnp.bool_
        )
        mask = mask.at[..., self._local_to_global_action].set(ts.action_mask)

        # only the first obs channel goes through the lookup
        obs = ts.obs.at[..., 0].set(self._local_to_global_obs[ts.obs[..., 0]])

        return ts._replace(
            last_action=self._local_to_global_action[ts.last_action],
            obs=obs,
            action_mask=mask,
        )

    def reset(self, rng_key: jax.Array):
        state, ts = self._env.reset(rng_key)
        return state, self._encode_timestep(ts)

    def step(self, state, action: jax.Array, rng_key: jax.Array):
        state, ts = self._env.step(
            state, self._global_to_local_action[action], rng_key
        )
        return state, self._encode_timestep(ts)

    @cached_property
    def observation_spec(self) -> ObservationSpec:
        spec = self._env.observation_spec
        if spec.max_value is None:
            raise ValueError("Obs must have a max value")
        if len(spec.shape) != 3 or spec.shape[2] != 4:
            raise ValueError("VocabWrapper only supports obs of (width, height, 4)")

        return make_obs_spec(spec.shape[0], spec.shape[1], len(self._obs_vocab))

    @cached_property
    def action_spec(self) -> ActionSpec:
        return DiscreteActionSpec(len(self._action_vocab))

    @property
    def obs_vocab(self) -> Vocabulary:
        return self._obs_vocab

    @property
    def action_vocab(self) -> Vocabulary:
        return self._action_vocab

    @property
    def num_agents(self) -> int:
        return self._env.num_agents

    @property
    def teams(self) -> jax.Array | None:
        # Environment.teams has a concrete default, so __getattr__ never
        # delegates it; forward explicitly.
        return self._env.teams

    def create_placeholder_logs(self) -> dict[str, Any]:
        return self._env.create_placeholder_logs()

    def create_logs(self, state) -> dict[str, Any]:
        return self._env.create_logs(state)

    def get_render_settings(self) -> GridRenderSettings:
        return self._env.get_render_settings()

    def get_render_state(self, state: EnvState) -> GridRenderState:
        return self._env.get_render_state(state)
