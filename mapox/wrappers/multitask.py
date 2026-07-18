from mapox.envs.common import make_obs_spec
from mapox import TimeStep
from mapox.vocab import Vocabulary
from functools import cached_property

import jax
from jax import numpy as jnp

from mapox.environment import Environment
from mapox.specs import ActionSpec, DiscreteActionSpec, ObservationSpec
from mapox.wrappers.task_id_wrapper import TaskIdWrapper


def _stack_pytree(batch):
    return jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *batch)


class MultiTaskWrapper(Environment):
    def __init__(self, envs: tuple[Environment], env_names: tuple[str]) -> None:
        self._action_vocab = Vocabulary()
        self._obs_vocab = Vocabulary()

        envs: list[Environment] = [TaskIdWrapper(env, task_id) for task_id, env in enumerate(envs)]

        for env in envs:
            self._action_vocab.extend(env.action_vocab.symbols)
            self._obs_vocab.extend(env.obs_vocab.symbols)

        self._envs = tuple(envs)
        self._env_names = env_names

        self._global_to_local_action = tuple([self.action_vocab.lut_to(env.action_vocab, default=0) for env in self._envs])
        self._local_to_global_action = tuple([env.action_vocab.lut_to(self.action_vocab, default=0) for env in self._envs])
        self._local_to_global_obs = tuple([env.obs_vocab.lut_to(self.obs_vocab, default=0) for env in self._envs])

        self._action_vocab.freeze()
        self._obs_vocab.freeze()

    def _encode_timestamp(self, ts: TimeStep, action_lut: jax.Array, obs_lut: jax.Array) -> TimeStep:
        mask = jnp.zeros((*ts.action_mask.shape[:-1], len(self._action_vocab)), jnp.bool_)
        mask = mask.at[..., action_lut].set(ts.action_mask)

        # only the first obs channel goes through the lookup
        obs = ts.obs.at[..., 0].set(obs_lut[ts.obs[..., 0]])

        return ts._replace(
            last_action=action_lut[ts.last_action],
            obs=obs,
            action_mask=mask,
        )

    def reset(self, rng_key: jax.Array):
        rng_keys = jax.random.split(rng_key, len(self._envs))

        states = []
        timesteps = []

        for i, env in enumerate(self._envs):
            action_lut = self._local_to_global_action[i]
            obs_lut = self._local_to_global_obs[i]

            s, t = env.reset(rng_keys[i])
            t = self._encode_timestamp(t, action_lut, obs_lut)

            states.append(s)
            timesteps.append(t)

        return tuple(states), _stack_pytree(timesteps)

    def step(self, states, actions: jax.Array, rng_key: jax.Array):
        rng_keys = jax.random.split(rng_key, len(self._envs))

        state_out = []
        timesteps = []

        start = 0
        for i, env in enumerate(self._envs):
            action_lut = self._global_to_local_action[i]

            end = start + env.num_agents
            env_actions = actions[start:end]
            env_actions = action_lut[env_actions]

            s, t = env.step(states[i], env_actions, rng_keys[i])
            start = end

            action_lut = self._local_to_global_action[i]
            obs_lut = self._local_to_global_obs[i]
            t = self._encode_timestamp(t, action_lut, obs_lut)

            state_out.append(s)
            timesteps.append(t)

        return tuple(state_out), _stack_pytree(timesteps)

    @property
    def obs_vocab(self):
        return self._obs_vocab

    @property
    def action_vocab(self):
        return self._action_vocab

    @cached_property
    def observation_spec(self) -> ObservationSpec:
        first_spec = self._envs[0].observation_spec
        if first_spec.max_value is None:
            raise ValueError("Obs must have a max value")
        if len(first_spec.shape) != 3 or first_spec.shape[2] != 4:
            raise ValueError("Multitask wrapper only supports obs of (width, height, 4)")

        return make_obs_spec(first_spec.shape[0], first_spec.shape[1], len(self.obs_vocab))

    @cached_property
    def action_spec(self) -> ActionSpec:
        return DiscreteActionSpec(len(self.action_vocab))

    @property
    def num_agents(self) -> int:
        return sum([env.num_agents for env in self._envs])

    @property
    def num_tasks(self) -> int:
        return len(self._envs)

    @property
    def teams(self) -> jax.Array:
        return jnp.concatenate([env.teams for env in self._envs], axis=0)

    def create_placeholder_logs(self):
        return {
            name: env.create_placeholder_logs()
            for name, env in zip(self._env_names, self._envs)
        }

    def get_render_settings(self):
        raise NotImplementedError("MultiTaskWrapper does not support rendering")

    def get_render_state(self, state):
        raise NotImplementedError("MultiTaskWrapper does not support rendering")

    def create_logs(self, state):
        return {
            name: env.create_logs(s)
            for name, env, s in zip(self._env_names, self._envs, state)
        }
