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

        envs: tuple[Environment, ...] = [TaskIdWrapper(env, task_id) for task_id, env in enumerate(envs)]

        for env in envs:
            self._action_vocab.extend(env.action_vocab.symbols)
            self._obs_vocab.extend(env.obs_vocab.symbols)

        self._envs = envs
        self._env_names = env_names

        self._global_to_local_action = tuple([self.action_vocab.lut_to(env.action_vocab) for env in self._envs])
        self._local_to_global_action = tuple([env.action_vocab.lut_to(self.action_vocab) for env in self._envs])
        self._local_to_global_obs = tuple([env.obs_vocab.lut_to(self.obs_vocab) for env in self._envs])

        self._action_vocab.freeze()
        self._obs_vocab.freeze()

    def reset(self, rng_key: jax.Array):
        rng_keys = jax.random.split(rng_key, len(self._envs))

        states = []
        timesteps = []

        for i, env in enumerate(self._envs):
            action_lut = self._local_to_global_action[i]
            obs_lut = self._local_to_global_obs[i]

            s, t = env.reset(rng_keys[i])
            t = t._replace(
                last_action=action_lut[t.last_action],
                obs=obs_lut[t.obs]
            )

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
            t = t._replace(
                last_action=action_lut[t.last_action],
                obs=obs_lut[t.obs]
            )

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

        return first_spec._replace(max_value=len(self.obs_vocab))

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
