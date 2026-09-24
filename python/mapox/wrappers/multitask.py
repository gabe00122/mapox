from functools import cached_property

import jax
from jax import numpy as jnp

from mapox.environment import Environment
from mapox.renderer import GridRenderSettings
from mapox.specs import ActionSpec, DiscreteActionSpec, ObservationSpec
from mapox.vocab import Vocabulary
from mapox.wrappers.task_id_wrapper import TaskIdWrapper
from mapox.wrappers.vocab_wrapper import VocabWrapper


def _stack_pytree(batch):
    return jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *batch)


class MultiTaskWrapper(Environment):
    def __init__(
        self, envs: tuple[Environment, ...], env_names: tuple[str, ...]
    ) -> None:
        self._action_vocab = Vocabulary()
        self._obs_vocab = Vocabulary()

        for env in envs:
            self._action_vocab.extend(env.action_vocab.symbols)
            self._obs_vocab.extend(env.obs_vocab.symbols)

        self._action_vocab.freeze()
        self._obs_vocab.freeze()

        self._envs = tuple(
            VocabWrapper(
                TaskIdWrapper(env, task_id), self._action_vocab, self._obs_vocab
            )
            for task_id, env in enumerate(envs)
        )
        self._env_names = env_names
        self._enjoy_mode: int | None = None

    def set_enjoy_mode(self, task_id: int | None) -> None:
        if task_id is not None and not 0 <= task_id < len(self._envs):
            raise ValueError(
                f"task_id {task_id} out of range; this env has {len(self._envs)} tasks"
            )

        if self._enjoy_mode is not None:
            self._envs[self._enjoy_mode].set_enjoy_mode(None)

        self._enjoy_mode = task_id
        if task_id is not None:
            self._envs[task_id].set_enjoy_mode(0)

    @property
    def task_envs(self) -> tuple[VocabWrapper, ...]:
        """The wrapped sub-envs; each one presents the global vocabulary."""
        return self._envs

    def reset(self, rng_key: jax.Array):
        if self._enjoy_mode is not None:
            return self._envs[self._enjoy_mode].reset(rng_key)

        rng_keys = jax.random.split(rng_key, len(self._envs))

        states = []
        timesteps = []

        for env, env_key in zip(self._envs, rng_keys):
            s, t = env.reset(env_key)

            states.append(s)
            timesteps.append(t)

        return tuple(states), _stack_pytree(timesteps)

    def step(self, state, action: jax.Array, rng_key: jax.Array):
        if self._enjoy_mode is not None:
            return self._envs[self._enjoy_mode].step(state, action, rng_key)

        rng_keys = jax.random.split(rng_key, len(self._envs))

        state_out = []
        timesteps = []

        start = 0
        for i, env in enumerate(self._envs):
            end = start + env.num_agents
            s, t = env.step(state[i], action[start:end], rng_keys[i])
            start = end

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
        return self._envs[0].observation_spec

    @cached_property
    def action_spec(self) -> ActionSpec:
        return DiscreteActionSpec(len(self.action_vocab))

    @property
    def num_agents(self) -> int:
        if self._enjoy_mode is not None:
            return self._envs[self._enjoy_mode].num_agents
        return sum([env.num_agents for env in self._envs])

    @property
    def num_tasks(self) -> int:
        return len(self._envs)

    @property
    def task_names(self) -> list[str]:
        return list(self._env_names)

    @property
    def teams(self) -> jax.Array | None:
        if self._enjoy_mode is not None:
            return self._envs[self._enjoy_mode].teams

        all_teams = [env.teams for env in self._envs]
        if all(teams is None for teams in all_teams):
            return None

        # Envs without teams count as a single team (team 0)
        return jnp.concatenate(
            [
                teams if teams is not None else jnp.zeros(env.num_agents, jnp.int8)
                for env, teams in zip(self._envs, all_teams)
            ],
            axis=0,
        )

    def create_placeholder_logs(self):
        if self._enjoy_mode is not None:
            return {
                self._env_names[self._enjoy_mode]: self._envs[
                    self._enjoy_mode
                ].create_placeholder_logs()
            }
        return {
            name: env.create_placeholder_logs()
            for name, env in zip(self._env_names, self._envs)
        }

    def get_render_settings(self) -> GridRenderSettings:
        if self._enjoy_mode is not None:
            return self._envs[self._enjoy_mode].get_render_settings()
        raise NotImplementedError(
            "MultiTaskWrapper renders one task at a time; call set_enjoy_mode first"
        )

    def get_render_state(self, state):
        if self._enjoy_mode is not None:
            return self._envs[self._enjoy_mode].get_render_state(state)
        raise NotImplementedError(
            "MultiTaskWrapper renders one task at a time; call set_enjoy_mode first"
        )

    def create_logs(self, state):
        if self._enjoy_mode is not None:
            return {
                self._env_names[self._enjoy_mode]: self._envs[
                    self._enjoy_mode
                ].create_logs(state)
            }
        return {
            name: env.create_logs(s)
            for name, env, s in zip(self._env_names, self._envs, state)
        }
