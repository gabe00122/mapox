from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, TypeVar

import jax
from jax import Array

from mapox.renderer import GridRenderSettings, GridRenderState
from mapox.specs import ActionSpec, ObservationSpec
from mapox.timestep import TimeStep
from mapox.vocab import Vocabulary

EnvState = TypeVar("EnvState")


class Environment[EnvState](ABC):
    @abstractmethod
    def reset(self, rng_key: Array) -> tuple[EnvState, TimeStep]: ...

    @abstractmethod
    def step(
        self, state: EnvState, action: Array, rng_key: Array
    ) -> tuple[EnvState, TimeStep]: ...

    @abstractmethod
    def create_placeholder_logs(self) -> dict[str, Any]: ...

    @abstractmethod
    def create_logs(self, state) -> dict[str, Any]: ...

    @cached_property
    @abstractmethod
    def observation_spec(self) -> ObservationSpec: ...

    @cached_property
    @abstractmethod
    def action_spec(self) -> ActionSpec: ...

    @property
    @abstractmethod
    def num_agents(self) -> int: ...

    @property
    @abstractmethod
    def num_tasks(self) -> int: ...

    @property
    def teams(self) -> jax.Array | None:
        return None

    @abstractmethod
    def get_render_settings(self) -> GridRenderSettings: ...

    @abstractmethod
    def get_render_state(self, state: EnvState) -> GridRenderState: ...

    @property
    @abstractmethod
    def obs_vocab(self) -> Vocabulary: ...

    @property
    @abstractmethod
    def action_vocab(self) -> Vocabulary: ...
