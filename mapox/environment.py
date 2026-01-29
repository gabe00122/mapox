from mapox.renderer import GridRenderState, GridRenderSettings
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, TypeVar, Generic

import jax
from jax import Array

from mapox.timestep import TimeStep
from mapox.specs import ObservationSpec, ActionSpec
import enum


class StepType(enum.IntEnum):
    FIRST = 0
    MID = 1
    LAST = 2

EnvState = TypeVar("EnvState")

class Environment(ABC, Generic[EnvState]):
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
    def is_jittable(self) -> bool: ...

    @property
    def teams(self) -> jax.Array | None:
        return None

    @abstractmethod
    def get_render_settings(self) -> GridRenderSettings:
        ...

    @abstractmethod
    def get_render_state(self, state: EnvState) -> GridRenderState:
        ...
