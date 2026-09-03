from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mapox.environment import Environment
from mapox.envs.find_return import FindReturnConfig, FindReturnEnv
from mapox.envs.king_hill import KingHillConfig, KingHillEnv
from mapox.envs.prey import PreyConfig, PreyEnv
from mapox.envs.rust_env import (
    RustEnv,
    RustFindReturnConfig,
    RustScoutsConfig,
    RustSnakeConfig,
)
from mapox.envs.scouts import ScoutsConfig, ScoutsEnv
from mapox.envs.snake import SnakeConfig, SnakeEnv
from mapox.envs.traveling_salesman import (
    TravelingSalesmanConfig,
    TravelingSalesmanEnv,
)
from mapox.wrappers.multitask import MultiTaskWrapper
from mapox.wrappers.vector import VectorWrapper


class VecConfig(BaseModel):
    """Vectorized copies of one env; rust envs are stepped in parallel,
    JAX envs are vmapped."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    env_type: Literal["vec"] = "vec"

    num: int = 1
    env: EnvironmentConfig = Field(discriminator="env_type")


type EnvironmentConfig = (
    FindReturnConfig
    | TravelingSalesmanConfig
    | ScoutsConfig
    | KingHillConfig
    | PreyConfig
    | SnakeConfig
    | RustFindReturnConfig
    | RustScoutsConfig
    | RustSnakeConfig
    | VecConfig
)

VecConfig.model_rebuild()


class MultiTaskEnvConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    num: int = 1
    name: str
    env: EnvironmentConfig = Field(discriminator="env_type")


class MultiTaskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    env_type: Literal["multi"] = "multi"
    envs: tuple[MultiTaskEnvConfig, ...]

    @field_validator("envs", mode="before")
    @classmethod
    def coerce_envs(cls, v):
        # JSON gives list; accept list and turn into tuple
        return tuple(v) if isinstance(v, list) else v


class EnvironmentFactory:
    def __init__(self):
        self._registry: dict[str, Callable[[Any, int], Environment[Any]]] = {}
        self.register_env("find_return", FindReturnEnv)
        self.register_env("scouts", ScoutsEnv)
        self.register_env("traveling_salesman", TravelingSalesmanEnv)
        self.register_env("king_hill", KingHillEnv)
        self.register_env("prey", PreyEnv)
        self.register_env("snake", SnakeEnv)

    def register_env(self, name: str, fn: Callable[[Any, int], Environment[Any]]):
        self._registry[name] = fn

    def create_env(
        self,
        env_config: Any,
        length: int,
        env_name: str | None = None,
    ) -> Environment:
        if env_config.env_type.startswith("rust"):
            return RustEnv(env_config, length)

        if env_config.env_type == "vec":
            inner = self.create_env(env_config.env, length)
            return VectorWrapper(inner, env_config.num)

        if env_config.env_type == "multi":
            env_names = tuple(env_def.name for env_def in env_config.envs)

            if env_name is not None:
                if env_name not in env_names:
                    raise ValueError("Could not find environment matching env_name")
                task_id = env_names.index(env_name)

                # Every sub-env is built so the union vocab matches training;
                # only the selected one is kept, so the rest stay unvectorized.
                out_envs = tuple(
                    self.create_env(env_def.env, length)
                    for i, env_def in enumerate(env_config.envs)
                )
                wrapper = MultiTaskWrapper(out_envs, env_names)

                return wrapper.task_envs[task_id]

            out_envs = tuple(
                self.create_env(env_def.env, length)
                for env_def in env_config.envs
            )

            return MultiTaskWrapper(out_envs, env_names)

        if env_config.env_type in self._registry:
            env = self._registry[env_config.env_type](env_config, length)
            return env

        raise ValueError("Could not find env type matching that name")
