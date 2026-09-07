from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator

from mapox.environment import Environment
from mapox.envs.find_return import FindReturnConfig, FindReturnEnv
from mapox.envs.king_hill import KingHillConfig, KingHillEnv
from mapox.envs.prey import PreyConfig, PreyEnv
from mapox.envs.rust_env import (
    RustEnv,
    RustFindReturnConfig,
    RustMultiConfig,
    RustScoutsConfig,
    RustSnakeConfig,
    RustVecConfig,
)
from mapox.envs.scouts import ScoutsConfig, ScoutsEnv
from mapox.envs.snake import SnakeConfig, SnakeEnv
from mapox.envs.traveling_salesman import (
    TravelingSalesmanConfig,
    TravelingSalesmanEnv,
)
from mapox.wrappers.multitask import MultiTaskWrapper
from mapox.wrappers.vector import VectorWrapper


class EnvironmentConfig(BaseModel):
    """Top-level environment config. The env type is resolved through the
    EnvironmentFactory registry at make time, where the config is validated
    against the model the env registered with (if any)."""
    model_config = ConfigDict(extra="allow", frozen=True)
    env_type: str


class VecConfig(BaseModel):
    """Vectorized copies of one env; rust envs are stepped in parallel,
    JAX envs are vmapped."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    env_type: Literal["vec"] = "vec"

    num: int = 1
    env: EnvironmentConfig


class MultiTaskEnvConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    num: int = 1
    name: str
    env: EnvironmentConfig


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
        self._registry: dict[
            str, tuple[Callable[[Any, int], Environment[Any]], type[BaseModel]]
        ] = {}
        self.register_env("find_return", FindReturnEnv, FindReturnConfig)
        self.register_env("scouts", ScoutsEnv, ScoutsConfig)
        self.register_env("traveling_salesman", TravelingSalesmanEnv, TravelingSalesmanConfig)
        self.register_env("king_hill", KingHillEnv, KingHillConfig)
        self.register_env("prey", PreyEnv, PreyConfig)
        self.register_env("snake", SnakeEnv, SnakeConfig)
        self.register_env("rust_find_return", RustEnv, RustFindReturnConfig)
        self.register_env("rust_scouts", RustEnv, RustScoutsConfig)
        self.register_env("rust_snake", RustEnv, RustSnakeConfig)
        self.register_env("rust_vec", RustEnv, RustVecConfig)
        self.register_env("rust_multi", RustEnv, RustMultiConfig)

    def register_env(
        self,
        name: str,
        fn: Callable[[Any, int], Environment[Any]],
        config_model: type[BaseModel],
    ):
        self._registry[name] = (fn, config_model)

    def create_env(
        self,
        env_config: EnvironmentConfig,
        length: int,
        env_name: str | None = None,
    ) -> Environment:
        if env_config.env_type == "vec":
            config = VecConfig.model_validate(env_config.model_dump())
            inner = self.create_env(config.env, length)
            return VectorWrapper(inner, config.num)

        if env_config.env_type == "multi":
            config = MultiTaskConfig.model_validate(env_config.model_dump())
            env_names = tuple(env_def.name for env_def in config.envs)

            if env_name is not None:
                if env_name not in env_names:
                    raise ValueError("Could not find environment matching env_name")
                task_id = env_names.index(env_name)

                # Every sub-env is built so the union vocab matches training;
                # only the selected one is kept, so the rest stay unvectorized.
                out_envs = tuple(
                    self.create_env(env_def.env, length) for env_def in config.envs
                )
                wrapper = MultiTaskWrapper(out_envs, env_names)

                return wrapper.task_envs[task_id]

            out_envs = tuple(
                self.create_env(env_def.env, length) for env_def in config.envs
            )

            return MultiTaskWrapper(out_envs, env_names)

        entry = self._registry.get(env_config.env_type)
        if entry is None:
            raise ValueError(f"Could not find env type matching that name: {env_config.env_type}")
        fn, config_model = entry
        validated_config = config_model.model_validate(env_config.model_dump())
        return fn(validated_config, length)
