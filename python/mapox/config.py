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
    RustMultiConfig,
    RustMultiEnvSpec,
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

    def _build_sub_env(self, env_config: Any, length: int, num: int) -> Environment:
        """One sub-env of a multi config: rust sub-envs carry their vector
        count in the config (the rust side steps the copies in parallel),
        JAX sub-envs are vectorized here."""
        if env_config.env_type.startswith("rust"):
            if num == 1:
                return RustEnv(env_config, length)
            return RustEnv(RustVecConfig(num=num, env=env_config), length)
        return self.create_env(env_config, length, num)[0]

    def create_env(
        self,
        env_config: Any,
        length: int,
        vec_count: int = 1,
        env_name: str | None = None,
    ) -> tuple[Environment, int]:
        num_tasks = 1

        if env_config.env_type.startswith("rust"):
            if vec_count > 1:
                raise ValueError(
                    "vector count for rust envs is set in the config "
                    "(env_type='vec'), not via create_env's vec_count"
                )
            return RustEnv(env_config, length), 1

        if env_config.env_type == "vec":
            if env_config.env.env_type.startswith("rust"):
                # the rust VectorWrapper steps the copies in parallel
                return (
                    RustEnv(RustVecConfig(num=env_config.num, env=env_config.env), length),
                    1,
                )
            if env_config.num == 1:
                return self.create_env(env_config.env, length)
            inner, _ = self.create_env(env_config.env, length)
            return VectorWrapper(inner, env_config.num), 1

        if env_config.env_type == "multi":
            num_tasks = len(env_config.envs)
            env_names = tuple(env_def.name for env_def in env_config.envs)

            if env_name is not None:
                if env_name not in env_names:
                    raise ValueError("Could not find environment matching env_name")
                task_id = env_names.index(env_name)

                # Every sub-env is built so the union vocab matches training;
                # only the selected one is kept, so the rest stay unvectorized.
                out_envs = tuple(
                    self._build_sub_env(
                        env_def.env, length, vec_count if i == task_id else 1
                    )
                    for i, env_def in enumerate(env_config.envs)
                )
                wrapper = MultiTaskWrapper(out_envs, env_names)

                return wrapper.task_envs[task_id], num_tasks

            if all(env_def.env.env_type.startswith("rust") for env_def in env_config.envs):
                # an all-rust batch runs entirely on the rust side, where the
                # MultitaskWrapper is the vectorizer
                rust_config = RustMultiConfig(
                    envs=tuple(
                        RustMultiEnvSpec(**env_def.model_dump())
                        for env_def in env_config.envs
                    )
                )
                return RustEnv(rust_config, length), num_tasks

            out_envs = tuple(
                self._build_sub_env(env_def.env, length, env_def.num)
                for env_def in env_config.envs
            )

            return MultiTaskWrapper(out_envs, env_names), num_tasks

        if env_config.env_type in self._registry:
            env = self._registry[env_config.env_type](env_config, length)
            if vec_count > 1:
                env = VectorWrapper(env, vec_count)

            return env, 1

        raise ValueError("Could not find env type matching that name")
