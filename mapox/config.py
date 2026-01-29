from typing import Literal, Any, Callable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mapox.envs.king_hill import KingHillConfig, KingHillEnv
from mapox.envs.find_return import FindReturnConfig, FindReturnEnv
from mapox.envs.traveling_salesman import (
    TravelingSalesmanConfig,
    TravelingSalesmanEnv,
)
from mapox.envs.scouts import ScoutsConfig, ScoutsEnv

from mapox.environment import Environment, EnvState
from mapox.wrappers.task_id_wrapper import TaskIdWrapper
from mapox.wrappers.multitask import MultiTaskWrapper
from mapox.wrappers.vector import VectorWrapper

EnvironmentConfig = (
    FindReturnConfig
    | TravelingSalesmanConfig
    | ScoutsConfig
    | KingHillConfig
)


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
    _registry: dict[str, Callable[[Any, int], Environment[Any]]] = {}

    def __init__(self):
        self.register_env("find_return", FindReturnEnv)
        self.register_env("scouts", ScoutsEnv)
        self.register_env("traveling_salesman", TravelingSalesmanEnv)
        self.register_env("king_hill", KingHillEnv)

    def register_env(self, name: str, fn: Callable[[Any, int], Environment[Any]]):
        self._registry[name] = fn

    def create_env(
        self,
        env_config: Any,
        length: int,
        vec_count: int = 1,
        env_name: str | None = None,
    ) -> tuple[Environment, int]:
        num_tasks = 1

        if env_config.env_type == "multi":
            if env_name is not None:
                num_tasks = len(env_config.envs)
                for task_id, env_def in enumerate(env_config.envs):
                    if env_def.name == env_name:
                        return TaskIdWrapper(
                            self.create_env(env_def.env, length, vec_count=vec_count)[0], task_id
                        ), num_tasks
                raise ValueError("Could not find environment matching env_name")
            else:
                out_envs = []
                out_env_names = []
                num_tasks = len(env_config.envs)
                for env_def in env_config.envs:
                    out_envs.append(self.create_env(env_def.env, length, env_def.num)[0])
                    out_env_names.append(env_def.name)

                return  MultiTaskWrapper(tuple(out_envs), tuple(out_env_names)), num_tasks
        elif env_config.env_type in self._registry:
            env = self._registry[env_config.env_type](env_config, length)
            if vec_count > 1:
                env = VectorWrapper(env, vec_count)

            return env, 1

        raise ValueError("Could not find env type matching that name")
