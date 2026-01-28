from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mapox.envs.king_hill import KingHillConfig, KingHillEnv
from mapox.envs.grid_return import ReturnDiggingConfig, ReturnDiggingEnv
from mapox.envs.traveling_salesman import (
    TravelingSalesmanConfig,
    TravelingSalesmanEnv,
)
from mapox.envs.scouts import ScoutsConfig, ScoutsEnv
from mapox.client import GridworldClient

from mapox.client import EnvironmentClient
from mapox.environment import Environment
from mapox.wrappers.task_id_wrapper import TaskIdWrapper
from mapox.wrappers.multitask import MultiTaskWrapper
from mapox.wrappers.vector import VectorWrapper

type EnvironmentConfig = (
    ReturnDiggingConfig
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


def create_env(
    env_config: EnvironmentConfig | MultiTaskConfig,
    length: int,
    vec_count: int = 1,
    env_name: str | None = None,
) -> tuple[Environment, int]:
    num_tasks = 1
    if env_config.env_type == "multi" and env_name is not None:
        num_tasks = len(env_config.envs)
        for task_id, env_def in enumerate(env_config.envs):
            if env_def.name == env_name:
                return TaskIdWrapper(
                    create_env(env_def.env, length, vec_count=vec_count)[0], task_id
                ), num_tasks
        raise ValueError("Could not find environment matching env_name")

    match env_config.env_type:
        case "multi":
            out_envs = []
            out_env_names = []
            num_tasks = len(env_config.envs)
            for env_def in env_config.envs:
                out_envs.append(create_env(env_def.env, length, env_def.num)[0])
                out_env_names.append(env_def.name)

            env = MultiTaskWrapper(tuple(out_envs), tuple(out_env_names))
        case "return_digging":
            env = ReturnDiggingEnv(env_config, length)
        case "scouts":
            env = ScoutsEnv(env_config, length)
        case "traveling_salesman":
            env = TravelingSalesmanEnv(env_config, length)
        case "king_hill":
            env = KingHillEnv(env_config, length)
        case _:
            raise ValueError(f"Unknown environment type: {env_config.env_type}")

    if vec_count > 1:
        env = VectorWrapper(env, vec_count)

    return env, num_tasks


def create_client[State](env: Environment[State]) -> EnvironmentClient[State]:
    # use the grid client for all environments for now
    return GridworldClient(env)
