"""MAPOX: Multi-Agent Partially Observable gridworlds in JAX"""

__version__ = "0.1.0"

from mapox.timestep import TimeStep
from mapox.environment import Environment
from mapox.specs import ActionSpec, ObservationSpec
from mapox.wrappers.multitask import MultiTaskWrapper
from mapox.wrappers.vector import VectorWrapper
from mapox.client import GridworldClient
from mapox.utils.encode_one_hot import concat_one_hot
from mapox.config import create_env, create_client, ScoutsConfig, TravelingSalesmanConfig, KingHillConfig, MultiTaskConfig, EnvironmentConfig

__all__ = [
    "TimeStep",
    "ActionSpec",
    "ObservationSpec",
    "Environment",
    "MultiTaskWrapper",
    "VectorWrapper",
    "GridworldClient",
    "concat_one_hot",
    "create_env",
    "create_client",
    "ScoutsConfig",
    "TravelingSalesmanConfig",
    "KingHillConfig",
    "MultiTaskConfig",
    "EnvironmentConfig"
]
