"""MAPOX: Multi-Agent Partially Observable gridworlds in JAX"""

from importlib.metadata import version

__version__ = version("mapox")

from mapox.client import GridworldClient
from mapox.config import (
    EnvironmentConfig,
    EnvironmentFactory,
    FindReturnConfig,
    KingHillConfig,
    MultiTaskConfig,
    PreyConfig,
    ScoutsConfig,
    SnakeConfig,
    TravelingSalesmanConfig,
    VecConfig,
)
from mapox.environment import Environment
from mapox.specs import ActionSpec, ObservationSpec
from mapox.timestep import TimeStep
from mapox.utils.encode_one_hot import concat_one_hot
from mapox.wrappers.multitask import MultiTaskWrapper
from mapox.wrappers.vector import VectorWrapper
from mapox.wrappers.vocab_wrapper import VocabWrapper

__all__ = [
    "ActionSpec",
    "Environment",
    "EnvironmentConfig",
    "EnvironmentFactory",
    "FindReturnConfig",
    "GridworldClient",
    "KingHillConfig",
    "MultiTaskConfig",
    "MultiTaskWrapper",
    "ObservationSpec",
    "PreyConfig",
    "ScoutsConfig",
    "SnakeConfig",
    "TimeStep",
    "TravelingSalesmanConfig",
    "VecConfig",
    "VectorWrapper",
    "VocabWrapper",
    "concat_one_hot",
]
