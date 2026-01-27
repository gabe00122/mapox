"""MAPOX: Multi-Agent Partially Observable gridworlds in JAX"""

__version__ = "0.1.0"

from mapox.envs.scouts import ScoutsEnv, ScoutsConfig
from mapox.envs.king_hill import KingHillEnv, KingHillConfig
from mapox.envs.grid_return import ReturnDiggingEnv, ReturnDiggingConfig
from mapox.envs.traveling_salesman import TravelingSalesmanEnv, TravelingSalesmanConfig
from mapox.envs.renderer import GridworldRenderer, GridworldClient

__all__ = [
    "ScoutsEnv",
    "ScoutsConfig",
    "KingHillEnv",
    "KingHillConfig",
    "ReturnDiggingEnv",
    "ReturnDiggingConfig",
    "TravelingSalesmanEnv",
    "TravelingSalesmanConfig",
    "GridworldRenderer",
    "GridworldClient",
]
