import pytest
import jax

from mapox.envs.find_return import FindReturnConfig, FindReturnEnv
from mapox.envs.traveling_salesman import TravelingSalesmanConfig, TravelingSalesmanEnv
from mapox.envs.scouts import ScoutsConfig, ScoutsEnv
from mapox.envs.king_hill import KingHillConfig, KingHillEnv
from mapox.envs.prey import PreyConfig, PreyEnv

LENGTH = 32


def _make_envs():
    return [
        (
            "find_return",
            FindReturnEnv(FindReturnConfig(num_agents=2, num_flags=2), LENGTH),
        ),
        (
            "traveling_salesman",
            TravelingSalesmanEnv(
                TravelingSalesmanConfig(num_agents=2, num_flags=3), LENGTH
            ),
        ),
        (
            "scouts",
            ScoutsEnv(
                ScoutsConfig(num_scouts=1, num_harvesters=1, num_treasures=3), LENGTH
            ),
        ),
        ("king_hill", KingHillEnv(KingHillConfig(num_agents=4, num_flags=1), LENGTH)),
        (
            "prey",
            PreyEnv(
                PreyConfig(num_sneakers=2, num_chasers=1, num_food=3), LENGTH
            ),
        ),
    ]


@pytest.fixture(params=_make_envs(), ids=lambda p: p[0])
def env(request):
    """Yields each environment instance, parametrized across all 6 envs."""
    return request.param[1]


@pytest.fixture
def rng_key():
    return jax.random.key(42)
