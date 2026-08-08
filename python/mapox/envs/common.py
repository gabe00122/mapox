from typing import Sequence

import jax
from jax import numpy as jnp

from mapox.specs import OBS_DTYPE, ObservationSpec

# Row i is the (dx, dy) delta for move action i, for envs that registered
# symbols.MOVES via add_block at local ids 0..3: up, right, down, left.
DIRECTIONS = jnp.array([[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=jnp.int32)


def make_obs_spec(width: int, height: int, num_types: int) -> ObservationSpec:
    # CHANNELS:
    # TILE (indexed by the env's obs vocab; the rest are structural)
    # DIRECTION
    # TEAM ID
    # HEALTH
    max_num_types = 1 << 16
    if num_types > max_num_types:
        raise ValueError(
            f"obs vocab too large for uint16 tiles: {num_types} > {max_num_types}"
        )

    return ObservationSpec(
        dtype=OBS_DTYPE,
        shape=(width, height, 4),
        max_value=(
            num_types,
            5,  # none, up, right, down, left,
            3,  # none, red, blue
            3,  # 0, 1, 2
        ),
    )


def make_action_mask(
    action_ids: Sequence[int], num_actions: int, num_agents: int
) -> jax.Array:
    mask = [False] * num_actions

    for action in action_ids:
        mask[action] = True

    mask_array = jnp.array(mask, jnp.bool_)
    return jnp.repeat(mask_array[None, :], num_agents, axis=0)
