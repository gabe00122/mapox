import json

import jax
from jax import numpy as jnp


def load_map(path: str) -> jax.Array:
    """Load a map from a JSON file and return unpadded tiles."""
    with open(path) as f:
        data = json.load(f)

    if data.get("version") != 1:
        raise ValueError(f"Unsupported map version: {data.get('version')}")

    tile_list = data["tiles"]
    width = data["width"]
    height = data["height"]

    if len(tile_list) != width:
        raise ValueError(
            f"tiles has {len(tile_list)} rows but width is {width}"
        )
    for i, row in enumerate(tile_list):
        if len(row) != height:
            raise ValueError(
                f"tiles[{i}] has {len(row)} columns but height is {height}"
            )

    return jnp.array(tile_list, dtype=jnp.int8)
