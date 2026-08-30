import json

import jax
from jax import numpy as jnp

from mapox.vocab import Vocabulary


def load_map(path: str, vocab: Vocabulary) -> jax.Array:
    """Load a map from a JSON file and return unpadded tiles in local ids.

    The file's tile ints are private to the file; its legend maps them to
    symbol names, which are registered against `vocab` at load time. Call
    during env __init__, before the vocab is frozen, so legend symbols the
    env didn't register itself can still be added.
    """
    with open(path) as f:
        data = json.load(f)

    if data.get("version") != 2:
        raise ValueError(f"Unsupported map version: {data.get('version')}")

    legend = {
        int(file_id): vocab.add(symbol) for file_id, symbol in data["legend"].items()
    }

    tile_list = data["tiles"]
    width = data["width"]
    height = data["height"]

    if len(tile_list) != width:
        raise ValueError(f"tiles has {len(tile_list)} rows but width is {width}")
    for i, row in enumerate(tile_list):
        if len(row) != height:
            raise ValueError(
                f"tiles[{i}] has {len(row)} columns but height is {height}"
            )

    try:
        tiles = [[legend[t] for t in row] for row in tile_list]
    except KeyError as e:
        raise ValueError(f"tile id {e.args[0]} is not in the map's legend") from None

    return jnp.array(tiles, dtype=jnp.uint16)
