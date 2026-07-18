"""choose_positions — must only pick empty tiles, correct count, in bounds."""

import jax
from jax import numpy as jnp

from mapox.map_generator import choose_positions

# choose_positions takes the empty id explicitly; these are arbitrary local ids.
EMPTY = 0
WALL = 1


def _make_mixed_grid():
    """10x10 grid with walls on the border, empty inside."""
    tiles = jnp.full((10, 10), EMPTY, dtype=jnp.int8)
    tiles = tiles.at[0, :].set(WALL)
    tiles = tiles.at[-1, :].set(WALL)
    tiles = tiles.at[:, 0].set(WALL)
    tiles = tiles.at[:, -1].set(WALL)
    return tiles


def test_chooses_only_empty():
    tiles = _make_mixed_grid()
    key = jax.random.key(0)
    xs, ys = choose_positions(tiles, 5, EMPTY, key)

    for i in range(5):
        assert tiles[xs[i], ys[i]] == EMPTY


def test_non_zero_empty_id():
    # Guards the old bug where "empty" was hardcoded to id 0.
    tiles = jnp.full((6, 6), 3, dtype=jnp.int8).at[2:4, 2:4].set(7)
    key = jax.random.key(0)
    xs, ys = choose_positions(tiles, 4, 7, key)

    for i in range(4):
        assert tiles[xs[i], ys[i]] == 7


def test_correct_count():
    tiles = _make_mixed_grid()
    key = jax.random.key(0)
    xs, ys = choose_positions(tiles, 5, EMPTY, key)
    assert xs.shape == (5,)
    assert ys.shape == (5,)


def test_within_bounds():
    tiles = _make_mixed_grid()
    key = jax.random.key(0)
    xs, ys = choose_positions(tiles, 5, EMPTY, key)

    assert jnp.all(xs >= 0) and jnp.all(xs < 10)
    assert jnp.all(ys >= 0) and jnp.all(ys < 10)


def test_no_replacement_unique():
    tiles = _make_mixed_grid()
    key = jax.random.key(0)
    xs, ys = choose_positions(tiles, 5, EMPTY, key, replace=False)

    pairs = set()
    for i in range(5):
        pair = (int(xs[i]), int(ys[i]))
        assert pair not in pairs, f"Duplicate position: {pair}"
        pairs.add(pair)
