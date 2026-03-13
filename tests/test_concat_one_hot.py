"""concat_one_hot — core observation encoding utility."""

from jax import numpy as jnp

from mapox.utils.encode_one_hot import concat_one_hot


def test_basic():
    x = jnp.array([[0, 1]])  # (1, 2)
    result = concat_one_hot(x, (3, 4))
    expected = jnp.array([[1, 0, 0, 0, 1, 0, 0]], dtype=jnp.float32)
    assert jnp.array_equal(result, expected)


def test_offsets():
    x = jnp.array([[2, 0]])  # (1, 2)
    result = concat_one_hot(x, (4, 3))
    expected = jnp.array([[0, 0, 1, 0, 1, 0, 0]], dtype=jnp.float32)
    assert jnp.array_equal(result, expected)


def test_batched():
    x = jnp.array([[0, 1], [2, 0]])  # (2, 2)
    result = concat_one_hot(x, (4, 3))
    assert result.shape == (2, 7)
    # First row: index 0 in size-4, index 1 in size-3
    assert jnp.array_equal(result[0], jnp.array([1, 0, 0, 0, 0, 1, 0], dtype=jnp.float32))
    # Second row: index 2 in size-4, index 0 in size-3
    assert jnp.array_equal(result[1], jnp.array([0, 0, 1, 0, 1, 0, 0], dtype=jnp.float32))


def test_higher_rank():
    x = jnp.zeros((2, 3, 4), dtype=jnp.int32)
    result = concat_one_hot(x, (5, 5, 5, 5))
    assert result.shape == (2, 3, 20)
