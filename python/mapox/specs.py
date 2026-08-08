from typing import NamedTuple

from jax import numpy as jnp
from jax.typing import DTypeLike

VOCAB_DTYPE = jnp.uint16
OBS_DTYPE = VOCAB_DTYPE
ACTION_DTYPE = VOCAB_DTYPE


class ObservationSpec(NamedTuple):
    dtype: DTypeLike
    shape: tuple[int, ...]
    max_value: int | tuple[int, ...] | None = None


class DiscreteActionSpec(NamedTuple):
    n: int
    dtype: DTypeLike = ACTION_DTYPE


ActionSpec = DiscreteActionSpec
