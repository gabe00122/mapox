from typing import NamedTuple
from jax.typing import DTypeLike


class ObservationSpec(NamedTuple):
    dtype: DTypeLike
    shape: tuple[int, ...]
    max_value: int | tuple[int, ...] | None = None


class DiscreteActionSpec(NamedTuple):
    n: int


ActionSpec = DiscreteActionSpec
