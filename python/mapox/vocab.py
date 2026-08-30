from collections.abc import Iterable, Iterator
from typing import Self, cast

import jax
from jax import numpy as jnp

from mapox.specs import VOCAB_DTYPE


class FrozenVocabularyError(Exception): ...


class Vocabulary:
    def __init__(self, symbols: Iterable[str] = ()):
        self._symbols: list[str] = []
        self._ids: dict[str, int] = {}
        self._frozen = False

        self.extend(symbols)

    def add(self, symbol: str) -> int:
        id = self._ids.get(symbol)
        if id is None:
            if self._frozen:
                raise FrozenVocabularyError()

            id = len(self._symbols)
            self._symbols.append(symbol)
            self._ids[symbol] = id

        return id

    def id(self, symbol: str) -> int:
        id = self._ids.get(symbol)
        if id is None:
            raise KeyError(f"Symbol '{symbol}' not found")
        return id

    def extend(self, symbols: Iterable[str]) -> None:
        for s in symbols:
            self.add(s)

    def add_block(self, symbols: Iterable[str]) -> range:
        symbols = list(symbols)
        ids = [self.get(s) for s in symbols]
        nones = [id is None for id in ids]

        if len(ids) == 0:
            raise ValueError("Add block should not be empty")

        symbol_set = set()
        for s in symbols:
            if s in symbol_set:
                raise ValueError("Duplicate symbol found")
            symbol_set.add(s)

        if all(nones):
            ids = [self.add(s) for s in symbols]
            return range(ids[0], ids[-1] + 1)
        if any(nones):
            raise ValueError("Some ids did not exist")
        ids = cast(list[int], ids)

        id = ids[0]
        for next_id in ids[1:]:
            if id + 1 != next_id:
                raise ValueError("Vocab ids are not in a contiguous block")

            id = next_id

        return range(ids[0], ids[-1] + 1)

    def get(self, symbol: str) -> int | None:
        return self._ids.get(symbol)

    def all_ids(self, symbols: Iterable[str]) -> list[int]:
        return [self.id(s) for s in symbols]

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(self._symbols)

    @property
    def frozen(self) -> bool:
        return self._frozen

    def freeze(self) -> Self:
        self._frozen = True
        return self

    def __len__(self) -> int:
        return len(self._symbols)

    def __contains__(self, symbol: str) -> bool:
        return self.get(symbol) is not None

    def __iter__(self) -> Iterator[str]:
        return iter(self._symbols)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Vocabulary):
            return self.symbols == other.symbols
        else:
            raise NotImplementedError()

    def __hash__(self) -> int:
        # jit caches are keyed on static-argument hashes, so only an immutable
        # (frozen) vocab may be hashed; equal frozen vocabs share cache entries.
        if not self._frozen:
            raise TypeError(
                "Vocabulary is unhashable until frozen; call freeze() before "
                "using it as a jit static argument"
            )
        return hash(self.symbols)

    def lut_to(
        self,
        target: Vocabulary,
        *,
        default: int | None = None,
        dtype: jnp.dtype = VOCAB_DTYPE,
    ) -> jax.Array:
        ids: list[int] = []
        for s in self._symbols:
            if default is None:
                ids.append(target.id(s))
            else:
                id = target.get(s)
                ids.append(id if id is not None else default)

        return jnp.array(ids, dtype=dtype)
