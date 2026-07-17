from typing import Iterable, Self, cast, Iterator

class FrozenVocabularyError(Exception): ...


class Vocabulary:
    def __init__(self, symbols: Iterable[str] = ()):
        self._symbols: list[str] = []
        self._ids: dict[str, int] = {}
        self._next_id = 0
        self._frozen = False

        self.extend(symbols)

    def add(self, symbol: str) -> int:
        id = self._ids.get(symbol)
        if id is None:
            id = self._next_id
            self._next_id += 1
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
        ids = [self.get(s) for s in symbols]
        nones = [id is None for id in ids]
        if all(nones):
            ids = [self.add(s) for s in symbols]
            return range(ids[0], ids[-1])
        elif any(nones):
            raise ValueError("Some ids did not exist")
        ids = cast(list[int], ids)

        id = ids[0]
        for next_id in ids[1:]:
            if id + 1 != next_id:
                raise ValueError("Vocab ids are not in a contiguous block")

            id = next_id

        return range(ids[0], ids[-1])


    def get(self, symbol: str) -> int | None:
        return self._ids.get(symbol)

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(*self._symbols)

    @property
    def frozen(self) -> bool:
        return self._frozen

    def freeze(self) -> Self:
        self._frozen = True
        return self

    def __len__(self) -> int:
        return len(self.symbols)

    def __contains__(self, symbol: str) -> bool:
        return self.get(symbol) is not None

    def __iter__(self) -> Iterator[str]:
        return iter(self.symbols)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Vocabulary):
            return self.symbols == other.symbols
        else:
            return False
