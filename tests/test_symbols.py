import re

import mapox.symbols as sym
from mapox.envs.common import DIRECTIONS
from mapox.vocab import Vocabulary

SYMBOL_PATTERN = re.compile(r"^[a-z0-9]+(_[a-z0-9]+)*(/[a-z0-9]+(_[a-z0-9]+)*)?$")


def _string_constants() -> dict[str, str]:
    return {
        name: value
        for name, value in vars(sym).items()
        if name.isupper() and isinstance(value, str)
    }


def test_symbols_are_unique():
    # A duplicate string would silently merge two concepts into one id.
    constants = _string_constants()
    assert len(set(constants.values())) == len(constants)


def test_symbols_are_well_formed():
    for name, value in _string_constants().items():
        assert SYMBOL_PATTERN.fullmatch(value), f"{name} = {value!r}"


def test_moves_order_matches_directions():
    assert sym.MOVES == (sym.MOVE_UP, sym.MOVE_RIGHT, sym.MOVE_DOWN, sym.MOVE_LEFT)
    assert len(sym.MOVES) == DIRECTIONS.shape[0]


def test_moves_block_registration():
    # The intended env idiom: moves land at local ids 0..3 so
    # DIRECTIONS[action] indexing stays valid.
    v = Vocabulary()
    assert v.add_block(sym.MOVES) == range(0, 4)
