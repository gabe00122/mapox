import json
import tempfile
from pathlib import Path

from jax import numpy as jnp
import pytest

import mapox.symbols as SB
from mapox.map_loader import load_map
from mapox.vocab import Vocabulary

FIXTURE_PATH = str(Path(__file__).parent / "fixtures" / "test_map.json")


def _write_map(data) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(data, f)
    f.flush()
    return f.name


class TestLoadMap:
    def test_output_shape(self):
        tiles = load_map(FIXTURE_PATH, Vocabulary())

        assert tiles.shape == (10, 10)

    def test_tiles_use_local_ids(self):
        # The env registered symbols in its own order; the file's ints must
        # be translated through the legend into these local ids.
        vocab = Vocabulary()
        wall = vocab.add(SB.TILE_WALL)
        flag = vocab.add(SB.TILE_FLAG)
        tiles = load_map(FIXTURE_PATH, vocab)

        assert tiles[0, 3] == wall
        assert tiles[5, 5] == flag
        assert tiles[9, 7] == flag
        assert int(jnp.sum(tiles == flag)) == 2

    def test_legend_registers_unknown_symbols(self):
        vocab = Vocabulary()
        load_map(FIXTURE_PATH, vocab)

        assert SB.TILE_DECOR_1 in vocab
        assert SB.TILE_EMPTY in vocab

    def test_frozen_vocab_rejects_new_legend_symbols(self):
        vocab = Vocabulary([SB.TILE_EMPTY]).freeze()
        with pytest.raises(Exception):
            load_map(FIXTURE_PATH, vocab)

    def test_invalid_version(self):
        path = _write_map(
            {"version": 1, "width": 2, "height": 2, "tiles": [[0, 0], [0, 0]]}
        )
        with pytest.raises(ValueError, match="Unsupported map version"):
            load_map(path, Vocabulary())

    def test_tile_missing_from_legend(self):
        path = _write_map(
            {
                "version": 2,
                "width": 2,
                "height": 2,
                "legend": {"0": SB.TILE_EMPTY},
                "tiles": [[0, 0], [0, 5]],
            }
        )
        with pytest.raises(ValueError, match="legend"):
            load_map(path, Vocabulary())

    def test_shape_mismatch(self):
        path = _write_map(
            {
                "version": 2,
                "width": 3,
                "height": 2,
                "legend": {"0": SB.TILE_EMPTY},
                "tiles": [[0, 0], [0, 0]],
            }
        )
        with pytest.raises(ValueError, match="rows but width"):
            load_map(path, Vocabulary())
