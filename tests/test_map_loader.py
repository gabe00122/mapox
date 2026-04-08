import json
import tempfile
from pathlib import Path

from jax import numpy as jnp
import pytest

from mapox.map_loader import load_map
import mapox.envs.constance as GW

FIXTURE_PATH = str(Path(__file__).parent / "fixtures" / "test_map.json")


class TestLoadMap:
    def test_output_shape(self):
        tiles = load_map(FIXTURE_PATH)

        assert tiles.shape == (10, 10)

    def test_empty_tiles_present(self):
        tiles = load_map(FIXTURE_PATH)

        assert int(jnp.sum(tiles == GW.TILE_EMPTY)) > 0

    def test_flags_preserved(self):
        tiles = load_map(FIXTURE_PATH)

        # test_map.json has 2 flags: at (5,5) and (9,7)
        assert tiles[5, 5] == GW.TILE_FLAG
        assert tiles[9, 7] == GW.TILE_FLAG
        assert int(jnp.sum(tiles == GW.TILE_FLAG)) == 2

    def test_invalid_version(self):
        data = {"version": 99, "width": 2, "height": 2, "tiles": [[0, 0], [0, 0]]}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            with pytest.raises(ValueError, match="Unsupported map version"):
                load_map(f.name)

    def test_shape_mismatch(self):
        data = {"version": 1, "width": 3, "height": 2, "tiles": [[0, 0], [0, 0]]}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            with pytest.raises(ValueError, match="rows but width"):
                load_map(f.name)
