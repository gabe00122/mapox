import jax.numpy as jnp
import numpy as np
import pytest
from mapox.ascii import LEGEND, UNKNOWN_CHAR, AsciiRenderer, build_char_table
from mapox.envs.rust_env_numpy import (
    RustEnvNumpy,
    RustFindReturnConfig,
    RustScoutsConfig,
    RustSnakeConfig,
)
from mapox.symbols import AGENT_GENERIC, TILE_EMPTY, TILE_WALL
from mapox.vocab import Vocabulary

RUST_CONFIGS = {
    "find_return": RustFindReturnConfig(num_agents=2, width=12, height=12),
    "scouts": RustScoutsConfig(num_scouts=1, num_harvesters=1, width=12, height=12),
    "snake": RustSnakeConfig(num_agents=2, width=12, height=12),
}


def renderer_for(obs_symbols=(), action_symbols=()):
    return AsciiRenderer(
        Vocabulary(obs_symbols).freeze(), Vocabulary(action_symbols).freeze()
    )


def test_render_grid_reads_top_down_and_maps_ids_through_the_vocab():
    renderer = renderer_for([TILE_WALL, TILE_EMPTY, AGENT_GENERIC])

    # 2 wide, 3 tall; y index 0 is the bottom map row, drawn last
    obs = np.array(
        [
            [0, 1, 2],
            [1, 0, 1],
        ],
        dtype=np.uint16,
    )

    assert renderer.render_grid(obs) == ["a.", ".#", "#."]


def test_full_channel_observation_uses_the_tile_channel():
    renderer = renderer_for([TILE_WALL, TILE_EMPTY])
    tiles = np.array([[0, 1], [1, 0]], dtype=np.uint16)

    full = np.zeros((2, 2, 4), dtype=np.uint16)
    full[..., 0] = tiles
    full[..., 1:] = 7  # direction/team/health are not drawn

    assert renderer.render_grid(full) == renderer.render_grid(tiles)


def test_jax_observations_are_accepted():
    renderer = renderer_for([TILE_WALL, TILE_EMPTY])
    obs = jnp.array([[0, 1], [1, 0]], dtype=jnp.uint16)

    assert renderer.render_grid(obs) == [".#", "#."]


def test_render_joins_the_grid_rows():
    renderer = renderer_for([TILE_WALL, TILE_EMPTY])
    obs = np.array([[0, 1], [1, 0]], dtype=np.uint16)

    assert renderer.render(obs) == "\n".join(renderer.render_grid(obs))


def test_mask_and_ui_have_distinct_glyphs_at_the_top():
    renderer = renderer_for(["mask", "ui", TILE_EMPTY])

    # one column, y running bottom to top: empty, ui, mask
    obs = np.array([[2, 1, 0]], dtype=np.uint16)

    # fog of war is blank; the UI band is a rule so the two do not look alike
    assert renderer.render_grid(obs) == [" ", "-", "."]


def test_unknown_symbols_and_out_of_range_ids():
    renderer = renderer_for(["tile/mystery"])

    assert renderer.render_grid(np.array([[0]], dtype=np.uint16)) == [UNKNOWN_CHAR]
    assert renderer.render_grid(np.array([[1]], dtype=np.uint16)) == [" "]


def test_char_table_is_indexed_by_observation_id():
    renderer = renderer_for([TILE_WALL, TILE_EMPTY])

    assert renderer.char_table == (LEGEND[TILE_WALL], LEGEND[TILE_EMPTY])
    assert renderer.obs_vocab.symbols == (TILE_WALL, TILE_EMPTY)


def test_build_char_table_falls_back_to_the_unknown_char():
    assert build_char_table(["tile/mystery"]) == [UNKNOWN_CHAR]


def test_available_actions_follows_the_mask_in_vocab_order():
    renderer = renderer_for(
        action_symbols=["move/up", "move/right", "move/down", "stay"]
    )

    mask = np.array([False, True, False, True])
    assert renderer.available_actions(mask) == ["move/right", "stay"]
    assert renderer.available_actions(jnp.zeros(4, dtype=jnp.bool_)) == []
    assert renderer.available_actions(np.zeros(4, dtype=np.uint8)) == []


def test_available_actions_rejects_a_mask_for_another_action_space():
    renderer = renderer_for(action_symbols=["move/up", "stay"])

    with pytest.raises(ValueError, match="action mask"):
        renderer.available_actions(np.array([True]))
    with pytest.raises(ValueError, match="action mask"):
        renderer.available_actions(np.ones((1, 2), dtype=bool))


def test_render_grid_rejects_a_batch_of_observations():
    renderer = renderer_for([TILE_EMPTY])

    with pytest.raises(ValueError, match="observation must be"):
        renderer.render_grid(np.zeros((2, 3, 3, 4), dtype=np.uint16))


def test_every_env_symbol_has_a_glyph(env, rng_key):
    renderer = AsciiRenderer(env.obs_vocab, env.action_vocab)
    _, timestep = env.reset(rng_key)

    width, height = env.observation_spec.shape[:2]
    rows = renderer.render_grid(np.asarray(timestep.obs)[0])

    assert len(rows) == height
    assert all(len(row) == width for row in rows)
    assert UNKNOWN_CHAR not in "".join(rows)


@pytest.mark.parametrize("config", RUST_CONFIGS.values(), ids=RUST_CONFIGS)
def test_rust_envs_render_and_include_the_ui_band(config):
    env = RustEnvNumpy(config, 64)
    timestep = env.reset(0)
    renderer = AsciiRenderer(env.obs_vocab, env.action_vocab)

    width, height = env.observation_spec.shape[:2]
    rows = renderer.render_grid(timestep.obs[0])

    assert len(rows) == height
    assert all(len(row) == width for row in rows)
    assert UNKNOWN_CHAR not in "".join(rows)

    # the rust envs append a two-row UI band at the top of the window
    assert rows[:2] == ["-" * width, "-" * width]
    # the field of view below it has something in it (the agent at least)
    assert any(char != " " for row in rows[2:] for char in row)


def test_rust_available_actions_match_the_mask():
    env = RustEnvNumpy(RustFindReturnConfig(num_agents=2, width=12, height=12), 64)
    timestep = env.reset(0)
    renderer = AsciiRenderer(env.obs_vocab, env.action_vocab)

    for agent in range(env.num_agents):
        mask = np.asarray(timestep.action_mask[agent])
        assert renderer.available_actions(mask) == [
            env.action_vocab.symbols[i] for i in np.flatnonzero(mask)
        ]
