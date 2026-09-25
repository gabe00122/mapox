"""The rust env's full-map render state, exposed through the Python bridge."""

import jax
import numpy as np
import pytest
from mapox.ascii import UNKNOWN_CHAR, AsciiRenderer
from mapox.envs.rust_env_jax import RustEnvJax
from mapox.envs.rust_env_numpy import (
    RustEnvNumpy,
    RustFindReturnConfig,
    RustMultiConfig,
    RustMultiEnvSpec,
    RustScoutsConfig,
    RustSnakeConfig,
)

# (config, (view_width, view_height, ui_height)): every env adds its two-row
# UI band to the configured view_height.
CONFIGS = {
    "find_return": (
        RustFindReturnConfig(
            num_agents=2, width=12, height=10, view_width=9, view_height=7
        ),
        (9, 9, 2),
    ),
    "scouts": (
        RustScoutsConfig(
            num_scouts=1,
            num_harvesters=1,
            width=12,
            height=10,
            view_width=9,
            view_height=7,
        ),
        (9, 9, 2),
    ),
    "snake": (
        RustSnakeConfig(num_agents=1, width=12, height=10, view_width=9, view_height=7),
        (9, 9, 2),
    ),
}


@pytest.mark.parametrize("config, view", CONFIGS.values(), ids=CONFIGS)
def test_render_settings_describe_the_map_and_window(config, view):
    env = RustEnvNumpy(config, 64)

    settings = env.get_render_settings()
    assert settings.tile_width == config.width
    assert settings.tile_height == config.height
    assert (settings.view_width, settings.view_height, settings.ui_height) == view
    assert settings.obs_vocab.symbols == env.obs_vocab.symbols


@pytest.mark.parametrize("config, view", CONFIGS.values(), ids=CONFIGS)
def test_render_state_is_the_full_map_with_agents_placed(config, view):
    env = RustEnvNumpy(config, 64)
    env.reset(0)

    settings = env.get_render_settings()
    state = env.get_render_state()

    assert state.tilemap.shape == (settings.tile_width, settings.tile_height)
    assert state.agent_positions.shape == (env.num_agents, 2)
    assert np.all(state.agent_positions >= 0)
    assert np.all(state.agent_positions[:, 0] < settings.tile_width)
    assert np.all(state.agent_positions[:, 1] < settings.tile_height)

    symbols = env.obs_vocab.symbols
    assert state.tilemap.max() < len(symbols)
    for x, y in state.agent_positions:
        assert symbols[state.tilemap[x, y]].startswith("agent/")


def test_render_state_tracks_agent_movement():
    env = RustEnvNumpy(RustSnakeConfig(num_agents=1, width=12, height=10), 64)
    env.reset(0)
    before = env.get_render_state().agent_positions.copy()

    legal = np.flatnonzero(env.buffers.action_mask[0])
    assert len(legal) > 0
    env.step(np.array([legal[0]], dtype=np.uint16))

    after = env.get_render_state().agent_positions
    assert not np.array_equal(before, after)


def test_ascii_renderer_draws_the_full_render_state():
    env = RustEnvNumpy(RustFindReturnConfig(num_agents=2, width=12, height=10), 64)
    env.reset(0)

    settings = env.get_render_settings()
    renderer = AsciiRenderer(env.obs_vocab, env.action_vocab)
    rows = renderer.render_grid(env.get_render_state().tilemap)

    assert len(rows) == settings.tile_height
    assert all(len(row) == settings.tile_width for row in rows)
    assert UNKNOWN_CHAR not in "".join(rows)
    # the full map has no UI band; that only exists in the observation window
    assert "-" not in "".join(rows)


def test_jax_wrapper_exposes_the_same_render_state():
    env = RustEnvJax(RustFindReturnConfig(num_agents=1, width=12, height=10), 64)
    env.reset(jax.random.key(0))

    settings = env.get_render_settings()
    state = env.get_render_state(None)

    assert state.tilemap.shape == (settings.tile_width, settings.tile_height)
    assert state.agent_positions.shape == (env.num_agents, 2)


def test_multitask_render_state_ids_index_the_union_vocab():
    env = RustEnvNumpy(
        RustMultiConfig(
            envs=(
                RustMultiEnvSpec(
                    name="scouts",
                    num=1,
                    env=RustScoutsConfig(
                        num_scouts=1, num_harvesters=1, width=12, height=10
                    ),
                ),
                RustMultiEnvSpec(
                    name="fr",
                    num=1,
                    env=RustFindReturnConfig(num_agents=1, width=12, height=10),
                ),
            )
        ),
        64,
    )
    env.reset(0)

    settings = env.get_render_settings()
    state = env.get_render_state()

    assert settings.obs_vocab.symbols == env.obs_vocab.symbols
    assert state.tilemap.max() < len(settings.obs_vocab.symbols)
    # the wrapper renders the selected task (its first envs entry when no
    # enjoy mode is set), not every task's agents at once
    assert state.agent_positions.shape == (2, 2)
