"""Rendering smoke — every env must be drawable through the symbol-keyed
sprite table.

Two layers: a deterministic check that each registered obs symbol has a
sprite (catches typos and missing art even for tiles absent from a sampled
frame), and a real headless frame render (catches everything else: settings
wiring, cache build, channel-variant indexing).
"""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pytest

from mapox.renderer import GridworldRenderer, tilemap


@pytest.fixture(scope="module")
def renderer():
    return GridworldRenderer(screen_width=320, screen_height=320, fps=1000)


def test_every_obs_symbol_has_a_sprite(env):
    missing = set(env.obs_vocab.symbols) - set(tilemap.keys())
    assert not missing, f"no sprite registered for {sorted(missing)}"


def test_renders_a_frame(env, renderer, rng_key):
    state, _ = env.reset(rng_key)
    renderer.set_env(env.get_render_settings())
    renderer.render(env.get_render_state(state))
    renderer.render_agent_view(env.get_render_state(state))
