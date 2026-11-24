"""The scripted Pac-Man in solutions/pacman/agent.py, against the real env.

The script carries its own copy of the game's rules, since it may only look
at the ASCII frames, so these tests are what notices the rust env and that
copy drifting apart.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from mapox.ascii import AsciiRenderer
from mapox.envs.rust_env_numpy import RustEnvNumpy, RustPacmanConfig

SCRIPT = Path(__file__).parent.parent / "solutions" / "pacman" / "agent.py"


@pytest.fixture(scope="module")
def pacman_agent():
    spec = importlib.util.spec_from_file_location("pacman_agent", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # dataclasses look their module up by name while the class body runs
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def play(pacman_agent, config, steps, seed=0, tuning=None, check=None):
    """Run the agent on frames alone; `check(agent, full_map)` runs each step."""
    env = RustEnvNumpy(config, steps)
    renderer = AsciiRenderer(env.obs_vocab, env.action_vocab)
    agent = pacman_agent.PacmanAgent(
        pacman_agent.Rules.from_config(config), tuning or pacman_agent.Tuning()
    )
    timestep = env.reset(seed)
    action = np.zeros(1, np.uint16)
    reward = 0.0
    for _ in range(steps):
        action[0] = env.action_vocab.id(agent.act(renderer.render(timestep.obs[0])))
        if check is not None:
            check(agent, renderer.render_grid(env.get_render_state().tilemap))
        timestep = env.step(action)
        reward += float(timestep.reward[0])
    return agent, reward


def test_the_scripts_maze_is_the_envs_maze(pacman_agent):
    env = RustEnvNumpy(RustPacmanConfig(), 8)
    renderer = AsciiRenderer(env.obs_vocab, env.action_vocab)
    env.reset(0)
    full_map = renderer.render_grid(env.get_render_state().tilemap)

    assert len(full_map) == pacman_agent.H
    glyphs = {"#": "#", ".": "o", "x": "@", " ": ".", "p": ">"}
    ghosts = dict(zip(pacman_agent.GHOST_MARKERS, pacman_agent.GHOST_GLYPHS))
    for theirs, ours in zip(full_map, pacman_agent.MAZE):
        assert theirs == "".join(glyphs.get(ch) or ghosts[ch] for ch in ours)


@pytest.mark.parametrize("randomize", [False, True])
def test_tracking_agrees_with_the_hidden_state(pacman_agent, randomize):
    """Everything the agent infers from its frames holds on the full map,
    which it never sees: where it is, which pellets are left, and that each
    ghost is somewhere it still thinks possible."""

    def check(agent, full_map):
        for row, line in enumerate(full_map):
            for x, ch in enumerate(line):
                cell = row * pacman_agent.W + x
                if ch in pacman_agent.PACMAN_GLYPHS:
                    assert agent.pos == cell
                elif ch in ".o@" and pacman_agent.PBIT[cell]:
                    assert bool(agent.pellets & pacman_agent.PBIT[cell]) == (ch != ".")
                elif ch in pacman_agent.GHOST_GLYPHS:
                    belief = agent.beliefs[pacman_agent.GHOST_GLYPHS.index(ch)]
                    assert any(
                        state[0] == cell and state[3] == pacman_agent.NORMAL
                        for state in belief
                    )

    config = RustPacmanConfig(randomize_starting_position=randomize)
    agent, _ = play(pacman_agent, config, steps=300, seed=3, check=check)
    assert agent.stats["reseeds"] == 0


def test_it_eats_nearly_every_step_under_the_default_rewards(pacman_agent):
    _, reward = play(pacman_agent, RustPacmanConfig(), steps=200)
    assert reward >= 190


def test_playing_to_clear_clears_the_maze(pacman_agent):
    tuning = pacman_agent.replace(pacman_agent.Tuning(), **pacman_agent.PLAY_TO_CLEAR)
    agent, reward = play(pacman_agent, RustPacmanConfig(), steps=700, tuning=tuning)
    # a second round only begins once the first maze is cleared or lost
    assert agent.stats["rounds"] >= 2
    assert reward >= pacman_agent.NPELLET
