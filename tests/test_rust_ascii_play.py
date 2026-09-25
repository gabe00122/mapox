"""Real headless episodes through the rust play script's ASCII view."""

import numpy as np
import pytest
from mapox.agent import RandomAgent
from mapox.ascii import UNKNOWN_CHAR, AsciiRenderer
from mapox.envs.rust_env_numpy import (
    RustEnvNumpy,
    RustFindReturnConfig,
    RustScoutsConfig,
    RustSnakeConfig,
)
from mapox.rust_play import ascii_frame, run_ascii

CONFIGS = {
    "find_return": RustFindReturnConfig(
        num_agents=2, width=14, height=12, view_width=9, view_height=7
    ),
    "scouts": RustScoutsConfig(
        num_scouts=1,
        num_harvesters=1,
        width=14,
        height=12,
        view_width=9,
        view_height=7,
    ),
    "snake": RustSnakeConfig(
        num_agents=2, width=14, height=12, view_width=9, view_height=7
    ),
}

EPISODES = 2
STEPS = 16


@pytest.fixture(params=CONFIGS.values(), ids=list(CONFIGS))
def env(request):
    return RustEnvNumpy(request.param, STEPS)


def episode_frames(env, outputs):
    """The episode frames in `outputs`, after run_ascii's observation legend."""

    return outputs[2 * len(env.obs_vocab.symbols) :]


def test_real_episodes_render_ascii_frames(env):
    outputs = []
    run_ascii(
        env,
        RandomAgent(env.action_spec),
        STEPS,
        seed=0,
        episodes=EPISODES,
        output=outputs.append,
    )
    frames = episode_frames(env, outputs)

    # the reset frame plus one per step, for every episode
    assert len(frames) == EPISODES * (STEPS + 1)

    width, height = env.observation_spec.shape[:2]
    ui_height = env.get_render_settings().ui_height

    for frame in frames:
        lines = frame.splitlines()
        assert lines[0].startswith("episode")

        grid = lines[1 : 1 + height]
        assert len(grid) == height
        assert all(len(row) == width for row in grid)
        assert UNKNOWN_CHAR not in "\n".join(grid)
        # the UI band is a rule, never confused with fog of war
        assert grid[:ui_height] == ["-" * width] * ui_height

        actions = lines[1 + height].removeprefix("actions: ").split(", ")
        assert actions and all(action in env.action_vocab for action in actions)


def test_full_map_frames_use_the_render_state(env):
    outputs = []
    run_ascii(
        env,
        RandomAgent(env.action_spec),
        STEPS,
        seed=1,
        full_map=True,
        output=outputs.append,
    )
    frames = episode_frames(env, outputs)

    settings = env.get_render_settings()
    assert len(frames) == STEPS + 1

    for frame in frames:
        lines = frame.splitlines()
        assert lines[0].endswith("full map")

        grid = lines[1 : 1 + settings.tile_height]
        assert len(grid) == settings.tile_height
        assert all(len(row) == settings.tile_width for row in grid)
        assert UNKNOWN_CHAR not in "\n".join(grid)
        # the full map has no UI band; that only exists in the observation
        assert "-" not in "\n".join(grid)


def test_focus_selects_the_agent_view_and_ids_are_checked(env):
    outputs = []
    run_ascii(
        env,
        RandomAgent(env.action_spec),
        STEPS,
        seed=0,
        focus=1,
        output=outputs.append,
    )
    frames = episode_frames(env, outputs)
    assert frames[0].splitlines()[0] == "episode 0 step 0 agent 1"

    with pytest.raises(ValueError, match="focused agent"):
        run_ascii(
            env,
            RandomAgent(env.action_spec),
            STEPS,
            seed=0,
            focus=env.num_agents,
            output=lambda _: None,
        )


def test_observation_legend_is_printed_before_the_frames(env):
    outputs = []
    run_ascii(env, RandomAgent(env.action_spec), STEPS, seed=0, output=outputs.append)

    legend = outputs[: 2 * len(env.obs_vocab.symbols)]
    assert legend[::2] == ["Observation Legend:"] * len(env.obs_vocab.symbols)
    assert all(
        entry.endswith(f" = {symbol}")
        for entry, symbol in zip(legend[1::2], env.obs_vocab.symbols)
    )
    assert outputs[2 * len(env.obs_vocab.symbols)].splitlines()[0] == (
        "episode 0 step 0 agent 0"
    )


def test_ascii_frame_marks_the_legal_actions():
    env = RustEnvNumpy(RustFindReturnConfig(num_agents=1, width=14, height=12), 8)
    env.reset(0)

    renderer = AsciiRenderer(env.obs_vocab, env.action_vocab)
    mask = env.buffers.action_mask[0]

    frame = ascii_frame(renderer, env.buffers.obs[0], mask, title="t")
    lines = frame.splitlines()

    assert lines[0] == "t"
    assert lines[-1].startswith("actions: ")
    actions = lines[-1].removeprefix("actions: ").split(", ")
    assert actions == [env.action_vocab.symbols[i] for i in np.flatnonzero(mask)]
