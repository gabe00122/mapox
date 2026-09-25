from mapox.envs.rust_env_jax import RustEnvJax
import argparse
from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike

from mapox._core import enjoy
from mapox.agent import Agent, RandomAgent
from mapox.ascii import AsciiRenderer
from mapox.envs.rust_env_numpy import (
    RustEnvConfig,
    RustEnvNumpy,
    RustFindReturnConfig,
    RustScoutsConfig,
    RustSnakeConfig,
    RustVideoConfig,
)
from mapox.timestep import TimeStep


class RustAgentWrapper:
    def __init__(self, agent: Agent):
        self._agent = agent

    def act(self, obs, time, terminated, last_action, reward, action_mask):
        timestep = TimeStep(obs, time, terminated, last_action, reward, action_mask)
        actions = self._agent.act(timestep)

        return np.ascontiguousarray(actions, dtype=np.uint16)

    def reset(self, num_agents: int, seed: int):
        self._agent.reset(num_agents, seed)


def rust_enjoy(env: RustEnvNumpy, length: int, seed: int, agent: Agent):
    rust_agent = RustAgentWrapper(agent)
    enjoy(env.inner, length, seed, rust_agent)


def ascii_frame(
    renderer: AsciiRenderer,
    tiles: ArrayLike,
    action_mask: ArrayLike,
    *,
    title: str,
) -> str:
    """One frame: a status line, the ASCII grid, and the legal action names."""

    actions = renderer.available_actions(action_mask)
    legal = ", ".join(actions) if actions else "none"

    return "\n".join([title, *renderer.render_grid(tiles), f"actions: {legal}"])


def run_ascii(
    env: RustEnvJax | RustEnvNumpy,
    agent: Agent,
    length: int,
    seed: int,
    *,
    episodes: int = 1,
    focus: int = 0,
    full_map: bool = False,
    output: Callable[[str], None] = print,
) -> None:
    """Run episodes headlessly and print the ASCII view instead of a window.

    The agent drives every agent, exactly as the native viewer's random
    policy does; `focus` only chooses whose observation and action mask are
    printed. `full_map` prints the env's whole render-state tile map instead
    of the focused agent's crop. `output` receives one string per printed
    frame, so callers can capture the frames instead of writing to stdout.
    """

    if isinstance(env, RustEnvJax):
        env = env._env

    if not 0 <= focus < env.num_agents:
        raise ValueError(f"focused agent {focus} outside 0..{env.num_agents - 1}")

    renderer = AsciiRenderer(env.obs_vocab, env.action_vocab)
    for line in renderer.print_legend():
        output("Observation Legend:")
        output(line)

    for episode in range(episodes):
        episode_seed = seed + episode
        agent.reset(env.num_agents, episode_seed)
        timestep = env.reset(episode_seed)

        for step in range(length + 1):
            if full_map:
                tiles = env.get_render_state().tilemap
                title = f"episode {episode} step {step} full map"
            else:
                tiles = timestep.obs[focus]
                title = f"episode {episode} step {step} agent {focus}"

            output(
                ascii_frame(
                    renderer, tiles, timestep.action_mask[focus], title=title
                )
            )

            if step == length:
                break
            timestep = env.step(np.asarray(agent.act(timestep)))


CONFIGS = {
    "find_return": RustFindReturnConfig(),
    "scouts": RustScoutsConfig(),
    "snake": RustSnakeConfig(),
}


def wrap_video(config: RustEnvConfig, length: int, video_path: str) -> RustEnvConfig:
    return RustVideoConfig(
        env=config,
        output_dir=video_path,
        width=12 * 80,
        height=12 * 70,
        fps=10,
        record_steps=length,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Play a rust MAPOX environment")
    parser.add_argument("--env", default="find_return", choices=list(CONFIGS))
    parser.add_argument(
        "--ascii",
        action="store_true",
        help="run headless and print the ASCII view instead of the native window",
    )
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--agent", type=int, default=0, help="focused agent for the ASCII view"
    )
    parser.add_argument(
        "--every", type=int, default=1, help="print one frame every N steps"
    )
    parser.add_argument(
        "--map",
        action="store_true",
        help="print the full render-state map instead of the agent crop",
    )
    parser.add_argument(
        "--video-path",
        type=str,
        default=None
    )
    args = parser.parse_args()

    if args.ascii:
        env = RustEnvNumpy(CONFIGS[args.env], args.steps)
        agent = RandomAgent(env.action_spec)
        run_ascii(
            env,
            agent,
            args.steps,
            args.seed,
            episodes=args.episodes,
            focus=args.agent,
            full_map=args.map,
        )
        return

    config = CONFIGS[args.env]
    if args.video_path is not None:
        config = wrap_video(config, args.steps, args.video_path)
    env = RustEnvNumpy(config, args.steps)
    rng_agent = RandomAgent(env.action_spec)

    rust_enjoy(env, args.steps, args.seed, rng_agent)


if __name__ == "__main__":
    main()
