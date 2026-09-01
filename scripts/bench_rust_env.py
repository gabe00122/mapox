"""Direct throughput of the rust env through the pyo3 binding.

Steps mapox._core.Env in a plain Python loop with numpy buffers.
With --vec-count > 1 the config is wrapped in a `vec` env_type and the
mapox-core VectorWrapper steps the copies in parallel (crates/mapox-core/src/wrappers/vector.rs).

Run: uv run python scripts/bench_rust_env.py [--env {find_return,scouts,snake}]
                                      [--agents N] [--vec-count K]
"""

import argparse
import math
import time

import numpy as np
from mapox._core import Env
from mapox.envs.rust_env import (
    RustFindReturnConfig,
    RustScoutsConfig,
    RustSnakeConfig,
    RustVecConfig,
)

STEPS = 6400  # timed steps (128 x 50, as in the old JAX rollout bench)
WARMUP_STEPS = 128


def _human(x: float) -> str:
    for div, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if x >= div:
            return f"{x / div:.2f}{suffix}"
    return f"{x:,.0f}"


def make_config(name: str, agents_per_env: int, side: int):
    if name == "scouts":
        # keep the default 1:1 scout:harvester split
        num_scouts = agents_per_env // 2
        return RustScoutsConfig(
            num_scouts=num_scouts,
            num_harvesters=agents_per_env - num_scouts,
            width=side,
            height=side,
        )
    if name == "snake":
        return RustSnakeConfig(num_agents=agents_per_env, width=side, height=side)
    return RustFindReturnConfig(num_agents=agents_per_env, width=side, height=side)


def make_buffers(env):
    num_agents, view_width, view_height, channels = env.observation_shape
    return (
        np.zeros((num_agents, view_width, view_height, channels), np.uint16),
        np.zeros((num_agents,), np.int32),
        np.zeros((num_agents,), np.bool_),
        np.zeros((num_agents,), np.uint16),
        np.zeros((num_agents,), np.float32),
        np.zeros((num_agents, env.num_actions), np.bool_),
        np.zeros((num_agents,), np.int32),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env",
        choices=["find_return", "scouts", "snake"],
        default="find_return",
        help="which rust env to bench",
    )
    parser.add_argument(
        "--agents",
        type=int,
        default=4096,
        help="total number of agents across all vectorized envs",
    )
    parser.add_argument(
        "--vec-count",
        type=int,
        default=16,
        help="number of vectorized envs (Rust-side VectorWrapper)",
    )
    args = parser.parse_args()

    if args.agents < 1 or args.vec_count < 1:
        parser.error("--agents and --vec-count must be positive")
    if args.agents % args.vec_count != 0:
        parser.error(
            f"--agents ({args.agents}) must be divisible by --vec-count ({args.vec_count})"
        )

    agents_per_env = args.agents // args.vec_count
    # keep spawn density low as agent count grows (~8 tiles per agent floor)
    side = max(40, math.isqrt(args.agents * 8) + 1)

    config = make_config(args.env, agents_per_env, side)
    env = Env(RustVecConfig(num=args.vec_count, env=config).model_dump_json(), STEPS)
    assert env.num_agents == args.agents, (env.num_agents, args.agents)
    obs, step_time, terminated, last_action, reward, action_mask, task_ids = (
        make_buffers(env)
    )

    num_agents = env.num_agents
    num_actions = env.num_actions
    rng = np.random.default_rng(0)

    env.reset(0, obs, step_time, terminated, last_action, reward, action_mask, task_ids)

    def step_once():
        actions = rng.integers(0, num_actions, size=num_agents, dtype=np.uint16)
        env.step(
            actions,
            obs,
            step_time,
            terminated,
            last_action,
            reward,
            action_mask,
            task_ids,
        )

    for _ in range(WARMUP_STEPS):
        step_once()

    start = time.perf_counter()
    for _ in range(STEPS):
        step_once()
    elapsed = time.perf_counter() - start

    steps_per_sec = STEPS / elapsed
    print(
        f"rust {args.env} ({args.vec_count} envs x {agents_per_env} agents = "
        f"{num_agents}, {side}x{side}, {STEPS} steps, numpy+python loop):"
    )
    print(f"  env steps:   {_human(steps_per_sec)} steps/s")
    print(f"  agent steps: {_human(steps_per_sec * num_agents)} steps/s")
    print(f"  per agent:   {steps_per_sec / num_agents:.2f} steps/s")
    print(f"  latency:     {elapsed / STEPS * 1e6:,.0f} us/step")


if __name__ == "__main__":
    main()
