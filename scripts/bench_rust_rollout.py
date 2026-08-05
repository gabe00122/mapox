"""Rollout-fill throughput for the rust env behind RustEnv's io_callback bridge.

A jitted fori_loop fills a 128-step rollout with uniform-random actions,
mirroring scripts/bench_jax_rollout.py so the numbers are comparable.

Run: uv run python scripts/bench_rust_rollout.py [num_agents]
CPU: JAX_PLATFORMS=cpu uv run python scripts/bench_rust_rollout.py [num_agents]
"""

import math
import sys
import time

import jax
from jax import numpy as jnp

from mapox.envs.rust_env import RustEnv

ROLLOUT_LENGTH = 128
REPEATS = 50


def main() -> None:
    num_agents = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    # keep spawn density low as agent count grows (~8 tiles per agent floor)
    side = max(40, math.isqrt(num_agents * 8) + 1)
    env = RustEnv(
        f'{{"env_type": "find_return", "num_agents": {num_agents // 128},'
        f' "width": {side}, "height": {side}}}'
    )
    num_actions = env.action_spec.n

    _, timestep = jax.jit(env.reset)(jax.random.key(0))
    empty_rollout = jax.tree.map(
        lambda x: jnp.zeros((ROLLOUT_LENGTH, *x.shape), x.dtype), timestep
    )

    @jax.jit
    def fill_rollout(rng_key):
        def body(i, carry):
            rng_key, rollout = carry
            rng_key, action_key, step_key = jax.random.split(rng_key, 3)

            actions = jax.random.randint(
                action_key, (num_agents,), minval=0, maxval=num_actions
            )
            _, timestep = env.step(None, actions, step_key)
            rollout = jax.tree.map(lambda buf, x: buf.at[i].set(x), rollout, timestep)

            return rng_key, rollout

        return jax.lax.fori_loop(0, ROLLOUT_LENGTH, body, (rng_key, empty_rollout))

    # compile + warmup
    _, rollout = jax.tree.map(jax.block_until_ready, fill_rollout(jax.random.key(1)))

    start = time.perf_counter()
    for i in range(REPEATS):
        _, rollout = fill_rollout(jax.random.key(2 + i))
    jax.tree.map(jax.block_until_ready, rollout)
    elapsed = time.perf_counter() - start

    total_steps = ROLLOUT_LENGTH * REPEATS
    steps_per_sec = total_steps / elapsed
    print(
        f"jax {jax.default_backend()} (rust env, {num_agents} agents, {side}x{side}): "
        f"{steps_per_sec:.0f} steps/s, {steps_per_sec * num_agents:.3e} agent-steps/s "
        f"({elapsed / total_steps * 1e6:.0f} us/step)"
    )


if __name__ == "__main__":
    main()
