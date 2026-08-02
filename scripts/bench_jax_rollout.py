"""Rollout-fill throughput for the pure-JAX FindReturnEnv under VectorWrapper.

Run: uv run python scripts/bench_jax_rollout.py
CPU: JAX_PLATFORMS=cpu uv run python scripts/bench_jax_rollout.py
"""

import time

import jax
from jax import numpy as jnp

from mapox.envs.find_return import FindReturnConfig, FindReturnEnv
from mapox.wrappers.vector import VectorWrapper

AGENTS_PER_ENV = 32
TOTAL_AGENTS = 4096
ROLLOUT_LENGTH = 128
REPEATS = 100


def main() -> None:
    vec_count = TOTAL_AGENTS // AGENTS_PER_ENV
    config = FindReturnConfig(num_agents=AGENTS_PER_ENV)
    env = VectorWrapper(FindReturnEnv(config, length=ROLLOUT_LENGTH), vec_count)

    num_actions = env.action_spec.n

    state, timestep = jax.jit(env.reset)(jax.random.key(0))
    empty_rollout = jax.tree.map(
        lambda x: jnp.zeros((ROLLOUT_LENGTH, *x.shape), x.dtype), timestep
    )

    @jax.jit
    def fill_rollout(state, rng_key):
        def body(i, carry):
            state, rng_key, rollout = carry
            rng_key, action_key, step_key = jax.random.split(rng_key, 3)

            actions = jax.random.randint(
                action_key, (TOTAL_AGENTS,), minval=0, maxval=num_actions
            )
            state, timestep = env.step(state, actions, step_key)
            rollout = jax.tree.map(
                lambda buf, x: buf.at[i].set(x), rollout, timestep
            )

            return state, rng_key, rollout

        return jax.lax.fori_loop(
            0, ROLLOUT_LENGTH, body, (state, rng_key, empty_rollout)
        )

    # compile + warmup
    state, _, rollout = jax.tree.map(
        jax.block_until_ready, fill_rollout(state, jax.random.key(1))
    )

    start = time.perf_counter()
    for i in range(REPEATS):
        state, _, rollout = fill_rollout(state, jax.random.key(2 + i))
    jax.tree.map(jax.block_until_ready, (state, rollout))
    elapsed = time.perf_counter() - start

    total_steps = ROLLOUT_LENGTH * REPEATS
    steps_per_sec = total_steps / elapsed
    print(
        f"jax {jax.default_backend()} ({vec_count} envs x {AGENTS_PER_ENV} agents): "
        f"{steps_per_sec:.0f} steps/s, {steps_per_sec * TOTAL_AGENTS:.3e} agent-steps/s "
        f"({total_steps} steps in {elapsed:.2f}s)"
    )


if __name__ == "__main__":
    main()
