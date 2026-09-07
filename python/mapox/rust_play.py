import argparse

import numpy as np

from mapox._core import enjoy
from mapox.agent import Agent, RandomAgent
from mapox.envs.rust_env import (
    RustEnv,
    RustFindReturnConfig,
    RustScoutsConfig,
    RustSnakeConfig,
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


def rust_enjoy(env: RustEnv, length: int, seed: int, agent: Agent):
    rust_agent = RustAgentWrapper(agent)
    enjoy(env.inner, length, seed, rust_agent)


CONFIGS = {
    "find_return": RustFindReturnConfig(
        view_width=15, view_height=15, width=80, height=70, mapgen_threshold=0.07
    ),
    "scouts": RustScoutsConfig(
        view_width=15,
        view_height=15,
        ui_height=2,
        width=80,
        height=70,
        mapgen_threshold=0.07,
    ),
    "snake": RustSnakeConfig(
        view_width=15,
        view_height=15,
        num_agents=4096,
        food_spawn_prob=0.001,
        width=500,
        height=500
    ),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play a rust MAPOX environment")
    parser.add_argument("--env", default="find_return", choices=list(CONFIGS))
    args = parser.parse_args()

    env = RustEnv(CONFIGS[args.env], 512)
    rng_agent = RandomAgent(env.action_spec)

    rust_enjoy(env, 512, 0, rng_agent)
