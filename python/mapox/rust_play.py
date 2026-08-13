from mapox.envs.rust_env import RustEnv, RustFindReturnConfig
import jax
from mapox.agent import Agent, RandomAgent
from mapox._core import enjoy

import numpy as np

from mapox.timestep import TimeStep
import mapox


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


if __name__ == "__main__":
    env = RustEnv(RustFindReturnConfig())
    rng_agent = RandomAgent(env.action_spec)

    rust_enjoy(env, 512, 0, rng_agent)
