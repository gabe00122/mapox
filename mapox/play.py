import argparse
from mapox.envs.king_hill import KingHillConfig
from mapox.envs.traveling_salesman import TravelingSalesmanConfig
from mapox.envs.scouts import ScoutsConfig
from functools import partial

from mapox.environment import EnvState, Environment
from mapox.agent import Agent, AgentState, RandomAgent
from mapox.timestep import TimeStep
from mapox.config import EnvironmentFactory, FindReturnConfig
from mapox.client import GridworldClient
import mapox.envs.constance as GW


import pygame
import jax
from einops import rearrange


def get_action_from_keydown(event: pygame.event.Event | None):
    if event is None or event.type != pygame.KEYDOWN:
        return None

    key = event.key
    if key in (pygame.K_w, pygame.K_UP):
        return GW.MOVE_UP
    elif key in (pygame.K_s, pygame.K_DOWN):
        return GW.MOVE_DOWN
    elif key in (pygame.K_a, pygame.K_LEFT):
        return GW.MOVE_LEFT
    elif key in (pygame.K_d, pygame.K_RIGHT):
        return GW.MOVE_RIGHT
    elif key == pygame.K_PERIOD:
        return GW.STAY
    elif key == pygame.K_SPACE:
        return GW.PRIMARY_ACTION
    elif key == pygame.K_e:
        return GW.DIG_ACTION

    return None


def add_seq_dim(ts: TimeStep):
    return jax.tree.map(lambda x: rearrange(x, "b ... -> b 1 ..."), ts)


@partial(jax.jit, static_argnums=(0,), donate_argnums=(1, 3))
def step(env: Environment[EnvState], env_state: EnvState, actions: jax.Array, rng_key: jax.Array) -> tuple[EnvState, TimeStep, jax.Array]:
    env_key, rng_key = jax.random.split(rng_key)

    env_state, timestep = env.step(env_state, actions, env_key)
    return env_state, timestep, rng_key

def play(
    env: Environment[EnvState],
    agent: Agent[AgentState],
    agent_state: AgentState,
    rng_key: jax.Array = jax.random.PRNGKey(42),
    video_path: str | None = None,
    size: int = 960,
    human_control: bool = True,
    pov: bool = False,
) -> None:
    focused_agent = 0

    client = GridworldClient(env, fps=30, screen_width=size, screen_height=size)
    client.focus_agent(focused_agent)

    env_key, rng_key = jax.random.split(rng_key)
    env_state, timestep = env.reset(env_key)
    client.render(env_state, timestep, pov)

    cumulative_reward = 0.0

    step_count = 0
    while not timestep.terminated[..., 0].item():
        human_action = None

        for event in pygame.event.get():
            if client.handle_event(event):
                continue
            if event.type == pygame.QUIT:
                break

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_n:
                    focused_agent += 1
                    focused_agent %= env.num_agents
                    client.focus_agent(focused_agent)
                    continue

                human_action = get_action_from_keydown(event)

        if human_action is not None or not human_control:
            agent_state, actions, rng_key = agent.sample_actions(agent_state, timestep, rng_key)

            if human_control:
                actions = actions.at[focused_agent].set(human_action)

            env_state, timestep, rng_key = step(env, env_state, actions, rng_key)

            if video_path is not None:
                client.record_frame()
            step_count += 1
            reward = timestep.reward[focused_agent].item()
            cumulative_reward += reward
            print(f"reward: {reward}")
            client.render(env_state, timestep, pov)

    print(f"Cumulative reward: {cumulative_reward}")

    if video_path is not None:
        client.save_video(video_path)


def main():
    parser = argparse.ArgumentParser(description="Play a MAPOX environment")
    parser.add_argument(
        "--env",
        default="find_return",
        choices=["find_return", "scouts", "traveling_salesman", "king_hill"],
        help="Which environment to run",
    )
    args = parser.parse_args()

    env_factory = EnvironmentFactory()

    config_cls = {
        "find_return": FindReturnConfig,
        "scouts": ScoutsConfig,
        "traveling_salesman": TravelingSalesmanConfig,
        "king_hill": KingHillConfig,
    }[args.env]

    config = config_cls()
    env, _ = env_factory.create_env(config, 512)

    agent = RandomAgent(env.action_spec)
    agent_state = None

    play(env, agent, agent_state, pov=True)

if __name__ == "__main__":
    main()
