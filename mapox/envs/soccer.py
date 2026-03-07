from functools import cached_property, partial
from typing import NamedTuple, Literal

import jax
from jax import numpy as jnp
from pydantic import BaseModel, ConfigDict
from mapox.environment import Environment
from mapox.map_generator import generate_decor_tiles
from mapox.specs import DiscreteActionSpec, ObservationSpec
from mapox.timestep import TimeStep
from mapox.renderer import GridRenderSettings, GridRenderState
import mapox.envs.constance as GW

# Movement directions including STAY (index 4 = [0, 0])
MOVE_DIRECTIONS = jnp.array([[0, 1], [1, 0], [0, -1], [-1, 0], [0, 0]], dtype=jnp.int32)


class SoccerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    env_type: Literal["soccer"] = "soccer"

    team_size: int = 3
    width: int = 30
    height: int = 20
    view_width: int = 11
    view_height: int = 11
    goal_width: int = 6
    goal_reward: float = 1.0


class SoccerState(NamedTuple):
    agents_pos: jax.Array  # (num_agents, 2)
    ball_pos: jax.Array  # (2,)
    time: jax.Array  # ()
    score: jax.Array  # (2,) cumulative [red_goals, blue_goals]
    rewards: jax.Array  # ()
    tiles: jax.Array  # (padded_w, padded_h) base map


class SoccerEnv(Environment[SoccerState]):
    def __init__(self, config: SoccerConfig, length: int) -> None:
        super().__init__()

        self._length = length
        self._config = config
        self._num_agents = config.team_size * 2

        self.width = config.width
        self.height = config.height

        self.view_width = config.view_width
        self.view_height = config.view_height
        self.pad_width = self.view_width // 2
        self.pad_height = self.view_height // 2

        self.padded_width = self.width + self.pad_width * 2
        self.padded_height = self.height + self.pad_height * 2

        # Goal zone x-coordinates (padded coords)
        center_x = self.pad_width + self.width // 2
        half_goal = config.goal_width // 2
        self._goal_x_min = center_x - half_goal
        self._goal_x_max = center_x - half_goal + config.goal_width - 1

        # Goal y-coordinates (padded coords)
        self._bottom_goal_y = self.pad_height
        self._top_goal_y = self.pad_height + self.height - 1

        # Teams: first half red (0), second half blue (1)
        team_size = config.team_size
        self._teams = jnp.concatenate(
            [
                jnp.zeros(team_size, dtype=jnp.int8),
                jnp.ones(team_size, dtype=jnp.int8),
            ]
        )

        self._action_mask = GW.make_action_mask(
            [GW.MOVE_UP, GW.MOVE_RIGHT, GW.MOVE_DOWN, GW.MOVE_LEFT, GW.STAY],
            self.num_agents,
        )

        self._start_positions = self._initial_positions()
        self._start_ball_pos = jnp.array(
            [self.pad_width + self.width // 2, self.pad_height + self.height // 2],
            dtype=jnp.int32,
        )

    def _initial_positions(self) -> jax.Array:
        team_size = self._config.team_size
        center_x = self.pad_width + self.width // 2

        # Red team on bottom quarter
        red_y = self.pad_height + self.height // 4
        red_xs = jnp.arange(team_size, dtype=jnp.int32) + center_x - team_size // 2
        red_pos = jnp.stack(
            [red_xs, jnp.full(team_size, red_y, dtype=jnp.int32)], axis=-1
        )

        # Blue team on top quarter
        blue_y = self.pad_height + 3 * self.height // 4
        blue_xs = jnp.arange(team_size, dtype=jnp.int32) + center_x - team_size // 2
        blue_pos = jnp.stack(
            [blue_xs, jnp.full(team_size, blue_y, dtype=jnp.int32)], axis=-1
        )

        return jnp.concatenate([red_pos, blue_pos], axis=0)

    def _pad_tiles(self, tiles, fill):
        return jnp.pad(
            tiles,
            pad_width=(
                (self.pad_width, self.pad_width),
                (self.pad_height, self.pad_height),
            ),
            mode="constant",
            constant_values=fill,
        )

    def _generate_tiles(self, rng_key):
        tiles = generate_decor_tiles(self.width, self.height, rng_key)
        tiles = self._pad_tiles(tiles, GW.TILE_WALL)

        # Place goal tiles in padded coords
        goal_xs = (
            jnp.arange(self._config.goal_width, dtype=jnp.int32) + self._goal_x_min
        )
        tiles = tiles.at[goal_xs, self._bottom_goal_y].set(GW.TILE_GOAL)
        tiles = tiles.at[goal_xs, self._top_goal_y].set(GW.TILE_GOAL)

        return tiles

    def reset(self, rng_key: jax.Array) -> tuple[SoccerState, TimeStep]:
        tiles = self._generate_tiles(rng_key)

        state = SoccerState(
            agents_pos=self._start_positions,
            ball_pos=self._start_ball_pos,
            time=jnp.int32(0),
            score=jnp.zeros(2, dtype=jnp.float32),
            rewards=jnp.float32(0.0),
            tiles=tiles,
        )

        actions = jnp.zeros((self.num_agents,), dtype=jnp.int32)
        rewards = jnp.zeros((self.num_agents,), dtype=jnp.float32)

        return state, self.encode_observations(state, actions, rewards)

    def step(
        self, state: SoccerState, action: jax.Array, rng_key: jax.Array
    ) -> tuple[SoccerState, TimeStep]:
        num_agents = self._num_agents
        tiles = state.tiles

        # Random execution order
        order = jax.random.permutation(rng_key, num_agents)

        def body(i, carry):
            positions, ball_pos = carry
            agent_idx = order[i]
            agent_action = action[agent_idx]

            direction = MOVE_DIRECTIONS[agent_action]
            proposed = positions[agent_idx] + direction

            # Wall check
            is_wall = tiles[proposed[0], proposed[1]] == GW.TILE_WALL

            # Agent collision (is another agent at proposed?)
            other_mask = jnp.arange(num_agents) != agent_idx
            at_proposed = jnp.all(positions == proposed, axis=-1)
            occupied_by_other = jnp.any(at_proposed & other_mask)

            # Ball check
            on_ball = jnp.all(proposed == ball_pos)

            # Blocked by wall, or by another agent when not pushing the ball
            blocked = is_wall | (occupied_by_other & ~on_ball)
            new_pos = jnp.where(blocked, positions[agent_idx], proposed)

            # Ball push: agent moved onto ball and is actually moving (not staying)
            hits_ball = jnp.all(new_pos == ball_pos) & (agent_action < 4)

            new_ball_pos = ball_pos + direction
            ball_wall = tiles[new_ball_pos[0], new_ball_pos[1]] == GW.TILE_WALL
            ball_occupied = jnp.any(
                jnp.all(positions == new_ball_pos, axis=-1) & other_mask
            )
            ball_blocked = ball_wall | ball_occupied

            # If ball blocked, agent can't push through - stays in place
            agent_blocked_by_ball = hits_ball & ball_blocked
            final_pos = jnp.where(agent_blocked_by_ball, positions[agent_idx], new_pos)
            final_ball = jnp.where(hits_ball & ~ball_blocked, new_ball_pos, ball_pos)

            positions = positions.at[agent_idx].set(final_pos)
            return positions, final_ball

        new_positions, new_ball_pos = jax.lax.fori_loop(
            0, num_agents, body, (state.agents_pos, state.ball_pos)
        )

        # Goal detection
        ball_in_goal_x = (new_ball_pos[0] >= self._goal_x_min) & (
            new_ball_pos[0] <= self._goal_x_max
        )
        ball_in_bottom_goal = ball_in_goal_x & (new_ball_pos[1] == self._bottom_goal_y)
        ball_in_top_goal = ball_in_goal_x & (new_ball_pos[1] == self._top_goal_y)

        scored = ball_in_bottom_goal | ball_in_top_goal

        # Red scores when ball enters top goal (blue's goal)
        # Blue scores when ball enters bottom goal (red's goal)
        goal_reward = self._config.goal_reward
        red_reward = (
            ball_in_top_goal.astype(jnp.float32)
            - ball_in_bottom_goal.astype(jnp.float32)
        ) * goal_reward
        blue_reward = -red_reward

        rewards = jnp.where(self._teams == 0, red_reward, blue_reward)

        # Update score
        new_score = state.score.at[0].add(ball_in_top_goal.astype(jnp.float32))
        new_score = new_score.at[1].add(ball_in_bottom_goal.astype(jnp.float32))

        # Reset positions on goal
        new_positions = jnp.where(scored, self._start_positions, new_positions)
        new_ball_pos = jnp.where(scored, self._start_ball_pos, new_ball_pos)

        state = SoccerState(
            agents_pos=new_positions,
            ball_pos=new_ball_pos,
            time=state.time + 1,
            score=new_score,
            rewards=state.rewards + jnp.mean(rewards),
            tiles=state.tiles,
        )

        return state, self.encode_observations(state, action, rewards)

    def _render_tiles(self, state: SoccerState):
        tiles = state.tiles

        # Render ball
        tiles = tiles.at[state.ball_pos[0], state.ball_pos[1]].set(GW.AGENT_BALL)

        # Render agents on top
        tiles = tiles.at[state.agents_pos[:, 0], state.agents_pos[:, 1]].set(
            GW.AGENT_GENERIC
        )

        directions = jnp.zeros_like(tiles, dtype=jnp.int8)

        teams = jnp.zeros_like(tiles, dtype=jnp.int8)
        teams = teams.at[state.agents_pos[:, 0], state.agents_pos[:, 1]].set(
            self.teams + 1
        )

        health = jnp.zeros_like(tiles, dtype=jnp.int8)

        return jnp.concatenate(
            (
                tiles[..., None],
                directions[..., None],
                teams[..., None],
                health[..., None],
            ),
            axis=-1,
        )

    def encode_observations(self, state: SoccerState, actions, rewards) -> TimeStep:
        @partial(jax.vmap, in_axes=(None, 0))
        def _encode_view(tiles, positions):
            return jax.lax.dynamic_slice(
                tiles,
                (
                    positions[0] - self.view_width // 2,
                    positions[1] - self.view_height // 2,
                    0,
                ),
                (self.view_width, self.view_height, self.observation_spec.shape[-1]),
            )

        tiles = self._render_tiles(state)
        view = _encode_view(tiles, state.agents_pos)

        time = jnp.repeat(state.time[None], self.num_agents, axis=0)

        return TimeStep(
            obs=view,
            time=time,
            last_action=actions,
            reward=rewards,
            action_mask=self._action_mask,
            terminated=jnp.equal(time, self._length - 1),
        )

    @cached_property
    def observation_spec(self) -> ObservationSpec:
        return GW.make_obs_spec(self.view_width, self.view_height)

    @cached_property
    def action_spec(self) -> DiscreteActionSpec:
        return DiscreteActionSpec(n=GW.NUM_ACTIONS)

    @property
    def is_jittable(self) -> bool:
        return True

    @property
    def num_agents(self) -> int:
        return self._num_agents

    @property
    def teams(self) -> jax.Array:
        return self._teams

    def create_placeholder_logs(self):
        return {"rewards": jnp.float32(0.0)}

    def create_logs(self, state: SoccerState):
        return {"rewards": state.rewards}

    def get_render_state(self, state: SoccerState) -> GridRenderState:
        tiles = self._render_tiles(state)

        return GridRenderState(
            tilemap=tiles,
            agent_positions=state.agents_pos,
        )

    def get_render_settings(self) -> GridRenderSettings:
        return GridRenderSettings(
            tile_width=self.width,
            tile_height=self.height,
            view_width=self.view_width,
            view_height=self.view_height,
        )
