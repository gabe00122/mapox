from functools import cached_property, partial
from typing import Literal, NamedTuple

import jax
from jax import numpy as jnp
from pydantic import BaseModel, ConfigDict

import mapox.envs.constance as GW
from mapox.environment import Environment
from mapox.renderer import GridRenderSettings, GridRenderState
from mapox.specs import DiscreteActionSpec, ObservationSpec
from mapox.timestep import TimeStep


class EmbodiedCommConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    env_type: Literal["embodied_comm"] = "embodied_comm"

    grid_size: int = 2
    num_colors: int = 8
    view_width: int = 9
    view_height: int = 9
    timeout_penalty: float = 0.0
    win_reward: float = 1.0


class EmbodiedCommState(NamedTuple):
    agents_pos: jax.Array  # (2, 2) int32, padded coords
    committed: jax.Array  # (2,) bool
    map: jax.Array  # (W_padded, H_padded) int8; colors baked in as TILE_COLOR_*
    time: jax.Array  # () int32
    rewards: jax.Array  # () float32, cumulative episode reward


class EmbodiedCommEnv(Environment[EmbodiedCommState]):
    def __init__(self, config: EmbodiedCommConfig, length: int) -> None:
        super().__init__()

        self._config = config
        self._length = length
        self._num_agents = 2

        self.grid_size = config.grid_size
        self.num_colors = config.num_colors
        if self.num_colors < self.grid_size * self.grid_size:
            raise ValueError(
                f"num_colors ({self.num_colors}) must be >= grid_size**2 "
                f"({self.grid_size**2}) so each agent can sample distinct colors."
            )
        if self.num_colors > 8:
            raise ValueError(
                f"num_colors ({self.num_colors}) must be <= 8; "
                f"only TILE_COLOR_0..7 are defined in constance.py."
            )

        self.view_width = config.view_width
        self.view_height = config.view_height
        self.pad_width = self.view_width // 2
        self.pad_height = self.view_height // 2

        self.unpadded_width = 2 * self.grid_size + 1
        self.unpadded_height = self.grid_size
        self.padded_width = self.unpadded_width + 2 * self.pad_width
        self.padded_height = self.unpadded_height + 2 * self.pad_height

        self._agent1_origin_x = self.pad_width
        self._agent2_origin_x = self.pad_width + self.grid_size + 1
        self._origin_y = self.pad_height

        dxs, dys = jnp.meshgrid(
            jnp.arange(self.grid_size, dtype=jnp.int32),
            jnp.arange(self.grid_size, dtype=jnp.int32),
            indexing="ij",
        )
        self._flat_dxs = dxs.flatten()
        self._flat_dys = dys.flatten()

        self._agent1_paint_xs = self._agent1_origin_x + self._flat_dxs
        self._agent1_paint_ys = self._origin_y + self._flat_dys
        self._agent2_paint_xs = self._agent2_origin_x + self._flat_dxs
        self._agent2_paint_ys = self._origin_y + self._flat_dys

        self._action_mask = GW.make_action_mask(
            [
                GW.MOVE_UP,
                GW.MOVE_RIGHT,
                GW.MOVE_DOWN,
                GW.MOVE_LEFT,
                GW.STAY,
                GW.PRIMARY_ACTION,
            ],
            self._num_agents,
        )

    def _paint_map(self, colors_0: jax.Array, colors_1: jax.Array) -> jax.Array:
        base = jnp.full(
            (self.padded_width, self.padded_height),
            GW.TILE_WALL,
            dtype=jnp.int8,
        )
        tile_ids_0 = (GW.TILE_COLOR_0 + colors_0).astype(jnp.int8)
        tile_ids_1 = (GW.TILE_COLOR_0 + colors_1).astype(jnp.int8)
        base = base.at[self._agent1_paint_xs, self._agent1_paint_ys].set(
            tile_ids_0
        )
        base = base.at[self._agent2_paint_xs, self._agent2_paint_ys].set(
            tile_ids_1
        )
        return base

    def reset(self, rng_key: jax.Array) -> tuple[EmbodiedCommState, TimeStep]:
        color_key_0, color_key_1, pos_key_0, pos_key_1 = jax.random.split(
            rng_key, 4
        )

        colors_per_agent = self.grid_size * self.grid_size

        colors_0 = jax.random.choice(
            color_key_0,
            self.num_colors,
            shape=(colors_per_agent,),
            replace=False,
        )
        colors_1 = jax.random.choice(
            color_key_1,
            self.num_colors,
            shape=(colors_per_agent,),
            replace=False,
        )

        map = self._paint_map(colors_0, colors_1)

        start_xy_0 = jax.random.randint(
            pos_key_0, (2,), minval=0, maxval=self.grid_size
        )
        start_xy_1 = jax.random.randint(
            pos_key_1, (2,), minval=0, maxval=self.grid_size
        )

        agent0_pos = jnp.array(
            [
                self._agent1_origin_x + start_xy_0[0],
                self._origin_y + start_xy_0[1],
            ],
            dtype=jnp.int32,
        )
        agent1_pos = jnp.array(
            [
                self._agent2_origin_x + start_xy_1[0],
                self._origin_y + start_xy_1[1],
            ],
            dtype=jnp.int32,
        )
        agents_pos = jnp.stack([agent0_pos, agent1_pos], axis=0)

        state = EmbodiedCommState(
            agents_pos=agents_pos,
            committed=jnp.zeros((self._num_agents,), dtype=jnp.bool_),
            map=map,
            time=jnp.int32(0),
            rewards=jnp.float32(0.0),
        )

        actions = jnp.zeros((self._num_agents,), dtype=jnp.int32)
        rewards = jnp.zeros((self._num_agents,), dtype=jnp.float32)

        return state, self.encode_observations(state, actions, rewards)

    def step(
        self,
        state: EmbodiedCommState,
        action: jax.Array,
        rng_key: jax.Array,
    ) -> tuple[EmbodiedCommState, TimeStep]:
        map = state.map

        @partial(jax.vmap, in_axes=(0, 0, 0), out_axes=(0, 0))
        def _step_agent(local_position, local_action, committed):
            proposed = jnp.where(
                local_action < 4,
                local_position + GW.DIRECTIONS[local_action],
                local_position,
            )
            proposed = jnp.where(committed, local_position, proposed)

            new_tile = map[proposed[0], proposed[1]]
            new_pos = jnp.where(
                new_tile == GW.TILE_WALL, local_position, proposed
            )

            is_commit = local_action == GW.PRIMARY_ACTION
            new_committed = jnp.logical_or(committed, is_commit)

            return new_pos, new_committed

        new_positions, new_committed = _step_agent(
            state.agents_pos, action, state.committed
        )

        both_committed = jnp.all(new_committed)

        tile_0 = map[new_positions[0, 0], new_positions[0, 1]]
        tile_1 = map[new_positions[1, 0], new_positions[1, 1]]
        match = tile_0 == tile_1

        win = jnp.logical_and(both_committed, match)

        new_time = state.time + 1
        is_last_step = new_time == (self._length - 1)
        timed_out = jnp.logical_and(
            is_last_step, jnp.logical_not(both_committed)
        )
        is_terminal = jnp.logical_or(both_committed, is_last_step)

        reward_scalar = jnp.where(
            win,
            jnp.float32(self._config.win_reward),
            jnp.where(
                timed_out,
                jnp.float32(self._config.timeout_penalty),
                jnp.float32(0.0),
            ),
        )
        rewards = jnp.full(
            (self._num_agents,), reward_scalar, dtype=jnp.float32
        )

        new_state = state._replace(
            agents_pos=new_positions,
            committed=new_committed,
            time=new_time,
            rewards=state.rewards + reward_scalar,
        )
        timestep = self.encode_observations(new_state, action, rewards)

        reset_state, reset_timestep = self.reset(rng_key)
        new_state = jax.tree.map(
            lambda terminal, fresh: jnp.where(is_terminal, fresh, terminal),
            new_state,
            reset_state,
        )
        timestep = timestep._replace(
            obs=jnp.where(is_terminal, reset_timestep.obs, timestep.obs),
            time=jnp.where(is_terminal, reset_timestep.time, timestep.time),
        )

        return new_state, timestep

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

    def _render_tiles(
        self,
        state: EmbodiedCommState,
        for_agent: int | None = None,
    ) -> jax.Array:
        tiles = state.map

        if for_agent is not None:
            fog_xs = (
                self._agent2_paint_xs
                if for_agent == 0
                else self._agent1_paint_xs
            )
            fog_ys = (
                self._agent2_paint_ys
                if for_agent == 0
                else self._agent1_paint_ys
            )
            tiles = tiles.at[fog_xs, fog_ys].set(jnp.int8(GW.TILE_GRASS))

        tiles = tiles.at[state.agents_pos[:, 0], state.agents_pos[:, 1]].set(
            GW.AGENT_GENERIC
        )

        directions = jnp.zeros_like(tiles, dtype=jnp.int8)
        teams = jnp.zeros_like(tiles, dtype=jnp.int8)
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

    def encode_observations(
        self,
        state: EmbodiedCommState,
        actions: jax.Array,
        rewards: jax.Array,
    ) -> TimeStep:
        @partial(jax.vmap, in_axes=(0, 0))
        def _encode_view(tiles, positions):
            return jax.lax.dynamic_slice(
                tiles,
                (
                    positions[0] - self.view_width // 2,
                    positions[1] - self.view_height // 2,
                    0,
                ),
                (
                    self.view_width,
                    self.view_height,
                    self.observation_spec.shape[-1],
                ),
            )

        per_agent_tiles = jnp.stack(
            [
                self._render_tiles(state, for_agent=0),
                self._render_tiles(state, for_agent=1),
            ],
            axis=0,
        )
        view = _encode_view(per_agent_tiles, state.agents_pos)

        time = jnp.repeat(state.time[None], self._num_agents, axis=0)

        both_committed = jnp.all(state.committed)
        terminated = jnp.logical_or(
            jnp.equal(time, self._length - 1),
            jnp.repeat(both_committed[None], self._num_agents, axis=0),
        )

        return TimeStep(
            obs=view,
            time=time,
            last_action=actions,
            reward=rewards,
            action_mask=self._action_mask,
            terminated=terminated,
        )

    def create_placeholder_logs(self):
        return {"rewards": jnp.float32(0.0)}

    def create_logs(self, state: EmbodiedCommState):
        return {"rewards": state.rewards}

    def get_render_state(self, state: EmbodiedCommState) -> GridRenderState:
        return GridRenderState(
            tilemap=self._render_tiles(state),
            agent_positions=state.agents_pos,
        )

    def get_render_settings(self) -> GridRenderSettings:
        return GridRenderSettings(
            tile_width=self.unpadded_width,
            tile_height=self.unpadded_height,
            view_width=self.view_width,
            view_height=self.view_height,
        )
