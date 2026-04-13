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

GRID_SIZE = 2
NUM_COLORS = 8


class EmbodiedCommConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    env_type: Literal["embodied_comm"] = "embodied_comm"

    view_width: int = 9
    view_height: int = 9
    win_reward: float = 1.0
    full_info: bool = False


class EmbodiedCommState(NamedTuple):
    agents_pos: jax.Array  # (2, 2) int32, padded coords
    committed: jax.Array  # (2,) bool
    map: jax.Array  # (W_padded, H_padded) int8; colors baked in as TILE_COLOR_*
    time: jax.Array  # () int32
    rewards: jax.Array  # () float32, cumulative episode reward
    episodes: jax.Array


class EmbodiedCommEnv(Environment[EmbodiedCommState]):
    def __init__(self, config: EmbodiedCommConfig, length: int) -> None:
        super().__init__()
        self.config = config
        self.length = length

        self._pad_width = self.config.view_width // 2
        self._pad_height = self.config.view_height // 2

        self._action_mask = GW.make_action_mask(
            [
                GW.MOVE_UP,
                GW.MOVE_RIGHT,
                GW.MOVE_DOWN,
                GW.MOVE_LEFT,
                GW.PRIMARY_ACTION,
            ],
            self.num_agents,
        )

        self._map_mask = self._generate_map_mask()

    def _generate_map(self, rng_key: jax.Array) -> jax.Array:
        grid_a_key, grid_b_key = jax.random.split(rng_key, 2)
        grid_a_colors = (
            jax.random.choice(
                grid_a_key, NUM_COLORS, (GRID_SIZE, GRID_SIZE), replace=False
            )
            + GW.TILE_COLOR_0
        )
        grid_b_colors = (
            jax.random.choice(
                grid_b_key, NUM_COLORS, (GRID_SIZE, GRID_SIZE), replace=False
            )
            + GW.TILE_COLOR_0
        )

        separator = jnp.full((1, GRID_SIZE), GW.TILE_WALL)

        map = jnp.concatenate((grid_a_colors, separator, grid_b_colors), axis=0)

        map = jnp.pad(
            map,
            pad_width=(
                (self._pad_width, self._pad_width),
                (self._pad_height, self._pad_height),
            ),
            mode="constant",
            constant_values=GW.TILE_WALL,
        )

        map = map.astype(jnp.int8)

        return map

    def _generate_map_mask(self) -> jax.Array:
        grid_a = jnp.full((GRID_SIZE, GRID_SIZE), 1, dtype=jnp.int32)
        grid_b = jnp.full((GRID_SIZE, GRID_SIZE), 2, dtype=jnp.int32)

        sep = jnp.full((1, GRID_SIZE), 0, dtype=jnp.int32)

        mask = jnp.concatenate((grid_a, sep, grid_b), axis=0)
        mask = jnp.pad(
            mask,
            pad_width=(
                (self._pad_width, self._pad_width),
                (self._pad_height, self._pad_height),
            ),
            mode="empty",
        )

        mask = mask.astype(jnp.int8)

        return mask

    def reset(self, rng_key: jax.Array) -> tuple[EmbodiedCommState, TimeStep]:
        map = self._generate_map(rng_key)
        agents_pos = jnp.array(
            [
                [self._pad_width, self._pad_height],
                [self._pad_width + GRID_SIZE + 1, self._pad_height],
            ],
            dtype=jnp.int32,
        )

        committed = jnp.zeros((self.num_agents,), jnp.bool_)
        time = jnp.int32(0)
        rewards = jnp.float32(0.0)
        episodes = jnp.int32(0)

        actions = jnp.zeros((self.num_agents,), dtype=jnp.int32)
        state = EmbodiedCommState(
            map=map,
            agents_pos=agents_pos,
            committed=committed,
            time=time,
            rewards=rewards,
            episodes=episodes,
        )

        return state, self.encode_observations(
            state,
            actions,
            jnp.zeros((self.num_agents,)),
            jnp.zeros((self.num_agents,), dtype=jnp.bool_),
        )

    def _maybe_reset(
        self,
        state: EmbodiedCommState,
        all_committed: jax.Array,
        rng_key: jax.Array,
    ) -> EmbodiedCommState:
        map = jnp.where(
            all_committed,
            self._generate_map(rng_key),
            state.map,
        )
        committed = jnp.where(all_committed, False, state.committed)

        return state._replace(map=map, committed=committed)

    def step(
        self,
        state: EmbodiedCommState,
        action: jax.Array,
        rng_key: jax.Array,
    ) -> tuple[EmbodiedCommState, TimeStep]:
        @partial(jax.vmap, in_axes=(0, 0))
        def _step_agent(local_position, local_action):
            target_position = (
                local_position + GW.DIRECTIONS[jnp.minimum(local_action, 3)]
            )
            target_tile = state.map[target_position[0], target_position[1]]

            target_position = jnp.where(
                (target_tile == GW.TILE_WALL)
                | (local_action == GW.PRIMARY_ACTION),
                local_position,
                target_position,
            )

            return target_position

        committed = state.committed | (action == GW.PRIMARY_ACTION)
        action = jnp.where(committed, GW.PRIMARY_ACTION, action)

        agents_pos = _step_agent(state.agents_pos, action)

        # rewards and reset
        all_committed = jnp.all(committed)
        color_a = state.map[agents_pos[0, 0], agents_pos[0, 1]]
        color_b = state.map[agents_pos[1, 0], agents_pos[1, 1]]
        win = all_committed & (color_a == color_b)

        reward = self.config.win_reward * win
        rewards = jnp.full((self.num_agents,), reward)

        state = state._replace(
            agents_pos=agents_pos,
            committed=committed,
            time=state.time + 1,
            rewards=state.rewards + reward,
            episodes=state.episodes + all_committed,
        )
        terminated = jnp.full(
            (self.num_agents,), all_committed, dtype=jnp.bool_
        )

        next_state = self._maybe_reset(state, all_committed, rng_key)

        return next_state, self.encode_observations(
            state,
            action,
            rewards,
            terminated,
        )

    @cached_property
    def observation_spec(self) -> ObservationSpec:
        return GW.make_obs_spec(self.config.view_width, self.config.view_height)

    @cached_property
    def action_spec(self) -> DiscreteActionSpec:
        return DiscreteActionSpec(n=GW.NUM_ACTIONS)

    @property
    def is_jittable(self) -> bool:
        return True

    @property
    def num_agents(self) -> int:
        return 2

    def _render_tiles(
        self, state: EmbodiedCommState, pov: jax.Array | None = None
    ):
        tiles = state.map
        if pov is not None and not self.config.full_info:
            hidden = (self._map_mask > 0) & (self._map_mask != pov + 1)
            tiles = jnp.where(hidden, GW.TILE_EMPTY, tiles)

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
        terminated: jax.Array,
    ) -> TimeStep:
        @partial(jax.vmap, in_axes=(None, 0, 0))
        def _encode_view(state, agent_id, positions):
            tiles = self._render_tiles(state, agent_id)
            return jax.lax.dynamic_slice(
                tiles,
                (
                    positions[0] - self.config.view_width // 2,
                    positions[1] - self.config.view_height // 2,
                    0,
                ),
                (
                    self.config.view_width,
                    self.config.view_height,
                    self.observation_spec.shape[-1],
                ),
            )

        view = _encode_view(
            state, jnp.arange(self.num_agents), state.agents_pos
        )

        time = jnp.repeat(state.time[None], self.num_agents, axis=0)

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
        return {"rewards": state.rewards / state.episodes}

    def get_render_state(self, state: EmbodiedCommState) -> GridRenderState:
        return GridRenderState(
            tilemap=self._render_tiles(state),
            agent_positions=state.agents_pos,
        )

    def get_render_settings(self) -> GridRenderSettings:
        return GridRenderSettings(
            tile_width=GRID_SIZE * 2 + 1,
            tile_height=GRID_SIZE,
            view_width=self.config.view_width,
            view_height=self.config.view_height,
        )
