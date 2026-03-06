from functools import cached_property, partial
from typing import NamedTuple, Literal

import jax
from jax import numpy as jnp
from pydantic import BaseModel, ConfigDict

from mapox.map_generator import generate_decor_tiles, generate_perlin_noise_2d
from mapox.environment import Environment
from mapox.specs import DiscreteActionSpec, ObservationSpec
from mapox.timestep import TimeStep
from mapox.renderer import GridRenderSettings, GridRenderState
import mapox.envs.constance as GW


class StealthConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    env_type: Literal["stealth"] = "stealth"

    num_sneakers: int = 3
    num_chasers: int = 2
    width: int = 40
    height: int = 20
    view_width: int = 11
    view_height: int = 11
    grass_threshold: float = -0.15
    spawn_strip_width: int = 3
    catch_reward: float = 1.0
    cross_reward: float = 1.0


class StealthState(NamedTuple):
    sneaker_pos: jax.Array       # (num_sneakers, 2)
    chaser_pos: jax.Array        # (num_chasers, 2)
    chaser_spawn_pos: jax.Array  # (num_chasers, 2) fixed per episode
    tiles: jax.Array             # (padded_w, padded_h)
    time: jax.Array              # ()
    rewards: jax.Array           # () cumulative for logging
    left_spawns: jax.Array       # (max_spawns, 2) valid left-side positions
    left_spawn_count: jax.Array  # () number of valid left spawns


class StealthEnv(Environment[StealthState]):
    def __init__(self, config: StealthConfig, length: int) -> None:
        super().__init__()

        self._length = length
        self._config = config
        self._num_sneakers = config.num_sneakers
        self._num_chasers = config.num_chasers

        self.unpadded_width = config.width
        self.unpadded_height = config.height

        self.view_width = config.view_width
        self.view_height = config.view_height
        self.pad_width = self.view_width // 2
        self.pad_height = self.view_height // 2

        self.padded_width = self.unpadded_width + self.pad_width * 2
        self.padded_height = self.unpadded_height + self.pad_height * 2

        self._goal_x = self.pad_width + self.unpadded_width - 1

        self._teams = jnp.concatenate([
            jnp.zeros(config.num_sneakers, dtype=jnp.int8),
            jnp.ones(config.num_chasers, dtype=jnp.int8),
        ])

        self._action_mask = GW.make_action_mask(
            [GW.MOVE_UP, GW.MOVE_RIGHT, GW.MOVE_DOWN, GW.MOVE_LEFT, GW.STAY],
            self.num_agents,
        )

    def _generate_map(self, rng_key):
        wall_key, grass_key, decor_key, rng_key = jax.random.split(rng_key, 4)

        w, h = self.unpadded_width, self.unpadded_height

        decor = generate_decor_tiles(w, h, decor_key)

        # Perlin noise for walls (res must divide dimensions)
        wall_noise = generate_perlin_noise_2d((w, h), (2, 2), rng_key=wall_key)
        tiles = jnp.where(wall_noise > 0.1, jnp.int8(GW.TILE_WALL), decor)

        # Clear the left spawn strip and right goal strip
        strip = self._config.spawn_strip_width
        left_strip = decor[:strip, :]
        right_strip = decor[-strip:, :]
        tiles = tiles.at[:strip, :].set(left_strip)
        tiles = tiles.at[-strip:, :].set(right_strip)

        # Perlin noise for grass
        grass_noise = generate_perlin_noise_2d((w, h), (2, 2), rng_key=grass_key)
        is_grass = (grass_noise < self._config.grass_threshold) & (tiles != GW.TILE_WALL)
        # No grass on spawn/goal strips
        strip_mask = jnp.ones((w, h), dtype=jnp.bool_)
        strip_mask = strip_mask.at[:strip, :].set(False)
        strip_mask = strip_mask.at[-strip:, :].set(False)
        is_grass = is_grass & strip_mask
        tiles = jnp.where(is_grass, jnp.int8(GW.TILE_GRASS), tiles)

        # Collect left-side spawn positions (empty tiles in left strip, unpadded coords)
        left_tiles = tiles[:strip, :]
        is_empty = (left_tiles == GW.TILE_EMPTY) | (
            (left_tiles >= GW.TILE_DECOR_1) & (left_tiles <= GW.TILE_DECOR_4)
        )
        max_spawns = strip * h
        x_spawns, y_spawns = jnp.where(
            is_empty, size=max_spawns, fill_value=-1
        )
        spawn_count = jnp.sum(is_empty)

        # Pad tiles
        tiles = jnp.pad(
            tiles,
            pad_width=(
                (self.pad_width, self.pad_width),
                (self.pad_height, self.pad_height),
            ),
            mode="constant",
            constant_values=GW.TILE_WALL,
        )

        # Convert to padded coords
        x_spawns = x_spawns + self.pad_width
        y_spawns = y_spawns + self.pad_height
        left_spawns = jnp.stack((x_spawns, y_spawns), axis=1)

        return tiles, left_spawns, spawn_count

    def _spawn_sneakers(self, left_spawns, spawn_count, rng_key):
        indices = jax.random.randint(
            rng_key, (self._num_sneakers,), minval=0, maxval=spawn_count
        )
        return left_spawns[indices]

    def _spawn_chasers(self, tiles, rng_key):
        w, h = self.unpadded_width, self.unpadded_height
        strip = self._config.spawn_strip_width

        # Chasers spawn in the middle area (not in strips)
        inner = tiles[
            self.pad_width + strip : self.pad_width + w - strip,
            self.pad_height : self.pad_height + h,
        ]
        is_open = (inner != GW.TILE_WALL)
        max_inner = (w - strip * 2) * h
        x_spawns, y_spawns = jnp.where(is_open, size=max_inner, fill_value=-1)
        open_count = jnp.sum(is_open)
        x_spawns = x_spawns + self.pad_width + strip
        y_spawns = y_spawns + self.pad_height

        indices = jax.random.randint(
            rng_key, (self._num_chasers,), minval=0, maxval=open_count
        )
        return jnp.stack((x_spawns[indices], y_spawns[indices]), axis=1)

    def reset(self, rng_key: jax.Array) -> tuple[StealthState, TimeStep]:
        map_key, sneaker_key, chaser_key = jax.random.split(rng_key, 3)

        tiles, left_spawns, spawn_count = self._generate_map(map_key)
        sneaker_pos = self._spawn_sneakers(left_spawns, spawn_count, sneaker_key)
        chaser_pos = self._spawn_chasers(tiles, chaser_key)

        state = StealthState(
            sneaker_pos=sneaker_pos,
            chaser_pos=chaser_pos,
            chaser_spawn_pos=chaser_pos.copy(),
            tiles=tiles,
            time=jnp.int32(0),
            rewards=jnp.float32(0.0),
            left_spawns=left_spawns,
            left_spawn_count=spawn_count,
        )

        actions = jnp.zeros((self.num_agents,), dtype=jnp.int32)
        rewards = jnp.zeros((self.num_agents,), dtype=jnp.float32)

        return state, self.encode_observations(state, actions, rewards)

    def step(
        self, state: StealthState, action: jax.Array, rng_key: jax.Array
    ) -> tuple[StealthState, TimeStep]:
        move_key, respawn_key, rng_key = jax.random.split(rng_key, 3)

        num_agents = self.num_agents
        ns = self._num_sneakers
        nc = self._num_chasers
        tiles = state.tiles

        # Combine all positions for unified movement resolution
        all_pos = jnp.concatenate([state.sneaker_pos, state.chaser_pos], axis=0)

        # Team masks for same-team collision (only block same team)
        agent_teams = jnp.concatenate([
            jnp.zeros(ns, dtype=jnp.int32),
            jnp.ones(nc, dtype=jnp.int32),
        ])

        # Random execution order
        order = jax.random.permutation(move_key, num_agents)

        def body(i, carry):
            positions = carry
            agent_idx = order[i]
            agent_action = action[agent_idx]

            direction = GW.DIRECTIONS[jnp.minimum(agent_action, 3)]
            is_stay = agent_action == GW.STAY
            proposed = positions[agent_idx] + direction
            proposed = jnp.where(is_stay, positions[agent_idx], proposed)

            is_wall = tiles[proposed[0], proposed[1]] == GW.TILE_WALL

            # Only block on same-team agents
            same_team = agent_teams == agent_teams[agent_idx]
            other_mask = jnp.arange(num_agents) != agent_idx
            occupied_by_teammate = jnp.any(
                jnp.all(positions == proposed, axis=-1) & other_mask & same_team
            )
            blocked = is_wall | occupied_by_teammate

            new_pos = jnp.where(blocked, positions[agent_idx], proposed)
            positions = positions.at[agent_idx].set(new_pos)
            return positions

        all_pos = jax.lax.fori_loop(0, num_agents, body, all_pos)

        new_sneaker_pos = all_pos[:ns]
        new_chaser_pos = all_pos[ns:]

        # --- Crossing detection ---
        crossed = new_sneaker_pos[:, 0] >= self._goal_x
        cross_rewards = crossed.astype(jnp.float32) * self._config.cross_reward

        # --- Catch detection ---
        # Chaser catches sneaker if adjacent (Manhattan distance <= 1)
        # (num_sneakers, num_chasers, 2)
        diff = jnp.abs(new_sneaker_pos[:, None, :] - new_chaser_pos[None, :, :])
        adjacent = jnp.sum(diff, axis=-1) <= 1  # (num_sneakers, num_chasers)
        sneaker_caught = jnp.any(adjacent, axis=1)  # (num_sneakers,)
        chaser_caught = jnp.any(adjacent, axis=0)   # (num_chasers,)

        sneaker_catch_penalty = sneaker_caught.astype(jnp.float32) * self._config.catch_reward
        chaser_catch_reward = chaser_caught.astype(jnp.float32) * self._config.catch_reward

        sneaker_rewards = cross_rewards - sneaker_catch_penalty
        chaser_rewards = chaser_catch_reward

        # --- Respawns ---
        sneaker_needs_respawn = crossed | sneaker_caught
        chaser_needs_respawn = chaser_caught

        # Sneakers respawn to random left-side positions
        sneaker_respawn_keys = jax.random.split(respawn_key, ns + 1)
        respawn_key = sneaker_respawn_keys[0]
        sneaker_respawn_keys = sneaker_respawn_keys[1:]

        def respawn_sneaker(i, pos):
            idx = jax.random.randint(
                sneaker_respawn_keys[i], (), minval=0, maxval=state.left_spawn_count
            )
            new_pos = state.left_spawns[idx]
            return jnp.where(sneaker_needs_respawn[i], new_pos, pos)

        new_sneaker_pos = jax.vmap(respawn_sneaker)(
            jnp.arange(ns), new_sneaker_pos
        )

        # Chasers respawn to their fixed spawn positions
        new_chaser_pos = jnp.where(
            chaser_needs_respawn[:, None], state.chaser_spawn_pos, new_chaser_pos
        )

        rewards = jnp.concatenate([sneaker_rewards, chaser_rewards])

        state = StealthState(
            sneaker_pos=new_sneaker_pos,
            chaser_pos=new_chaser_pos,
            chaser_spawn_pos=state.chaser_spawn_pos,
            tiles=state.tiles,
            time=state.time + 1,
            rewards=state.rewards + jnp.mean(rewards),
            left_spawns=state.left_spawns,
            left_spawn_count=state.left_spawn_count,
        )

        return state, self.encode_observations(state, action, rewards)

    def _render_tiles_full(self, state: StealthState):
        """Render all agents unconditionally (for human/debug renderer)."""
        tiles = state.tiles

        directions = jnp.zeros_like(tiles, dtype=jnp.int8)
        teams = jnp.zeros_like(tiles, dtype=jnp.int8)
        health = jnp.zeros_like(tiles, dtype=jnp.int8)

        # Sneakers
        tiles = tiles.at[state.sneaker_pos[:, 0], state.sneaker_pos[:, 1]].set(
            GW.AGENT_SCOUT
        )
        teams = teams.at[state.sneaker_pos[:, 0], state.sneaker_pos[:, 1]].set(1)

        # Chasers
        tiles = tiles.at[state.chaser_pos[:, 0], state.chaser_pos[:, 1]].set(
            GW.AGENT_HARVESTER
        )
        teams = teams.at[state.chaser_pos[:, 0], state.chaser_pos[:, 1]].set(2)

        return jnp.concatenate(
            (
                tiles[..., None],
                directions[..., None],
                teams[..., None],
                health[..., None],
            ),
            axis=-1,
        )

    def _render_tiles_concealed(self, state: StealthState):
        """Render agents with grass concealment (for observations)."""
        tiles = state.tiles
        base_tiles = state.tiles

        directions = jnp.zeros_like(tiles, dtype=jnp.int8)
        teams = jnp.zeros_like(tiles, dtype=jnp.int8)
        health = jnp.zeros_like(tiles, dtype=jnp.int8)

        # Sneakers: only visible if NOT on grass
        sneaker_on_grass = base_tiles[
            state.sneaker_pos[:, 0], state.sneaker_pos[:, 1]
        ] == GW.TILE_GRASS
        sneaker_visible = ~sneaker_on_grass

        visible_sneaker_tile = jnp.where(sneaker_visible, GW.AGENT_GENERIC, tiles[
            state.sneaker_pos[:, 0], state.sneaker_pos[:, 1]
        ])
        tiles = tiles.at[state.sneaker_pos[:, 0], state.sneaker_pos[:, 1]].set(
            visible_sneaker_tile
        )
        visible_sneaker_team = jnp.where(sneaker_visible, jnp.int8(1), jnp.int8(0))
        teams = teams.at[state.sneaker_pos[:, 0], state.sneaker_pos[:, 1]].set(
            visible_sneaker_team
        )

        # Chasers: only visible if NOT on grass
        chaser_on_grass = base_tiles[
            state.chaser_pos[:, 0], state.chaser_pos[:, 1]
        ] == GW.TILE_GRASS
        chaser_visible = ~chaser_on_grass

        visible_chaser_tile = jnp.where(chaser_visible, GW.AGENT_GENERIC, tiles[
            state.chaser_pos[:, 0], state.chaser_pos[:, 1]
        ])
        tiles = tiles.at[state.chaser_pos[:, 0], state.chaser_pos[:, 1]].set(
            visible_chaser_tile
        )
        visible_chaser_team = jnp.where(chaser_visible, jnp.int8(2), jnp.int8(0))
        teams = teams.at[state.chaser_pos[:, 0], state.chaser_pos[:, 1]].set(
            visible_chaser_team
        )

        return jnp.concatenate(
            (
                tiles[..., None],
                directions[..., None],
                teams[..., None],
                health[..., None],
            ),
            axis=-1,
        )

    def encode_observations(self, state: StealthState, actions, rewards) -> TimeStep:
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

        tiles = self._render_tiles_concealed(state)
        agents_pos = jnp.concatenate(
            [state.sneaker_pos, state.chaser_pos], axis=0
        )
        view = _encode_view(tiles, agents_pos)

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
        return self._num_sneakers + self._num_chasers

    @property
    def teams(self) -> jax.Array:
        return self._teams

    def create_placeholder_logs(self):
        return {"rewards": jnp.float32(0.0)}

    def create_logs(self, state: StealthState):
        return {"rewards": state.rewards}

    def get_render_state(self, state: StealthState) -> GridRenderState:
        tiles = self._render_tiles_full(state)
        agent_positions = jnp.concatenate(
            [state.sneaker_pos, state.chaser_pos], axis=0
        )
        return GridRenderState(
            tilemap=tiles,
            agent_positions=agent_positions,
        )

    def get_render_settings(self) -> GridRenderSettings:
        return GridRenderSettings(
            tile_width=self.unpadded_width,
            tile_height=self.unpadded_height,
            view_width=self.view_width,
            view_height=self.view_height,
        )
