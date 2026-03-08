# sneakers = prey, chasers = predators
from functools import cached_property, partial
from typing import Literal, NamedTuple

import jax
from jax import numpy as jnp
from pydantic import BaseModel, ConfigDict

import mapox.envs.constance as GW
from mapox.environment import Environment
from mapox.map_generator import generate_decor_tiles, generate_perlin_noise_2d
from mapox.renderer import GridRenderSettings, GridRenderState
from mapox.specs import DiscreteActionSpec, ObservationSpec
from mapox.timestep import TimeStep


class StealthConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    env_type: Literal["stealth"] = "stealth"

    num_sneakers: int = 3
    num_chasers: int = 2
    width: int = 40
    height: int = 40
    view_width: int = 11
    view_height: int = 11
    grass_threshold: float = -0.5
    num_food: int = 15
    food_regrow_time: int = 10
    max_fullness: int = 40
    initial_fullness: int = 20
    fullness_per_food: int = 20
    fullness_per_catch: int = 20
    full_threshold: int = 30
    survival_reward_scale: float = 0.05


class StealthState(NamedTuple):
    sneaker_pos: jax.Array  # (num_sneakers, 2)
    chaser_pos: jax.Array  # (num_chasers, 2)
    tiles: jax.Array  # (padded_w, padded_h)
    time: jax.Array  # ()
    rewards: jax.Array  # () cumulative for logging
    food_pos: jax.Array  # (num_food, 2)
    food_timer: jax.Array  # (num_food,) 0 = present, >0 = respawning
    fullness: jax.Array  # (num_agents,) 0 = starving, decrements each step


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

        self._teams = jnp.concatenate(
            [
                jnp.zeros(config.num_sneakers, dtype=jnp.int8),
                jnp.ones(config.num_chasers, dtype=jnp.int8),
            ]
        )

        self._action_mask = GW.make_action_mask(
            [GW.MOVE_UP, GW.MOVE_RIGHT, GW.MOVE_DOWN, GW.MOVE_LEFT, GW.STAY],
            self.num_agents,
        )

    def _generate_map(self, rng_key):
        wall_key, grass_key, decor_key, rng_key = jax.random.split(rng_key, 4)

        w, h = self.unpadded_width, self.unpadded_height

        decor = generate_decor_tiles(w, h, decor_key)

        # Perlin noise for walls (res must divide dimensions)
        wall_noise = generate_perlin_noise_2d((w, h), (8, 8), rng_key=wall_key)
        tiles = jnp.where(wall_noise > 0.1, jnp.int8(GW.TILE_WALL), decor)

        # Perlin noise for grass
        grass_noise = generate_perlin_noise_2d((w, h), (8, 8), rng_key=grass_key)
        is_grass = (grass_noise < self._config.grass_threshold) & (
            tiles != GW.TILE_WALL
        )
        tiles = jnp.where(is_grass, jnp.int8(GW.TILE_GRASS), tiles)

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

        return tiles

    def _spawn(self, tiles, n, rng_key, allow_grass: bool = False):
        """Pick n random open tiles in the playable area."""
        w, h = self.unpadded_width, self.unpadded_height
        inner = tiles[
            self.pad_width : self.pad_width + w,
            self.pad_height : self.pad_height + h,
        ]
        is_open = inner != GW.TILE_WALL
        if not allow_grass:
            is_open = is_open & (inner != GW.TILE_GRASS)
        max_open = w * h
        x_open, y_open = jnp.where(is_open, size=max_open, fill_value=-1)
        open_count = jnp.sum(is_open)
        x_open = x_open + self.pad_width
        y_open = y_open + self.pad_height

        indices = jax.random.randint(rng_key, (n,), minval=0, maxval=open_count)
        return jnp.stack((x_open[indices], y_open[indices]), axis=1)

    def reset(self, rng_key: jax.Array) -> tuple[StealthState, TimeStep]:
        map_key, sneaker_key, chaser_key, food_key = jax.random.split(rng_key, 4)

        tiles = self._generate_map(map_key)
        sneaker_pos = self._spawn(tiles, self._num_sneakers, sneaker_key)
        chaser_pos = self._spawn(tiles, self._num_chasers, chaser_key)
        food_pos = self._spawn(tiles, self._config.num_food, food_key, allow_grass=True)

        state = StealthState(
            sneaker_pos=sneaker_pos,
            chaser_pos=chaser_pos,
            tiles=tiles,
            time=jnp.int32(0),
            rewards=jnp.float32(0.0),
            food_pos=food_pos,
            food_timer=jnp.zeros(self._config.num_food, dtype=jnp.int32),
            fullness=jnp.full(self.num_agents, self._config.initial_fullness, dtype=jnp.int32),
        )

        actions = jnp.zeros((self.num_agents,), dtype=jnp.int32)
        rewards = jnp.zeros((self.num_agents,), dtype=jnp.float32)
        terminated = jnp.zeros(self.num_agents, dtype=jnp.bool_)

        return state, self.encode_observations(state, actions, rewards, terminated)

    def step(
        self, state: StealthState, action: jax.Array, rng_key: jax.Array
    ) -> tuple[StealthState, TimeStep]:
        move_key, respawn_key, food_key, rng_key = jax.random.split(rng_key, 4)

        num_agents = self.num_agents
        ns = self._num_sneakers
        nf = self._config.num_food
        tiles = state.tiles
        cfg = self._config

        # --- Movement resolution ---
        all_pos = jnp.concatenate([state.sneaker_pos, state.chaser_pos], axis=0)

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

            other_mask = jnp.arange(num_agents) != agent_idx
            is_occupied = jnp.any(jnp.all(positions == proposed, axis=-1) & other_mask)
            blocked = is_wall | is_occupied

            new_pos = jnp.where(blocked, positions[agent_idx], proposed)
            positions = positions.at[agent_idx].set(new_pos)
            return positions

        all_pos = jax.lax.fori_loop(0, num_agents, body, all_pos)

        new_sneaker_pos = all_pos[:ns]
        new_chaser_pos = all_pos[ns:]

        # --- Fullness: decrement then eat ---
        fullness = jnp.maximum(state.fullness - 1, 0)
        can_eat = fullness < cfg.full_threshold
        sneaker_can_eat = can_eat[:ns]
        chaser_can_eat = can_eat[ns:]

        # --- Food eating (sneakers only, gated by fullness) ---
        food_present = state.food_timer == 0  # (num_food,)

        food_sneaker_diff = state.food_pos[:, None, :] - new_sneaker_pos[None, :, :]
        food_sneaker_match = (
            jnp.all(food_sneaker_diff == 0, axis=-1) & sneaker_can_eat[None, :]
        )  # (num_food, num_sneakers)

        food_eaten = food_present & jnp.any(food_sneaker_match, axis=1)  # (num_food,)
        first_eater = jnp.argmax(food_sneaker_match, axis=1)  # (num_food,)

        # Accumulate fullness gain per sneaker
        eaten_gain = food_eaten.astype(jnp.int32) * cfg.fullness_per_food
        sneaker_gain = jnp.zeros(ns, dtype=jnp.int32).at[first_eater].add(eaten_gain)

        # Update food timers
        new_food_timer = jnp.where(
            food_eaten, cfg.food_regrow_time, state.food_timer
        )
        new_food_timer = jnp.where(
            new_food_timer > 0, new_food_timer - 1, new_food_timer
        )

        # Respawn food that just hit 0
        food_respawning = (state.food_timer == 1) & (~food_eaten)
        respawn_food_pos = self._spawn(tiles, nf, food_key, allow_grass=True)
        new_food_pos = jnp.where(
            food_respawning[:, None], respawn_food_pos, state.food_pos
        )

        # --- Catch detection (gated by chaser fullness) ---
        diff = jnp.abs(new_sneaker_pos[:, None, :] - new_chaser_pos[None, :, :])
        adjacent = jnp.sum(diff, axis=-1) <= 1  # (num_sneakers, num_chasers)
        adjacent = adjacent & chaser_can_eat[None, :]

        sneaker_caught = jnp.any(adjacent, axis=1)  # (num_sneakers,)
        chaser_caught = jnp.any(adjacent, axis=0)  # (num_chasers,)

        chaser_gain = chaser_caught.astype(jnp.int32) * cfg.fullness_per_catch

        # --- Apply fullness gains ---
        gain = jnp.concatenate([sneaker_gain, chaser_gain])
        fullness = jnp.minimum(fullness + gain, cfg.max_fullness)

        # --- Termination: starvation or caught ---
        starving = fullness == 0
        time_up = jnp.equal(state.time + 1, self._length)
        terminated = starving | time_up
        terminated = terminated.at[:ns].set(terminated[:ns] | sneaker_caught)

        # --- Respawn terminated agents ---
        respawn_pos = self._spawn(tiles, num_agents, respawn_key)
        all_pos = jnp.where(terminated[:, None], respawn_pos, all_pos)
        fullness = jnp.where(terminated, cfg.initial_fullness, fullness)

        new_sneaker_pos = all_pos[:ns]
        new_chaser_pos = all_pos[ns:]

        # --- Survival rewards (diminishing returns based on fullness) ---
        rewards = cfg.survival_reward_scale * (fullness.astype(jnp.float32) / cfg.max_fullness)
        rewards = jnp.where(terminated, 0.0, rewards)

        state = StealthState(
            sneaker_pos=new_sneaker_pos,
            chaser_pos=new_chaser_pos,
            tiles=state.tiles,
            time=state.time + 1,
            rewards=state.rewards + jnp.mean(rewards),
            food_pos=new_food_pos,
            food_timer=new_food_timer,
            fullness=fullness,
        )

        return state, self.encode_observations(state, action, rewards, terminated)

    def _render_tiles(self, state: StealthState, conceal: bool = False):
        """Render agents and food. If conceal=True, agents on grass are hidden."""
        tiles = state.tiles

        directions = jnp.zeros_like(tiles, dtype=jnp.int8)
        teams = jnp.zeros_like(tiles, dtype=jnp.int8)
        health = jnp.zeros_like(tiles, dtype=jnp.int8)

        # Food (present items only)
        food_present = state.food_timer == 0
        food_tile_vals = jnp.where(
            food_present,
            GW.TILE_FOOD,
            tiles[state.food_pos[:, 0], state.food_pos[:, 1]],
        )
        tiles = tiles.at[state.food_pos[:, 0], state.food_pos[:, 1]].set(food_tile_vals)

        def _apply_agents(tiles, pos, agent_tile, team_id):
            if conceal:
                on_grass = state.tiles[pos[:, 0], pos[:, 1]] == GW.TILE_GRASS
                visible = ~on_grass
                tile_vals = jnp.where(visible, agent_tile, tiles[pos[:, 0], pos[:, 1]])
            else:
                tile_vals = agent_tile
            tiles = tiles.at[pos[:, 0], pos[:, 1]].set(tile_vals)
            return tiles

        tiles = _apply_agents(tiles, state.sneaker_pos, GW.AGENT_SCOUT, 1)
        tiles = _apply_agents(tiles, state.chaser_pos, GW.AGENT_HARVESTER, 2)

        # Map fullness to health channel: 0=starving, 1=low, 2=high
        half = self._config.max_fullness // 2
        agents_pos = jnp.concatenate([state.sneaker_pos, state.chaser_pos], axis=0)
        agent_health = jnp.where(
            state.fullness == 0, jnp.int8(0),
            jnp.where(state.fullness <= half, jnp.int8(1), jnp.int8(2)),
        )
        if conceal:
            on_grass = state.tiles[agents_pos[:, 0], agents_pos[:, 1]] == GW.TILE_GRASS
            agent_health = jnp.where(on_grass, jnp.int8(0), agent_health)
        health = health.at[agents_pos[:, 0], agents_pos[:, 1]].set(agent_health)

        return jnp.concatenate(
            (
                tiles[..., None],
                directions[..., None],
                teams[..., None],
                health[..., None],
            ),
            axis=-1,
        )

    def encode_observations(self, state: StealthState, actions, rewards, terminated) -> TimeStep:
        @partial(jax.vmap, in_axes=(None, 0))
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

        tiles = self._render_tiles(state, conceal=True)
        agents_pos = jnp.concatenate([state.sneaker_pos, state.chaser_pos], axis=0)
        view = _encode_view(tiles, agents_pos)

        time = jnp.repeat(state.time[None], self.num_agents, axis=0)

        return TimeStep(
            obs=view,
            time=time,
            last_action=actions,
            reward=rewards,
            action_mask=self._action_mask,
            terminated=terminated,
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
        tiles = self._render_tiles(state)
        agent_positions = jnp.concatenate([state.sneaker_pos, state.chaser_pos], axis=0)
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
