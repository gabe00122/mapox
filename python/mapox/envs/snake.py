from functools import cached_property
from typing import Literal, NamedTuple

import jax
from jax import numpy as jnp
from pydantic import BaseModel, ConfigDict

import mapox.symbols as SB
from mapox.environment import Environment
from mapox.envs.common import DIRECTIONS, make_action_mask, make_obs_spec
from mapox.map_generator import generate_decor_tiles, register_decor_tiles
from mapox.vocab import Vocabulary
from mapox.renderer import GridRenderSettings, GridRenderState
from mapox.specs import DiscreteActionSpec, ObservationSpec
from mapox.timestep import TimeStep


class SnakeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    env_type: Literal["snake"] = "snake"

    num_agents: int = 4
    width: int = 24
    height: int = 24
    view_width: int = 11
    view_height: int = 11
    initial_length: int = 3
    max_length: int = 64
    initial_food: int = 8
    food_spawn_prob: float = 0.15
    food_reward: float = 1.0
    death_reward: float = -1.0


class SnakeState(NamedTuple):
    body: jax.Array  # (num_agents, max_length, 2) ring buffer of segment cells
    head_slot: jax.Array  # (num_agents,) buffer slot holding the head
    head_pos: jax.Array  # (num_agents, 2)
    direction: jax.Array  # (num_agents,) 0..3
    length: jax.Array  # (num_agents,)
    food: jax.Array  # (padded_w, padded_h) bool
    tiles: jax.Array  # (padded_w, padded_h)
    time: jax.Array  # ()
    rewards: jax.Array  # () cumulative for logging
    deaths: jax.Array  # () cumulative for logging


class SnakeEnv(Environment[SnakeState]):
    """Multiplayer snake.

    Each body is a fixed-size ring buffer of board cells: the head lives at
    slot `head_slot` and the `length` most recent slots (wrapping backwards)
    are the body, tail last. A move advances the head slot and writes the new
    head cell there with a single scattered write; the tail vacates implicitly
    by falling out of the length window. Eating extends the window by one so
    the tail stays put — until `max_length`, where pellets are still consumed
    and rewarded but the snake stops growing. Occupancy grids are re-scattered
    from the buffers on demand, so state size and per-step body work scale
    with num_agents * max_length rather than num_agents * width * height.

    All snakes move at once. Collisions are resolved against post-tail-move
    occupancy, so a snake may safely follow a tail that vacates this step, but
    the body of a snake dying this same step still kills. Half of a dead
    snake's body (alternating segments) turns into food and the snake respawns
    at a random open cell. Reversing into your own neck is masked out and, if
    forced, keeps the snake moving straight.
    """

    def __init__(self, config: SnakeConfig, length: int) -> None:
        super().__init__()

        if config.max_length < config.initial_length:
            raise ValueError("max_length must be at least initial_length")

        self._length = length
        self._config = config

        self.unpadded_width = config.width
        self.unpadded_height = config.height

        self.view_width = config.view_width
        self.view_height = config.view_height
        self.pad_width = self.view_width // 2
        self.pad_height = self.view_height // 2

        self.padded_width = self.unpadded_width + self.pad_width * 2
        self.padded_height = self.unpadded_height + self.pad_height * 2

        self._obs_vocab = Vocabulary()
        self._action_vocab = Vocabulary()

        register_decor_tiles(self._obs_vocab)
        self._tile_empty = self._obs_vocab.add(SB.TILE_EMPTY)
        self._tile_wall = self._obs_vocab.add(SB.TILE_WALL)
        self._tile_food = self._obs_vocab.add(SB.TILE_FOOD)
        self._agent_snake_head = self._obs_vocab.add(SB.AGENT_SNAKE_HEAD)
        self._agent_snake_body = self._obs_vocab.add(SB.AGENT_SNAKE_BODY)

        moves = self._action_vocab.add_block(SB.MOVES)
        assert moves == range(0, 4)  # DIRECTIONS indexing and reversal math

        self._base_action_mask = make_action_mask(
            list(moves), len(self._action_vocab), self.num_agents
        )

        self._obs_vocab.freeze()
        self._action_vocab.freeze()

    def _sample_open_cells(self, open_mask: jax.Array, n: int, rng_key: jax.Array):
        """Uniformly sample n distinct open cells (random scores + top-k)."""
        scores = jax.random.uniform(rng_key, (open_mask.size,))
        scores = jnp.where(open_mask.reshape(-1), scores, -1.0)
        _, idx = jax.lax.top_k(scores, n)
        cx, cy = jnp.unravel_index(idx, open_mask.shape)
        return jnp.stack((cx, cy), axis=1).astype(jnp.int32)

    def _segment_offsets(self, head_slot: jax.Array) -> jax.Array:
        """(num_agents, max_length) steps behind the head for each slot; 0 = head.

        A slot holds a live segment iff its offset is < the snake's length.
        """
        slots = jnp.arange(self._config.max_length)
        return (head_slot[:, None] - slots[None, :]) % self._config.max_length

    def _owner_grid(
        self, body: jax.Array, head_slot: jax.Array, length: jax.Array
    ) -> jax.Array:
        """Scatter the ring buffers into a grid of agent index + 1 (0 = empty)."""
        alive = self._segment_offsets(head_slot) < length[:, None]
        owner = jnp.where(alive, jnp.arange(self.num_agents)[:, None] + 1, 0)
        return (
            jnp.zeros((self.padded_width, self.padded_height), dtype=jnp.int32)
            .at[body[..., 0], body[..., 1]]
            .max(owner)
        )

    def _make_action_mask(self, direction: jax.Array):
        opposite = (direction + 2) % 4
        return self._base_action_mask.at[jnp.arange(self.num_agents), opposite].set(
            False
        )

    def reset(self, rng_key: jax.Array) -> tuple[SnakeState, TimeStep]:
        decor_key, spawn_key, dir_key, food_key = jax.random.split(rng_key, 4)

        n = self.num_agents
        cfg = self._config

        tiles = generate_decor_tiles(
            self.unpadded_width, self.unpadded_height, self._obs_vocab, decor_key
        )
        tiles = jnp.pad(
            tiles,
            pad_width=(
                (self.pad_width, self.pad_width),
                (self.pad_height, self.pad_height),
            ),
            mode="constant",
            constant_values=self._tile_wall,
        )

        head_pos = self._sample_open_cells(tiles != self._tile_wall, n, spawn_key)
        direction = jax.random.randint(dir_key, (n,), minval=0, maxval=4)
        length = jnp.full(n, cfg.initial_length, dtype=jnp.int32)

        # Every slot starts at the spawn cell: the body "grows out" of it.
        body = jnp.broadcast_to(head_pos[:, None, :], (n, cfg.max_length, 2))
        head_slot = jnp.zeros(n, dtype=jnp.int32)

        occupied = (
            jnp.zeros((self.padded_width, self.padded_height), dtype=jnp.bool_)
            .at[head_pos[:, 0], head_pos[:, 1]]
            .set(True)
        )
        open_for_food = (tiles != self._tile_wall) & ~occupied
        food_cells = self._sample_open_cells(open_for_food, cfg.initial_food, food_key)
        food = jnp.zeros((self.padded_width, self.padded_height), dtype=jnp.bool_)
        food = food.at[food_cells[:, 0], food_cells[:, 1]].set(True)

        state = SnakeState(
            body=body,
            head_slot=head_slot,
            head_pos=head_pos,
            direction=direction,
            length=length,
            food=food,
            tiles=tiles,
            time=jnp.int32(0),
            rewards=jnp.float32(0.0),
            deaths=jnp.float32(0.0),
        )

        actions = jnp.zeros((n,), dtype=jnp.uint16)
        rewards = jnp.zeros((n,), dtype=jnp.float32)
        terminated = jnp.zeros((n,), dtype=jnp.bool_)

        return state, self.encode_observations(state, actions, rewards, terminated)

    def step(
        self, state: SnakeState, action: jax.Array, rng_key: jax.Array
    ) -> tuple[SnakeState, TimeStep]:
        spawn_key, dir_key, drop_key, drop_cell_key = jax.random.split(rng_key, 4)

        n = self.num_agents
        cfg = self._config
        agent_idx = jnp.arange(n)

        # Reversing (or a non-move action) keeps the snake moving straight.
        opposite = (state.direction + 2) % 4
        is_move = (action >= 0) & (action < 4) & (action != opposite)
        heading = jnp.where(is_move, action, state.direction)
        new_head = state.head_pos + DIRECTIONS[heading]
        hx, hy = new_head[:, 0], new_head[:, 1]

        eats = state.food[hx, hy]  # provisional: dying snakes don't consume, see below
        grows = eats & (state.length < cfg.max_length)

        # Occupancy after tails vacate: every live segment except the tail of
        # a non-growing snake.
        offsets = self._segment_offsets(state.head_slot)
        alive = offsets < state.length[:, None]
        stays = alive & ((offsets < state.length[:, None] - 1) | grows[:, None])
        xs, ys = state.body[..., 0], state.body[..., 1]
        occupied = (
            jnp.zeros((self.padded_width, self.padded_height), dtype=jnp.bool_)
            .at[xs, ys]
            .max(stays)
        )

        # Deaths: walls, any post-tail body (own, other, or same-step corpse),
        # two heads on one cell, or two heads passing through each other.
        hit_wall = state.tiles[hx, hy] == self._tile_wall
        hit_body = occupied[hx, hy]
        not_self = ~jnp.eye(n, dtype=jnp.bool_)
        same_target = jnp.all(new_head[:, None] == new_head[None, :], axis=-1)
        into_head = jnp.all(new_head[:, None] == state.head_pos[None, :], axis=-1)
        head_on = (same_target | (into_head & into_head.T)) & not_self
        died = hit_wall | hit_body | jnp.any(head_on, axis=1)

        eats = eats & ~died
        grows = grows & ~died
        eaten = jnp.zeros_like(state.food).at[hx, hy].max(eats)
        food = state.food & ~eaten

        # Half of a dead snake's body (alternating segments) turns into food.
        countdown = state.length[:, None] - offsets  # steps until the cell vacates
        corpse = alive & (countdown % 2 == 1) & died[:, None]
        food = food.at[xs, ys].max(corpse)

        time_up = jnp.equal(state.time + 1, self._length)
        respawned = died | time_up

        # Respawn on open cells, avoiding the survivors' bodies and new heads.
        survivor_stays = stays & ~respawned[:, None]
        occ = (
            jnp.zeros((self.padded_width, self.padded_height), dtype=jnp.bool_)
            .at[xs, ys]
            .max(survivor_stays)
            .at[hx, hy]
            .max(~respawned)
        )
        spawn_open = (state.tiles != self._tile_wall) & ~occ & ~food
        spawn_pos = self._sample_open_cells(spawn_open, n, spawn_key)

        head_pos = jnp.where(respawned[:, None], spawn_pos, new_head)
        direction = jnp.where(
            respawned, jax.random.randint(dir_key, (n,), minval=0, maxval=4), heading
        )
        length = jnp.where(respawned, cfg.initial_length, state.length + grows)

        # Advance the ring: write the new head one slot forward; the tail
        # drops out of the length window on its own. Respawned snakes get
        # every slot set to the spawn cell, so the body "grows out" of it.
        head_slot = (state.head_slot + 1) % cfg.max_length
        body = state.body.at[agent_idx, head_slot].set(new_head)
        body = jnp.where(respawned[:, None, None], spawn_pos[:, None, :], body)
        head_slot = jnp.where(respawned, 0, head_slot)

        # Random food drop on a uniformly chosen open cell.
        occupied_now = occ.at[head_pos[:, 0], head_pos[:, 1]].set(True)
        drop_open = (state.tiles != self._tile_wall) & ~occupied_now & ~food
        scores = jnp.where(
            drop_open.reshape(-1),
            jax.random.uniform(drop_cell_key, (drop_open.size,)),
            -1.0,
        )
        drop_idx = jnp.argmax(scores)
        drop = jax.random.bernoulli(drop_key, cfg.food_spawn_prob) & (
            scores[drop_idx] >= 0.0
        )
        dx, dy = jnp.unravel_index(drop_idx, drop_open.shape)
        food = food.at[dx, dy].max(drop)

        rewards = (
            eats.astype(jnp.float32) * cfg.food_reward
            + died.astype(jnp.float32) * cfg.death_reward
        )
        terminated = died | time_up

        state = SnakeState(
            body=body,
            head_slot=head_slot,
            head_pos=head_pos,
            direction=direction,
            length=length,
            food=food,
            tiles=state.tiles,
            time=state.time + 1,
            rewards=state.rewards + jnp.mean(rewards),
            deaths=state.deaths + jnp.sum(died).astype(jnp.float32),
        )

        return state, self.encode_observations(state, action, rewards, terminated)

    def _render_channels(self, state: SnakeState):
        owner = self._owner_grid(state.body, state.head_slot, state.length)

        tiles = jnp.where(state.food, jnp.uint16(self._tile_food), state.tiles)
        tiles = jnp.where(owner > 0, jnp.uint16(self._agent_snake_body), tiles)
        tiles = tiles.at[state.head_pos[:, 0], state.head_pos[:, 1]].set(
            jnp.uint16(self._agent_snake_head)
        )

        directions = jnp.zeros_like(state.tiles)
        directions = directions.at[state.head_pos[:, 0], state.head_pos[:, 1]].set(
            (state.direction + 1).astype(jnp.uint16)
        )

        return tiles, directions, owner

    def encode_observations(
        self, state: SnakeState, actions, rewards, terminated
    ) -> TimeStep:
        tiles, directions, owner = self._render_channels(state)
        health = jnp.zeros_like(tiles)

        # Owner ids ride along in the team channel and are remapped to the
        # egocentric 1 = self / 2 = other after the crop, so the per-agent
        # work is view-sized instead of map-sized.
        full = jnp.stack(
            (tiles, directions, owner.astype(jnp.uint16), health), axis=-1
        )

        def _encode_view(idx, pos):
            crop = jax.lax.dynamic_slice(
                full,
                (
                    pos[0] - self.view_width // 2,
                    pos[1] - self.view_height // 2,
                    0,
                ),
                (
                    self.view_width,
                    self.view_height,
                    self.observation_spec.shape[-1],
                ),
            )
            cell_owner = crop[..., 2]
            team = jnp.where(
                cell_owner == idx + 1,
                jnp.uint16(1),
                jnp.where(cell_owner > 0, jnp.uint16(2), jnp.uint16(0)),
            )
            return crop.at[..., 2].set(team)

        view = jax.vmap(_encode_view)(jnp.arange(self.num_agents), state.head_pos)

        time = jnp.repeat(state.time[None], self.num_agents, axis=0)

        return TimeStep(
            obs=view,
            time=time,
            last_action=jnp.asarray(actions, dtype=jnp.uint16),
            reward=rewards,
            action_mask=self._make_action_mask(state.direction),
            terminated=terminated,
        )

    @cached_property
    def observation_spec(self) -> ObservationSpec:
        return make_obs_spec(self.view_width, self.view_height, len(self._obs_vocab))

    @cached_property
    def action_spec(self) -> DiscreteActionSpec:
        return DiscreteActionSpec(n=len(self._action_vocab))

    @property
    def num_agents(self) -> int:
        return self._config.num_agents

    @property
    def obs_vocab(self) -> Vocabulary:
        return self._obs_vocab

    @property
    def action_vocab(self) -> Vocabulary:
        return self._action_vocab

    def create_placeholder_logs(self):
        return {"rewards": jnp.float32(0.0), "deaths": jnp.float32(0.0)}

    def create_logs(self, state: SnakeState):
        return {"rewards": state.rewards, "deaths": state.deaths}

    def get_render_state(self, state: SnakeState) -> GridRenderState:
        tiles, directions, _ = self._render_channels(state)
        zeros = jnp.zeros_like(tiles)
        tilemap = jnp.stack((tiles, directions, zeros, zeros), axis=-1)
        return GridRenderState(
            tilemap=tilemap,
            agent_positions=state.head_pos,
        )

    def get_render_settings(self) -> GridRenderSettings:
        return GridRenderSettings(
            obs_vocab=self._obs_vocab,
            tile_width=self.unpadded_width,
            tile_height=self.unpadded_height,
            view_width=self.view_width,
            view_height=self.view_height,
        )
