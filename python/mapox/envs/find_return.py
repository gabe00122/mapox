from functools import cached_property, partial
from typing import Literal, NamedTuple

import jax
from jax import numpy as jnp
from pydantic import BaseModel, ConfigDict

import mapox.symbols as SB
from mapox.environment import Environment
from mapox.envs.common import DIRECTIONS, make_action_mask, make_obs_spec
from mapox.map_generator import (
    choose_positions,
    fractal_noise,
    generate_decor_tiles,
    register_decor_tiles,
)
from mapox.map_loader import load_map
from mapox.renderer import GridRenderSettings, GridRenderState
from mapox.specs import DiscreteActionSpec, ObservationSpec
from mapox.timestep import TimeStep
from mapox.vocab import Vocabulary


class FindReturnConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    env_type: Literal["find_return"] = "find_return"

    num_agents: int = 1
    num_flags: int = 1

    width: int = 40
    height: int = 40
    view_width: int = 11
    view_height: int = 11

    mapgen_threshold: float = 0.3
    digging_timeout: int = 5
    treasure_reward: float = 1.0

    map_path: str | None = None


class FindReturnState(NamedTuple):
    agents_pos: jax.Array
    agents_timeout: jax.Array
    found_reward: jax.Array

    time: jax.Array

    map: jax.Array
    spawn_pos: jax.Array
    spawn_count: jax.Array

    rewards: jax.Array


class FindReturnEnv(Environment[FindReturnState]):
    def __init__(self, config: FindReturnConfig, length: int) -> None:
        super().__init__()

        self._config = config
        self._length = length

        self._obs_vocab = Vocabulary()
        self._action_vocab = Vocabulary()

        register_decor_tiles(self._obs_vocab)
        self._tile_empty = self._obs_vocab.add(SB.TILE_EMPTY)
        self._tile_destructible_wall = self._obs_vocab.add(SB.TILE_DESTRUCTIBLE_WALL)
        self._tile_wall = self._obs_vocab.add(SB.TILE_WALL)
        self._tile_flag = self._obs_vocab.add(SB.TILE_FLAG)
        self._agent_generic = self._obs_vocab.add(SB.AGENT_GENERIC)

        self.view_width = config.view_width
        self.view_height = config.view_height
        self.pad_width = self.view_width // 2
        self.pad_height = self.view_height // 2

        self.digging_timeout = config.digging_timeout
        self.treasure_reward = config.treasure_reward

        self._num_agents = config.num_agents
        self.num_flags = config.num_flags
        self.unpadded_width = config.width
        self.unpadded_height = config.height
        self.mapgen_threshold = config.mapgen_threshold

        if config.map_path is not None:
            self._loaded_tiles = load_map(config.map_path, self._obs_vocab)
            self.unpadded_width = self._loaded_tiles.shape[0]
            self.unpadded_height = self._loaded_tiles.shape[1]
        else:
            self._loaded_tiles = None

        self.width = self.unpadded_width + self.pad_width
        self.height = self.unpadded_height + self.pad_height

        direction_actions = self._action_vocab.add_block(
            [
                SB.MOVE_UP,
                SB.MOVE_RIGHT,
                SB.MOVE_DOWN,
                SB.MOVE_LEFT,
            ]
        )
        self._action_mask = make_action_mask(
            list(direction_actions), len(self._action_vocab), self.num_agents
        )

        self._action_vocab.freeze()
        self._obs_vocab.freeze()

    def _generate_map(self, rng_key):
        walls_key, decor_key = jax.random.split(rng_key, 2)
        noise = fractal_noise(
            self.unpadded_width, self.unpadded_height, [2, 4, 5, 8, 10], walls_key
        )

        tiles = generate_decor_tiles(
            self.unpadded_width, self.unpadded_height, self.obs_vocab, decor_key
        )
        tiles = jnp.where(noise > 0.05, jnp.uint16(self._tile_destructible_wall), tiles)

        # get the empty tiles for spawning
        x_spawns, y_spawns = jnp.where(
            tiles == self._tile_empty,
            size=self.unpadded_width * self.unpadded_height,
            fill_value=jnp.int8(-1),
        )
        spawn_count = jnp.sum(tiles == self._tile_empty)

        # pad the tiles
        tiles = jnp.pad(
            tiles,
            pad_width=(
                (self.pad_width, self.pad_width),
                (self.pad_height, self.pad_height),
            ),
            mode="constant",
            constant_values=self._tile_wall,
        )

        # pad the empty tiles
        y_spawns = y_spawns + self.pad_height
        x_spawns = x_spawns + self.pad_width
        spawn_pos = jnp.stack((x_spawns, y_spawns), axis=1)

        return tiles, spawn_pos, spawn_count

    def reset(self, rng_key: jax.Array) -> tuple[FindReturnState, TimeStep]:
        map_key, pos_key = jax.random.split(rng_key)

        if self._loaded_tiles is not None:
            # sprinkle decor tiles on empty cells
            unpadded_map = self._loaded_tiles
            decor = generate_decor_tiles(
                self.unpadded_width, self.unpadded_height, self.obs_vocab, map_key
            )
            unpadded_map = jnp.where(
                unpadded_map == self._tile_empty, decor, unpadded_map
            )

            # pad with walls
            map = jnp.pad(
                unpadded_map,
                pad_width=(
                    (self.pad_width, self.pad_width),
                    (self.pad_height, self.pad_height),
                ),
                mode="constant",
                constant_values=self._tile_wall,
            )

            # compute spawn positions after decor
            x_spawns, y_spawns = jnp.where(
                unpadded_map == self._tile_empty,
                size=self.unpadded_width * self.unpadded_height,
                fill_value=jnp.int8(-1),
            )
            spawn_count = jnp.sum(unpadded_map == self._tile_empty)
            x_spawns = x_spawns + self.pad_width
            y_spawns = y_spawns + self.pad_height
            spawn_pos = jnp.stack((x_spawns, y_spawns), axis=1)

            pos_x, pos_y = choose_positions(
                unpadded_map,
                self.num_agents,
                self._tile_empty,
                pos_key,
                replace=False,
            )

            pos_x = pos_x + self.pad_width
            pos_y = pos_y + self.pad_height
            agents_pos = jnp.stack((pos_x, pos_y), axis=1)
        else:
            map, spawn_pos, spawn_count = self._generate_map(map_key)

            unpadded_map = map[
                self.pad_width : -self.pad_width, self.pad_height : -self.pad_height
            ]

            pos_x, pos_y = choose_positions(
                unpadded_map,
                self.num_flags + self.num_agents,
                self._tile_empty,
                pos_key,
                replace=False,
            )

            pos_x = pos_x + self.pad_width
            pos_y = pos_y + self.pad_height
            positions = jnp.stack((pos_x, pos_y), axis=1)
            flag_pos = positions[: self.num_flags]
            agents_pos = positions[self.num_flags :]

            map = map.at[flag_pos[:, 0], flag_pos[:, 1]].set(self._tile_flag)

        state = FindReturnState(
            map=map,
            spawn_pos=spawn_pos,
            spawn_count=spawn_count,
            agents_pos=agents_pos,
            agents_timeout=jnp.zeros((self.num_agents,), dtype=jnp.int32),
            found_reward=jnp.zeros((self.num_agents,), dtype=jnp.bool_),
            time=jnp.int32(0),
            rewards=jnp.float32(0.0),
        )

        actions = jnp.zeros((self.num_agents,), dtype=jnp.uint16)
        rewards = jnp.zeros((self.num_agents,), dtype=jnp.float32)

        return state, self.encode_observations(state, actions, rewards)

    @cached_property
    def observation_spec(self) -> ObservationSpec:
        return make_obs_spec(self.view_width, self.view_height, len(self.obs_vocab))

    @cached_property
    def action_spec(self) -> DiscreteActionSpec:
        return DiscreteActionSpec(n=len(self.action_vocab))

    @property
    def num_agents(self) -> int:
        return self._num_agents

    @property
    def num_tasks(self) -> int:
        return 1

    def step(
        self, state: FindReturnState, action: jax.Array, rng_key: jax.Array
    ) -> tuple[FindReturnState, TimeStep]:
        @partial(jax.vmap, in_axes=(0, 0, 0, 0), out_axes=(0, 0, 0, 0))
        def _step_agent(local_position, timeout, local_action, random_position):
            def _step_timeout(local_position, timeout, local_action, random_position):
                return local_position, local_position, timeout - 1, 0.0

            def _step_move(local_position, timeout, local_action, random_position):
                target_pos = local_position + DIRECTIONS[local_action]

                new_tile = state.map[target_pos[0], target_pos[1]]

                # don't move if we are moving into a wall
                new_pos = jnp.where(
                    jnp.logical_or(
                        new_tile == self._tile_wall,
                        new_tile == self._tile_destructible_wall,
                    ),
                    local_position,
                    target_pos,
                )

                found_treasure = new_tile == self._tile_flag
                reward = jnp.where(found_treasure, self.treasure_reward, 0.0)

                # randomize position if the agent finds the reward
                new_pos = jnp.where(found_treasure, random_position, new_pos)

                # sets a timeout of the tile is dug
                timeout = jnp.where(
                    new_tile == self._tile_destructible_wall, self.digging_timeout, 0
                )

                return new_pos, target_pos, timeout, reward

            return jax.lax.cond(
                timeout > 0,
                _step_timeout,
                _step_move,
                local_position,
                timeout,
                local_action,
                random_position,
            )

        random_positions = state.spawn_pos[
            jax.random.randint(
                rng_key, (self._num_agents,), minval=0, maxval=state.spawn_count
            )
        ]
        new_position, target_pos, timeout, rewards = _step_agent(
            state.agents_pos, state.agents_timeout, action, random_positions
        )

        # dig actions
        target_tiles = state.map[target_pos[:, 0], target_pos[:, 1]]
        map = state.map.at[target_pos[:, 0], target_pos[:, 1]].set(
            jnp.where(
                target_tiles == self._tile_destructible_wall,
                self._tile_empty,
                target_tiles,
            )
        )
        # /dig actions

        state = state._replace(
            agents_pos=new_position,
            agents_timeout=timeout,
            found_reward=jnp.logical_or(state.found_reward, rewards),
            time=state.time + 1,
            rewards=state.rewards + jnp.mean(rewards),
            map=map,
        )

        return state, self.encode_observations(state, action, rewards)

    def _render_tiles(self, state: FindReturnState):
        tiles = state.map
        tiles = tiles.at[state.agents_pos[:, 0], state.agents_pos[:, 1]].set(
            self._agent_generic
        )

        directions = jnp.zeros_like(tiles)
        teams = jnp.zeros_like(tiles)
        health = jnp.zeros_like(tiles)

        return jnp.concatenate(
            (
                tiles[..., None],
                directions[..., None],
                teams[..., None],
                health[..., None],
            ),
            axis=-1,
        )

    def encode_observations(self, state: FindReturnState, actions, rewards) -> TimeStep:
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
            last_action=jnp.asarray(actions, dtype=jnp.uint16),
            reward=rewards,
            action_mask=self._action_mask,
            terminated=jnp.equal(time, self._length - 1),
        )

    def create_placeholder_logs(self):
        return {"rewards": jnp.float32(0.0)}

    def create_logs(self, state: FindReturnState):
        return {"rewards": state.rewards}

    def get_render_state(self, state: FindReturnState) -> GridRenderState:
        tiles = self._render_tiles(state)

        return GridRenderState(
            tilemap=tiles,
            agent_positions=state.agents_pos,
        )

    def get_render_settings(self) -> GridRenderSettings:
        return GridRenderSettings(
            obs_vocab=self.obs_vocab,
            tile_width=self.unpadded_width,
            tile_height=self.unpadded_height,
            view_width=self.view_width,
            view_height=self.view_height,
        )

    @property
    def obs_vocab(self) -> Vocabulary:
        return self._obs_vocab

    @property
    def action_vocab(self) -> Vocabulary:
        return self._action_vocab
