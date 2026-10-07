"""The rust env behind numpy buffers.

Owns the pyo3 env and the host buffers it fills in place: reset/step are plain
host calls that mutate those buffers and hand them back as a TimeStep, no JAX
involved. `rust_env_jax.RustEnvJax` wraps this with io_callbacks for jit.
"""

import json
from functools import cached_property
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from mapox._core import Env as _CoreEnv
from mapox.envs.common import make_obs_spec
from mapox.renderer import GridRenderSettings, GridRenderState
from mapox.specs import DiscreteActionSpec, ObservationSpec
from mapox.timestep import TimeStep
from mapox.vocab import Vocabulary


class RustFindReturnConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    env_type: Literal["rust_find_return"] = "rust_find_return"

    num_agents: int = 8
    num_flags: int = 1

    width: int = 80
    height: int = 70
    view_width: int = 15
    view_height: int = 15

    mapgen_threshold: float = 0.07
    water_threshold: float = -0.45
    digging_timeout: int = 5
    preparation_steps: int = 256
    treasure_reward: float = 1.0
    pipes_enabled: bool = True


class RustScoutsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    env_type: Literal["rust_scouts"] = "rust_scouts"

    num_scouts: int = 1
    num_harvesters: int = 1
    num_treasures: int = 12

    width: int = 80
    height: int = 70
    view_width: int = 15
    view_height: int = 15

    mapgen_threshold: float = 0.07
    water_threshold: float = -0.45
    # a harvester acts on one step in this many; the rest are spent standing
    harvesters_move_every: int = 6

    scout_reward: float = 1.0
    harvester_reward: float = 1.0


class RustVideoConfig(BaseModel):
    """Records the wrapped env's render view to mp4 clips with ffmpeg.

    Wraps outside a vector/multi env so there is one encoder per training run,
    not per instance. One post-step frame per recorded step; fps controls
    playback, not training speed.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    env_type: Literal["rust_video"] = "rust_video"

    env: RustEnvConfig = Field(discriminator="env_type")
    # a separate directory per run/worker; existing videos are never overwritten
    output_dir: str = "videos"
    # number of step calls (frames) in each clip
    record_steps: int = 256
    # start-to-start spacing, at least record_steps; None records only one clip
    interval_steps: int | None = 10_000
    # zero-based step index of the first frame: 0 records immediately after
    # the first step
    start_step: int = 0
    fps: int = 30
    # output pixels; both dimensions must be positive and even for
    # H.264/yuv420p
    width: int = 640
    height: int = 480
    # x264 CRF quality: 0 (lossless) to 51; higher compresses more
    crf: int = 23


class RustSnakeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    env_type: Literal["rust_snake"] = "rust_snake"

    num_agents: int = 16
    width: int = 80
    height: int = 70
    view_width: int = 15
    view_height: int = 15

    # every open tile independently grows a pellet with this probability
    # each step; snakes always start as a single cell on a bare board
    food_spawn_prob: float = 0.0001
    food_reward: float = 0.01
    death_reward: float = 0.0


class RustPacmanConfig(BaseModel):
    """Pac-Man on the classic maze. The maze is fixed, so there are no size
    keys; the view wraps through the side tunnel the way Pac-Man does."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    env_type: Literal["rust_pacman"] = "rust_pacman"

    view_width: int = 15
    view_height: int = 15

    # start each round on a random pellet instead of the classic spawn
    randomize_starting_position: bool = False
    # each ghost waits uniform [min, max) steps before its first move
    min_start_timeout: int = 0
    max_start_timeout: int = 49
    # steps a power pellet keeps the ghosts frightened
    frightened_time: int = 35
    # scatter/chase flips before the ghosts chase for good
    max_mode_changes: int = 6
    scatter_mode_length: int = 70
    chase_mode_length: int = 140

    pellet_reward: float = 1.0
    ghost_reward: float = 1.0
    death_reward: float = 0.0
    # paid on top of the last pellet's reward when the maze is cleared
    clear_reward: float = 0.0


class RustSurvivalConfig(BaseModel):
    """A first pass at a survival game: health, hunger, a two-slot inventory
    and crafting, rewarded for staying alive and well; see the rust env's
    docs for the rules."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    env_type: Literal["rust_survival"] = "rust_survival"

    num_agents: int = 8
    width: int = 80
    height: int = 70
    view_width: int = 15
    view_height: int = 15

    # shares of the map under water and rock: the low and high ground
    water_fraction: float = 0.10
    rock_fraction: float = 0.15
    # shares of the land between that are forest (trees, sticks) and scrub
    # (stones): its wettest and driest ground. The rest is meadow (berries,
    # tall grass, buried carrots).
    forest_fraction: float = 0.25
    scrub_fraction: float = 0.25

    # stats run 0..150
    start_hunger: int = 100
    start_health: int = 150
    start_temperature: int = 150
    # hunger drops by one every this many steps
    hunger_interval: int = 2
    # health lost each step spent at zero hunger
    starve_damage: int = 1
    # health grows by one every regen_interval steps while hunger is at
    # least regen_threshold
    regen_threshold: int = 100
    regen_interval: int = 4

    # the episode's last steps are winter: bushes and tall grass die, water
    # freezes into walkable ice, and nothing regrows
    winter_length: int = 300
    # in winter, temperature drops by one every chill_interval steps spent
    # off firelit ground; firelit ground warms by fire_warmth a step, any
    # season; at zero temperature health drops by freeze_damage a step
    chill_interval: int = 1
    fire_warmth: int = 5
    freeze_damage: int = 2

    berry_food: int = 20
    cooked_berry_food: int = 50
    # carrots are buried at reset, grabbed straight out of the ground, and
    # never grow back; held to a fire, a berry or carrot cooks at once
    carrot_food: int = 35
    cooked_carrot_food: int = 70
    bush_regrow_steps: int = 200
    # steps felling a tree takes, the axe use included; the agent can only
    # wait meanwhile, and a log drops at the end
    chop_steps: int = 6
    # steps clearing a bush (ripe, bare or dead) with the axe takes, likewise;
    # a stick drops at the end
    clear_bush_steps: int = 3
    # steps harvesting tall grass takes, the empty-handed use that starts it
    # included; a bundle of grass drops at the end
    harvest_steps: int = 3
    fire_burn_steps: int = 200
    # the last of those steps it burns low; wood stokes a low fire back up
    fire_low_steps: int = 50
    fire_light_radius: int = 4
    # a torch (stick + grass) lights torch_light_radius around whoever holds
    # it, without warming, and is gone after torch_burn_steps held
    torch_burn_steps: int = 100
    torch_light_radius: int = 3

    # steps of day then night in each cycle, starting at dawn
    day_length: int = 200
    night_length: int = 100
    # at night agents see this far, and lit ground as far as by day
    night_vision_radius: int = 2
    # the last steps of each day, over which sight closes in to the night's
    dusk_length: int = 50

    # nests in the forest, each hatching a spider every nightfall; spiders
    # hunt agents in the dark until dawn and never step on lit ground
    num_spider_eggs: int = 4
    spider_damage: int = 15
    # how far, walking, a spider tracks an agent
    spider_hunt_radius: int = 12

    # paid every step an agent ends alive
    alive_reward: float = 0.01
    # taken every step, each scaled by the stat's shortfall: health_penalty in
    # full at zero health and none at 150; hunger_penalty in full at zero
    # hunger and none from hunger_threshold up; temperature likewise
    health_penalty: float = 0.01
    hunger_penalty: float = 0.01
    hunger_threshold: int = 50
    temperature_penalty: float = 0.01
    temperature_threshold: int = 50


class RustVecConfig(BaseModel):
    """Vectorized copies of one rust env; the rust side steps them in parallel."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    env_type: Literal["rust_vec"] = "rust_vec"

    num: int = 1
    env: RustEnvConfig = Field(discriminator="env_type")


class RustMultiEnvSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str
    num: int = 1
    env: RustEnvConfig = Field(discriminator="env_type")


class RustMultiConfig(BaseModel):
    """A batch of rust envs; the rust MultitaskWrapper is the vectorizer."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    env_type: Literal["rust_multi"] = "rust_multi"

    envs: tuple[RustMultiEnvSpec, ...]

    @field_validator("envs", mode="before")
    @classmethod
    def coerce_envs(cls, v):
        # JSON gives list; accept list and turn into tuple
        return tuple(v) if isinstance(v, list) else v


type RustEnvConfig = (
    RustFindReturnConfig
    | RustScoutsConfig
    | RustSnakeConfig
    | RustPacmanConfig
    | RustSurvivalConfig
    | RustVideoConfig
    | RustVecConfig
    | RustMultiConfig
)

# the wrapper configs refer to RustEnvConfig, which includes them: the
# alias above is what makes the forward references resolvable
RustVecConfig.model_rebuild()
RustMultiEnvSpec.model_rebuild()
RustMultiConfig.model_rebuild()


class RustEnvNumpy:
    def __init__(self, config: RustEnvConfig, length: int):
        self._inner = _CoreEnv(config.model_dump_json(), length)

        _, view_width, view_height, channels = self._inner.observation_shape
        self._view_width = view_width
        self._view_height = view_height
        self._obs_shape = (view_width, view_height, channels)

        self._obs_vocab = Vocabulary(self._inner.obs_symbols).freeze()
        self._action_vocab = Vocabulary(self._inner.action_symbols).freeze()
        self._render_settings = self._read_render_settings()

        self._bind_buffers()

    def _read_render_settings(self) -> GridRenderSettings:
        """The current env's map and window layout.

        Read on every mode switch: a multitask wrapper lends its selected
        task's map, not the first task's.
        """
        tile_width, tile_height, view_width, view_height, ui_height = (
            self._inner.render_settings
        )
        return GridRenderSettings(
            obs_vocab=self._obs_vocab,
            tile_width=tile_width,
            tile_height=tile_height,
            view_width=view_width,
            view_height=view_height,
            ui_height=ui_height,
        )

    def _bind_buffers(self) -> None:
        """Size the shared buffers for the env's current agent count.

        Rust reads and writes exactly `num_agents` rows, and `set_enjoy_mode`
        narrows that to one copy of one task, so the buffers are rebound along
        with the mode — the same order the rust `RenderApp` allocates in.
        Whole-batch buffers under a mode are read past the end of the shorter
        array (the vocab wrapper's mask fill is where it trips).
        """
        num_agents = self._inner.num_agents
        num_actions = self._inner.num_actions

        self._obs = np.zeros((num_agents, *self._obs_shape), np.uint16)
        self._time = np.zeros((num_agents,), np.int32)
        self._terminated = np.zeros((num_agents,), np.bool_)
        self._last_action = np.zeros((num_agents,), np.uint16)
        self._reward = np.zeros((num_agents,), np.float32)
        self._action_mask = np.zeros((num_agents, num_actions), np.bool_)
        self._task_ids = np.zeros((num_agents,), np.int32)

        # rust writes into these buffers in place and they never change
        # identity, so the timestep viewing them is built once
        self._timestep = TimeStep(
            # Rust and JAX share uint16 buffers for observations and actions.
            obs=self._obs,
            time=self._time,
            terminated=self._terminated,
            last_action=self._last_action,
            reward=self._reward,
            action_mask=self._action_mask,
            task_ids=self._task_ids,
        )

    @property
    def inner(self) -> _CoreEnv:
        """The raw pyo3 env, for host-side drivers like `mapox._core.enjoy`."""
        return self._inner

    @property
    def buffers(self) -> TimeStep:
        """The shared numpy buffers: the same arrays on every call."""
        return self._timestep

    def reset(self, seed: int) -> TimeStep:
        self._inner.reset(
            int(seed),
            self._obs,
            self._time,
            self._terminated,
            self._last_action,
            self._reward,
            self._action_mask,
            self._task_ids,
        )
        return self._timestep

    def step(self, action: np.ndarray) -> TimeStep:
        if np.any(action < 0) or np.any(action > np.iinfo(np.uint16).max):
            raise ValueError("action id outside the Rust VocabId range")
        self._inner.step(
            np.ascontiguousarray(action, dtype=np.uint16),
            self._obs,
            self._time,
            self._terminated,
            self._last_action,
            self._reward,
            self._action_mask,
            self._task_ids,
        )
        return self._timestep

    @cached_property
    def observation_spec(self) -> ObservationSpec:
        # rust emits the leading OBS_CHANNELS of the canonical 4-channel
        # layout (currently just tile ids), so truncate the canonical spec
        # to the channel count the env actually produces
        full = make_obs_spec(self._view_width, self._view_height, len(self._obs_vocab))
        channels = self._obs.shape[-1]
        if channels > full.shape[-1]:
            raise ValueError(
                f"rust env produces {channels} obs channels, more than the "
                f"{full.shape[-1]} the canonical layout defines"
            )
        assert isinstance(full.max_value, tuple)
        return full._replace(
            shape=(*full.shape[:-1], channels),
            max_value=full.max_value[:channels],
        )

    @cached_property
    def action_spec(self) -> DiscreteActionSpec:
        return DiscreteActionSpec(n=self._inner.num_actions)

    def get_render_settings(self) -> GridRenderSettings:
        return self._render_settings

    def get_render_state(self, state: None = None) -> GridRenderState:
        """The full tile map and agent positions, straight from the rust env.

        The rust env owns its state, so `state` exists for interface symmetry
        with the Python envs and is ignored. Unlike the Python envs' states,
        the map is the unpadded interior, and it already includes the agents.
        """
        tilemap, agent_positions = self._inner.render_state()
        return GridRenderState(tilemap=tilemap, agent_positions=agent_positions)

    @property
    def num_agents(self) -> int:
        return self._inner.num_agents

    @property
    def num_tasks(self) -> int:
        return self._inner.num_tasks

    @property
    def task_names(self) -> list[str]:
        return self._inner.task_names

    @property
    def obs_vocab(self) -> Vocabulary:
        return self._obs_vocab

    @property
    def action_vocab(self) -> Vocabulary:
        return self._action_vocab

    def set_enjoy_mode(self, task_id: int | None) -> None:
        """Restrict stepping to one task, rebinding the buffers to its agents.

        Nothing else changes: the vocabularies are the wrapper's union ones
        either way, and `None` restores the whole batch. Everything the mode
        selects — agent count, map and window layout — is re-read here rather
        than kept from construction.
        """
        self._inner.set_enjoy_mode(task_id)
        self._bind_buffers()
        self._render_settings = self._read_render_settings()

    def consume_metrics(self) -> dict[str, Any]:
        return json.loads(self._inner.consume_metrics())
