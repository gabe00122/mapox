from functools import partial

import jax
import numpy as np
import pytest
from jax import numpy as jnp
from mapox._core import Env
from mapox.config import EnvironmentFactory
from mapox.envs.rust_env_jax import RustEnvJax
from mapox.envs.rust_env_numpy import (
    RustEnvNumpy,
    RustFindReturnConfig,
    RustMultiConfig,
    RustMultiEnvSpec,
    RustScoutsConfig,
    RustVecConfig,
)

# Full-field dumps, as RustScoutsConfig/RustFindReturnConfig model_dump_json()
# produce them: the rust side parses with serde and has no field defaults.
SCOUTS_JSON = (
    '{"env_type": "rust_scouts", "num_scouts": 1, "num_harvesters": 1, '
    '"num_treasures": 12, "width": 12, "height": 12, "view_width": 11, "view_height": 13, '
    '"ui_height": 2, "mapgen_threshold": 0.3, "water_threshold": -0.45, '
    '"harvesters_move_every": 6, "scout_reward": 1.0, "harvester_reward": 1.0}'
)
FR_JSON = (
    '{"env_type": "rust_find_return", "num_agents": 2, "num_flags": 1, '
    '"width": 12, "height": 12, "view_width": 11, "view_height": 11, '
    '"mapgen_threshold": 0.3, "water_threshold": -0.45, "digging_timeout": 5, '
    '"preparation_steps": 256, "treasure_reward": 1.0}'
)
MULTI_JSON = (
    '{"env_type": "rust_multi", "envs": ['
    '{"name": "scouts", "num": 2, "env": ' + SCOUTS_JSON + "},"
    '{"name": "fr", "num": 1, "env": ' + FR_JSON + "}]}"
)
VEC_JSON = '{"env_type": "rust_vec", "num": 3, "env": ' + SCOUTS_JSON + "}"


def test_multitask_json_builds_env_with_task_ids():
    env = Env(MULTI_JSON, 32)
    assert env.num_agents == 6  # 2 * (1 + 1) + 2
    assert env.task_names == ["scouts", "fr"]

    num_agents, w, h, c = env.observation_shape
    obs = np.zeros((num_agents, w, h, c), np.uint16)
    time = np.zeros((num_agents,), np.int32)
    terminated = np.zeros((num_agents,), np.bool_)
    last_action = np.zeros((num_agents,), np.uint16)
    reward = np.zeros((num_agents,), np.float32)
    action_mask = np.zeros((num_agents, env.num_actions), np.bool_)
    task_ids = np.zeros((num_agents,), np.int32)

    env.reset(1, obs, time, terminated, last_action, reward, action_mask, task_ids)
    assert task_ids.tolist() == [0, 0, 0, 0, 1, 1]


def test_vec_json_builds_num_copies():
    env = Env(VEC_JSON, 32)
    assert env.num_agents == 6  # 3 copies of (1 scout + 1 harvester)
    assert env.task_names == []

    _, w, h, c = env.observation_shape
    obs = np.zeros((env.num_agents, w, h, c), np.uint16)
    env.reset(
        1,
        obs,
        np.zeros((env.num_agents,), np.int32),
        np.zeros((env.num_agents,), np.bool_),
        np.zeros((env.num_agents,), np.uint16),
        np.zeros((env.num_agents,), np.float32),
        np.zeros((env.num_agents, env.num_actions), np.bool_),
        np.zeros((env.num_agents,), np.int32),
    )


def test_factory_routes_rust_multi_config_to_rust_env():
    config = RustMultiConfig(
        envs=(
            RustMultiEnvSpec(
                name="scouts",
                num=2,
                env=RustScoutsConfig(num_scouts=1, num_harvesters=1),
            ),
            RustMultiEnvSpec(
                name="fr",
                num=1,
                env=RustFindReturnConfig(num_agents=2),
            ),
        )
    )
    env = EnvironmentFactory().create_env(config, 32)
    assert isinstance(env, RustEnvJax)
    assert env.num_tasks == 2
    assert env.num_agents == 6


def test_factory_routes_rust_vec_config_to_rust_env():
    config = RustVecConfig(
        num=2,
        env=RustScoutsConfig(num_scouts=1, num_harvesters=1),
    )
    env = EnvironmentFactory().create_env(config, 32)
    assert isinstance(env, RustEnvJax)
    assert env.num_agents == 4


SCOUTS_TASK = RustMultiEnvSpec(
    name="scouts",
    num=2,
    env=RustScoutsConfig(num_scouts=1, num_harvesters=1, width=10, height=8),
)
FR_TASK = RustMultiEnvSpec(
    name="fr", num=1, env=RustFindReturnConfig(num_agents=2, width=12, height=12)
)
STANDALONE = {0: SCOUTS_TASK.env, 1: FR_TASK.env}


def multi_env() -> RustEnvNumpy:
    return RustEnvNumpy(RustMultiConfig(envs=(SCOUTS_TASK, FR_TASK)), 32)


def test_rust_env_task_names_follow_task_ids():
    assert multi_env().task_names == ["scouts", "fr"]
    assert RustEnvNumpy(SCOUTS_TASK.env, 32).task_names == []

    jax_env = RustEnvJax(RustMultiConfig(envs=(SCOUTS_TASK, FR_TASK)), 32)
    assert jax_env.task_names == ["scouts", "fr"]
    assert RustEnvJax(SCOUTS_TASK.env, 32).task_names == []


def test_enjoy_mode_rebinds_the_buffers_to_the_selected_task():
    """A narrowed mode must leave an env the wrapper can actually drive.

    Rust touches exactly `num_agents` rows, so whole-batch buffers are read
    past their end once the mode selects a smaller task; the map a task lends
    the renderer changes with the mode too.
    """
    env = multi_env()
    assert env.num_agents == 6
    assert env.get_render_settings().tile_width == 10  # the first task, scouts

    env.set_enjoy_mode(1)  # the find_return task: one copy of two agents
    assert env.num_agents == 2
    assert env.get_render_settings().tile_width == 12
    assert env.buffers.obs.shape[0] == 2
    assert env.buffers.action_mask.shape[0] == 2

    timestep = env.reset(1)
    assert timestep.task_ids.tolist() == [1, 1]
    assert env.step(np.zeros(env.num_agents, np.uint16)).obs.shape[0] == 2

    env.set_enjoy_mode(None)
    assert env.num_agents == 6
    assert env.get_render_settings().tile_width == 10
    assert env.reset(1).obs.shape == (6, *env.buffers.obs.shape[1:])


@pytest.mark.parametrize("task_id", STANDALONE)
def test_enjoy_mode_shows_the_task_the_single_env_shows(task_id):
    """One task inside the wrapper draws and steps as its single env does.

    The wrapper reports the union vocabulary, so that equality is in symbol
    space: ids of a later task are remapped into the union's id space.
    """
    single = RustEnvNumpy(STANDALONE[task_id], 32)
    multi = multi_env()
    multi.set_enjoy_mode(task_id)
    assert multi.num_agents == single.num_agents

    obs_chars = np.asarray(single.obs_vocab.symbols, dtype=object)
    union_chars = np.asarray(multi.obs_vocab.symbols, dtype=object)
    action_ids = np.asarray(
        [multi.action_vocab.id(symbol) for symbol in single.action_vocab.symbols],
        dtype=np.uint16,
    )

    one = single.reset(3)
    many = multi.reset(3)
    assert np.array_equal(obs_chars[one.obs[..., 0]], union_chars[many.obs[..., 0]])
    assert np.array_equal(one.action_mask, many.action_mask[:, action_ids])

    legal = np.asarray(
        [np.flatnonzero(mask)[0] for mask in one.action_mask], dtype=np.intp
    )
    one = single.step(legal.astype(np.uint16))
    many = multi.step(action_ids[legal])
    assert np.array_equal(obs_chars[one.obs[..., 0]], union_chars[many.obs[..., 0]])
    assert np.array_equal(one.reward, many.reward)
    assert np.array_equal(one.terminated, many.terminated)


def test_jit_step_recompiles_across_modes():
    """A jitted step must follow the buffers across mode switches.

    Training loops pass the env as a static jit argument, so its hash is the
    cache key; the io_callback result shapes captured in the trace change
    with the selected task and have to be retraced.
    """
    env = RustEnvJax(RustMultiConfig(envs=(SCOUTS_TASK, FR_TASK)), 32)

    @partial(jax.jit, static_argnums=(0,))
    def step(env, state, actions):
        return env.step(state, actions, None)

    for task_id in (None, 0, 1, 0):
        env.set_enjoy_mode(task_id)
        state, _ = env.reset(jax.random.key(0))

        actions = jnp.zeros((env.num_agents,), jnp.uint16)
        _, timestep = step(env, state, actions)

        assert timestep.obs.shape[0] == env.num_agents
        assert timestep.action_mask.shape[0] == env.num_agents
