import numpy as np
import pytest
from mapox._core import Env
from mapox.config import EnvironmentFactory
from mapox.envs.rust_env import (
    RustEnv,
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
    '{"name": "scouts", "num": 2, "env": ' + SCOUTS_JSON + '},'
    '{"name": "fr", "num": 1, "env": ' + FR_JSON + '}]}'
)
VEC_JSON = '{"env_type": "rust_vec", "num": 3, "env": ' + SCOUTS_JSON + '}'


def test_multitask_json_builds_env_with_task_ids():
    env = Env(MULTI_JSON, 32)
    assert env.num_agents == 6  # 2 * (1 + 1) + 2

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


def test_multi_json_rejects_mismatched_view_sizes():
    bad = (
        '{"env_type": "rust_multi", "envs": ['
        f'{{"name": "scouts", "num": 1, "env": {SCOUTS_JSON}}},'
        '{"name": "fr", "num": 1, "env": {"env_type": "rust_find_return", '
        '"num_agents": 2, "num_flags": 1, "width": 12, "height": 12, '
        '"view_width": 11, "view_height": 13, "mapgen_threshold": 0.3, '
        '"water_threshold": -0.45, "digging_timeout": 5, "preparation_steps": 256, '
        '"treasure_reward": 1.0}}]}'
    )
    with pytest.raises(ValueError, match="must share one view size"):
        Env(bad, 32)


def test_multi_json_rejects_zero_count():
    bad = MULTI_JSON.replace('"num": 2', '"num": 0')
    with pytest.raises(ValueError, match="num 0"):
        Env(bad, 32)


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
    assert isinstance(env, RustEnv)
    assert env.num_tasks == 2
    assert env.num_agents == 6


def test_factory_routes_rust_vec_config_to_rust_env():
    config = RustVecConfig(
        num=2,
        env=RustScoutsConfig(num_scouts=1, num_harvesters=1),
    )
    env = EnvironmentFactory().create_env(config, 32)
    assert isinstance(env, RustEnv)
    assert env.num_agents == 4


