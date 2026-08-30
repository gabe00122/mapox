import numpy as np

from mapox._core import Env

# Full-field dumps, as RustScoutsConfig/RustFindReturnConfig model_dump_json()
# produce them: the rust side parses with serde and has no field defaults.
MULTI_JSON = (
    '[{"config": {"env_type": "rust_scouts", "num_scouts": 1, "num_harvesters": 1, '
    '"num_treasures": 12, "width": 12, "height": 12, "view_width": 11, "view_height": 13, '
    '"ui_height": 2, "mapgen_threshold": 0.3, "water_threshold": -0.45, '
    '"harvesters_move_every": 6, "scout_reward": 1.0, "harvester_reward": 1.0}, "count": 2},'
    '{"config": {"env_type": "rust_find_return", "num_agents": 2, "num_flags": 1, '
    '"width": 12, "height": 12, "view_width": 11, "view_height": 11, '
    '"mapgen_threshold": 0.3, "water_threshold": -0.45, "digging_timeout": 5, '
    '"preparation_steps": 256, "treasure_reward": 1.0}, "count": 1}]'
)


def test_multitask_json_builds_env_with_task_ids():
    env = Env(MULTI_JSON, 32, 1)
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
