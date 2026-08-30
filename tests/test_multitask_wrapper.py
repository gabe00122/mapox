"""MultiTaskWrapper — vocab merge, boundary translation, and concatenation.

The wrapper presents heterogeneous local-vocab envs as one global-vocab env:
global ids in, global ids out, per-env translation at the boundary.
"""

import jax
from jax import numpy as jnp

import mapox.symbols as SB
from mapox.config import EnvironmentFactory, MultiTaskConfig, MultiTaskEnvConfig
from mapox.envs.find_return import FindReturnConfig, FindReturnEnv
from mapox.envs.traveling_salesman import TravelingSalesmanConfig, TravelingSalesmanEnv

LENGTH = 32

FR_CONFIG = {"env_type": "find_return", "num_agents": 2, "num_flags": 2}
TS_CONFIG = {"env_type": "traveling_salesman", "num_agents": 2, "num_flags": 3}


def _make_wrapper():
    config = MultiTaskConfig(
        envs=(
            MultiTaskEnvConfig(name="fr", env=FR_CONFIG),
            MultiTaskEnvConfig(name="ts", env=TS_CONFIG),
        ),
    )
    factory = EnvironmentFactory()
    wrapper, num_tasks = factory.create_env(config, LENGTH)
    return wrapper, num_tasks


def test_num_agents():
    wrapper, _ = _make_wrapper()
    assert wrapper.num_agents == 4


def test_num_tasks():
    _, num_tasks = _make_wrapper()
    assert num_tasks == 2


def test_merged_vocabs():
    wrapper, _ = _make_wrapper()

    # Union of both envs; shared symbols appear once.
    assert SB.TILE_DESTRUCTIBLE_WALL in wrapper.obs_vocab  # fr only
    assert SB.TILE_FLAG in wrapper.obs_vocab  # shared
    assert set(wrapper.action_vocab.symbols) == {*SB.MOVES, SB.STAY}

    assert wrapper.obs_vocab.frozen
    assert wrapper.action_vocab.frozen


def test_action_spec_is_global_width():
    wrapper, _ = _make_wrapper()
    assert wrapper.action_spec.n == len(wrapper.action_vocab) == 5


def test_observation_spec_keeps_channel_structure():
    wrapper, _ = _make_wrapper()
    max_value = wrapper.observation_spec.max_value

    # Channel 0 grows to the global vocab; the structural channels
    # (direction, team, health) keep their fixed sizes.
    assert max_value == (len(wrapper.obs_vocab), 5, 3, 3)


def test_reset_concatenates():
    wrapper, _ = _make_wrapper()
    key = jax.random.key(0)
    _, ts = wrapper.reset(key)

    assert ts.obs.shape[0] == wrapper.num_agents
    assert ts.reward.shape == (wrapper.num_agents,)
    assert ts.terminated.shape == (wrapper.num_agents,)
    assert ts.action_mask.shape == (wrapper.num_agents, len(wrapper.action_vocab))


def test_task_ids():
    wrapper, _ = _make_wrapper()
    key = jax.random.key(0)
    _, ts = wrapper.reset(key)

    assert ts.task_ids is not None
    assert ts.task_ids.shape == (wrapper.num_agents,)
    expected = jnp.array([0, 0, 1, 1])
    assert jnp.array_equal(ts.task_ids, expected)


def test_action_mask_lifted_to_global_slots():
    wrapper, _ = _make_wrapper()
    _, ts = wrapper.reset(jax.random.key(0))

    stay = wrapper.action_vocab.id(SB.STAY)
    moves = [wrapper.action_vocab.id(s) for s in SB.MOVES]

    # Moves are legal for every agent; stay only exists for ts (rows 2-3).
    for m in moves:
        assert jnp.all(ts.action_mask[:, m])
    assert not ts.action_mask[0, stay]
    assert not ts.action_mask[1, stay]
    assert ts.action_mask[2, stay]
    assert ts.action_mask[3, stay]


def test_obs_translates_only_the_tile_channel():
    wrapper, _ = _make_wrapper()
    key = jax.random.key(0)
    _, ts = wrapper.reset(key)

    # Rebuild the raw envs identically and replay the wrapper's key split:
    # the wrapper's obs must be the raw obs with channel 0 mapped through
    # the local->global LUT and channels 1..3 untouched.
    raw = [
        FindReturnEnv(
            FindReturnConfig(**{k: v for k, v in FR_CONFIG.items() if k != "env_type"}),
            LENGTH,
        ),
        TravelingSalesmanEnv(
            TravelingSalesmanConfig(
                **{k: v for k, v in TS_CONFIG.items() if k != "env_type"}
            ),
            LENGTH,
        ),
    ]
    keys = jax.random.split(key, 2)

    start = 0
    for env, env_key in zip(raw, keys):
        _, raw_ts = env.reset(env_key)
        lut = env.obs_vocab.lut_to(wrapper.obs_vocab)
        got = ts.obs[start : start + env.num_agents]

        assert jnp.array_equal(got[..., 0], lut[raw_ts.obs[..., 0]])
        assert jnp.array_equal(got[..., 1:], raw_ts.obs[..., 1:])
        start += env.num_agents


def test_step_translates_global_actions():
    wrapper, _ = _make_wrapper()
    k1, k2 = jax.random.split(jax.random.key(0))

    states, _ = wrapper.reset(k1)
    move_up = wrapper.action_vocab.id(SB.MOVE_UP)
    actions = jnp.full((wrapper.num_agents,), move_up, dtype=jnp.uint16)
    _, ts = wrapper.step(states, actions, k2)

    assert ts.obs.shape[0] == wrapper.num_agents
    assert ts.reward.shape == (wrapper.num_agents,)
    # last_action comes back in global ids.
    assert jnp.all(ts.last_action == move_up)


def test_untranslatable_action_is_safe():
    # stay is not in find_return's vocabulary; sending it must not crash,
    # and ts agents (who do have stay) must see it echoed back globally.
    wrapper, _ = _make_wrapper()
    k1, k2 = jax.random.split(jax.random.key(0))

    states, _ = wrapper.reset(k1)
    stay = wrapper.action_vocab.id(SB.STAY)
    actions = jnp.full((wrapper.num_agents,), stay, dtype=jnp.uint16)
    _, ts = wrapper.step(states, actions, k2)

    assert jnp.all(ts.last_action < len(wrapper.action_vocab))
    assert jnp.all(ts.last_action[2:] == stay)


def test_teams_none_when_no_env_has_teams():
    wrapper, _ = _make_wrapper()
    assert wrapper.teams is None


def test_teams_forwarded_through_task_id_wrapper():
    config = MultiTaskConfig(
        envs=(
            MultiTaskEnvConfig(
                name="fr",
                env={"env_type": "find_return", "num_agents": 2, "num_flags": 2},
            ),
            MultiTaskEnvConfig(
                name="kh",
                env={"env_type": "king_hill"},
            ),
        ),
    )
    factory = EnvironmentFactory()
    wrapper, _ = factory.create_env(config, LENGTH)

    teams = wrapper.teams
    assert teams is not None
    assert teams.shape == (wrapper.num_agents,)
    # find_return agents default to team 0, king_hill splits into teams 0 and 1
    assert jnp.array_equal(teams[:2], jnp.zeros(2, teams.dtype))
    assert teams.max() == 1

    single, _ = factory.create_env(config, LENGTH, env_name="kh")
    assert single.teams is not None
    assert jnp.array_equal(single.teams, wrapper.teams[2:])
