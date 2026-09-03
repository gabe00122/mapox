"""VocabWrapper — a single task env lifted into the multitask global vocab.

Extracting one env from a multitask config (create_env with env_name) must
present the same interface the policy saw in training: global specs and
global ids at the boundary, so checkpoints restore without shape mismatches.
"""

import jax
from jax import numpy as jnp

import mapox.symbols as SB
from mapox.config import EnvironmentFactory, MultiTaskConfig, MultiTaskEnvConfig

LENGTH = 32

CONFIG = MultiTaskConfig(
    envs=(
        MultiTaskEnvConfig(
            name="fr", env={"env_type": "find_return", "num_agents": 2, "num_flags": 2}
        ),
        MultiTaskEnvConfig(
            name="ts",
            env={"env_type": "traveling_salesman", "num_agents": 2, "num_flags": 3},
        ),
    ),
)


def _make(env_name, vec_count=1):
    factory = EnvironmentFactory()
    return factory.create_env(CONFIG, LENGTH, vec_count, env_name=env_name)


def test_single_env_matches_multitask_interface():
    full = EnvironmentFactory().create_env(CONFIG, LENGTH)
    single = _make("fr")

    assert full.num_tasks == 2
    assert single.num_tasks == 1
    assert single.action_spec == full.action_spec
    assert single.observation_spec == full.observation_spec
    assert single.action_vocab == full.action_vocab
    assert single.obs_vocab == full.obs_vocab
    assert single.num_agents == 2


def test_action_mask_lifted_to_global_slots():
    single = _make("fr")
    _, ts = single.reset(jax.random.key(0))

    assert ts.action_mask.shape == (single.num_agents, single.action_spec.n)
    # stay only exists in ts's vocabulary, so fr must mask it out.
    stay = single.action_vocab.id(SB.STAY)
    assert not ts.action_mask[:, stay].any()
    for m in SB.MOVES:
        assert ts.action_mask[:, single.action_vocab.id(m)].all()


def test_task_ids_preserved():
    single = _make("ts")
    _, ts = single.reset(jax.random.key(0))

    assert ts.task_ids is not None
    assert jnp.array_equal(ts.task_ids, jnp.full(single.num_agents, 1))


def test_step_translates_global_actions():
    single = _make("ts")
    k1, k2 = jax.random.split(jax.random.key(0))

    state, _ = single.reset(k1)
    move_up = single.action_vocab.id(SB.MOVE_UP)
    actions = jnp.full((single.num_agents,), move_up, dtype=jnp.uint16)
    _, ts = single.step(state, actions, k2)

    # last_action comes back in global ids.
    assert jnp.all(ts.last_action == move_up)


def test_obs_uses_global_ids():
    single = _make("fr")
    _, ts = single.reset(jax.random.key(0))

    # fr's tile channel must decode through the global vocab: every id it
    # emits exists there, and fr-only symbols keep their global slot.
    assert ts.obs[..., 0].max() < len(single.obs_vocab)
    assert SB.TILE_DESTRUCTIBLE_WALL in single.obs_vocab


def test_vec_count_applies_to_selected_env_only():
    single = _make("fr", vec_count=3)
    assert single.num_agents == 6

    _, ts = single.reset(jax.random.key(0))
    assert ts.obs.shape[0] == 6
