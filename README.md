# MAPOX

MAPOX is a small collection of JAX-native, multi-agent, partially-observable gridworld environments with a shared observation/action format and a simple pygame renderer.

The environments are functional (state in / state out), designed to work well with `jax.jit`/`jax.vmap`, and expose an `action_mask` for per-environment action subsets.

## Installation

MAPOX requires **Python 3.11+**.

```bash
uv add mapox
```

Notes:
- Video export uses `python-ffmpeg` and requires the `ffmpeg` binary available on your system PATH.

## Quick start

Environments implement a small interface:

- `reset(rng_key) -> (state, timestep)`
- `step(state, actions, rng_key) -> (state, timestep)`
- `timestep.obs`: `(num_agents, view_w, view_h, 4)` int8
- `timestep.action_mask`: `(num_agents, num_actions)` bool

```python
import jax
import jax.numpy as jnp
from mapox import EnvironmentFactory, FindReturnConfig

factory = EnvironmentFactory()
env, _ = factory.create_env(FindReturnConfig(num_agents=2), length=512)

rng = jax.random.PRNGKey(0)
state, ts = env.reset(rng)

# Sample a random *legal* action per agent using the action mask
rng, akey, skey = jax.random.split(rng, 3)
logits = jax.random.uniform(akey, (env.num_agents, env.action_spec.n))
actions = jnp.argmax(jnp.where(ts.action_mask, logits, -1e9), axis=-1)

state, ts = env.step(state, actions, skey)
print(ts.reward, ts.terminated)
```

### One-hot encoding observations

Observations are compact categorical channels. You can expand them into one-hot features:

```python
from mapox import concat_one_hot

# sizes is typically (NUM_TILE_TYPES, 5, 3, 3)
sizes = env.observation_spec.max_value
x = concat_one_hot(ts.obs, sizes)  # (..., sum(sizes))
```

## Interactive play / rendering

A simple interactive runner is provided:

```bash
python -m mapox.play --env king_hill
```

Controls (depending on the environment):
- Movement: `WASD` or arrow keys
- `n`: cycle focused agent

The renderer can show the full map or the focused agent’s POV (a local crop). See `mapox/play.py` for an example of using `GridworldClient` and recording video.

## Observation & action format

All environments share a unified discrete encoding defined in `mapox/envs/constance.py`.

### Actions

The action space is always `DiscreteActionSpec(n=7)` with IDs:

| id | action |
|---:|--------|
| 0 | move up |
| 1 | move right |
| 2 | move down |
| 3 | move left |
| 4 | stay |
| 5 | primary action |
| 6 | dig action |

Not every environment uses every action. Always consult `timestep.action_mask` before sampling.

### Observation

Each agent receives a local crop centered on itself:
`(view_width, view_height, 4)` with channels:

1. `tile_id` (terrain + agent types)
2. `direction` (0 = none, 1..4 = cardinal direction)
3. `team_id` (0 = none/neutral, 1 = red, 2 = blue)
4. `health` (0..2)

## Environments

All environment configs are Pydantic models and can be created through `EnvironmentFactory`.

- **Find & Return** (`FindReturnEnv`, `FindReturnConfig`)  
  Search for goal tiles in a procedurally-generated map. When an agent finds a flag it is rewarded and respawned elsewhere.

- **Scouts** (`ScoutsEnv`, `ScoutsConfig`)  
  Two roles: Harvesters “unlock” treasure tiles; Scouts collect unlocked treasures.

- **Traveling Salesman** (`TravelingSalesmanEnv`, `TravelingSalesmanConfig`)  
  Multiple flags are scattered. Each agent is rewarded the first time it reaches each flag; flags reset after all are collected.

- **King of the Hill** (`KingHillEnv`, `KingHillConfig`)  
  Two-team competitive environment with knights/archers, destructible walls, control points, and team-shared rewards.

## Wrappers

- `VectorWrapper(env, vec_count)`  
  Runs `vec_count` independent copies of an environment via `jax.vmap` and flattens the `(vec, agent)` dimensions into a single agent dimension.

- `MultiTaskWrapper((env1, env2, ...), (name1, name2, ...))`  
  Combines multiple environments into one by concatenating their agents. Adds a per-agent `task_ids` field to the `TimeStep` via `TaskIdWrapper`.

The `EnvironmentFactory` also supports a `MultiTaskConfig` that can build a multitask environment (and optionally vectorize each task).

## Acknowledgements

Rendering uses the [Urizen Onebit Tileset](https://vurmux.itch.io/urizen-onebit-tileset) by Vurmux.
