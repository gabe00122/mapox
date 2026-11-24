# Scripted Pac-Man

A hand-written agent for the rust `pacman` env. It plays from the same
ASCII frame a person would see, `mapox.ascii`'s render of Pac-Man's
egocentric view plus the two UI rows, and answers with an action name. It
never reads the full map, the env state, the seed, the reward or the done
flag.

It sets a baseline for RL policies on this env, and it shows what each reward
config actually pays for.

## Running it

```sh
uv run python solutions/pacman/agent.py                    # score 16 budgets of 1024 turns
uv run python solutions/pacman/agent.py --play-to-clear    # careful: clear every maze
uv run python solutions/pacman/agent.py --play-for-return  # maximise discounted return per episode
uv run python solutions/pacman/agent.py --show             # watch one budget as text
uv run python solutions/pacman/agent.py --config '{"death_reward": -0.5}'
uv run python solutions/pacman/agent.py --randomize-start
```

Every `Tuning` field is also a hidden flag (`--depth 24`, `--pace 0.7`, ...).

Terms: an **episode** is one maze, ending in a clear, a catch or the cut-off.
A **budget** is the 1024 turns that episodes run back to back in. The
headline number is reward per episode with the cut-off one counted, which is
what `consume_metrics` reports as `reward`.

## The approach

The agent knows the game the way a human player does: the maze layout, how
Pac-Man and each ghost move (scatter/chase targets, Pinky's lead, Clyde's
radius, the exit order out of the ghost house), and the env config (timers
and reward sizes). `MAZE` and `make_advance` copy
`crates/mapox-core/src/envs/pacman.rs`. They are a rulebook, not a view into
the running env.

From that rulebook and the frames it tracks four things:

1. **Its position.** It matches the wall pattern in view against
   precomputed patterns for every cell of the maze.
2. **The pellets left.** A bitmask, updated from where it has walked and
   checked against what is visible.
3. **The scatter/chase clock.** It counts steps since the round began.
4. **A belief over every ghost.** Each ghost is a small weighted set of
   states (a particle filter). The ghost rules carry it forward each step and
   each frame prunes it. A ghost seen once stays pinned down exactly until a
   power pellet sets it on a random walk.

Each step it searches its own moves 12–22 plies ahead against those beliefs,
with up to 6000 nodes, and plays the first move of the best line. All values
are in pellet units, meaning every reward is divided by `pellet_reward`.

The leaf valuation decides the play style:

- **Reward per step** (default). It follows a pellet tour (nearest-neighbour,
  then 2-opt/or-opt polishing) and charges every step a *pace*, the reward
  per step it expects, derived from the rewards. A plan is judged by how far
  it gets ahead of that pace. A catch is weighed at the env's own price.
- **Careful** (`--play-to-clear`). A catch costs the rest of the maze. Leaves
  value pellets left by distance, charge for each stranded fragment of
  pellets, and hold power pellets until they buy ghosts or survival. The best
  lines are checked 32 steps further for a way to live.
- **Discounted return** (`--play-for-return`). Each episode is played for its
  return at γ = 0.996, the way an RL value function counts it. A catch or a
  clear ends the episode and nothing after it counts.

## Results

1024-turn budgets, standard start:

| style | reward per budget | behaviour |
|---|---|---|
| default rewards | ~1030 | ~70 pellets per life, then walks into a ghost |
| default, `death_reward=-25` | ~773 | ~2.7 clears per budget |
| `--play-to-clear` | ~664 | 99.5% of mazes cleared, ~411 steps per clear |

Under the training rewards (pellet 0.03, ghost 0.3, catch −0.3, clear 3.0,
random start), the first-episode discounted return is 5.47 for
`--play-for-return`, 5.34 for the default style and 5.20 for
`--play-to-clear`. The web-demo-v6 policy makes about 6.03, mostly by eating
ghosts sooner.

## Lessons

- **The default rewards make dying optimal.** With `death_reward=0`, a catch
  just swaps the maze for a full one, so the best policy eats dense pellets
  and dies on purpose. Reward per life and reward per budget rank agents in
  opposite order, so always say which one you mean.
- **Nearly every careful-play death is a frightened ghost out of view**,
  whose belief has spread over hundreds of states. Most of the fix was to
  stop eating power pellets with no ghost near.
- **Horizon artifacts were the recurring bug class.** A leaf term charging
  distance to the nearest pellet leaves one pellet uneaten. Allowing "stand
  still" at the root makes the agent wait for the horizon to shift.
- **A ghost that turns normal on Pac-Man's own cell is hidden by his glyph**,
  so the belief filter has to allow any ghost state there.

## Tests

`tests/test_pacman_agent.py` checks the agent's copy of the maze and its
belief tracking against the real env's full map, so it catches the rust env
and this rulebook drifting apart. It also checks the two headline behaviours:
eating nearly every step under the default rewards, and clearing the maze
with `--play-to-clear`.

```sh
uv run pytest tests/test_pacman_agent.py
```
