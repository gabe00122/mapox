# Soccer Environment

Two teams of agents play on an open gridworld field with goals at opposite ends. All agents submit actions simultaneously; execution order is randomized each step.

## Game Design

### Field
- Open rectangle (no internal walls/obstacles), padded with walls for observation slicing.
- Goals are 1-tile-deep zones spanning a configurable width (`goal_width`), centered on opposite edges.
- Bottom goal (y = `pad_height`) is defended by the red team.
- Top goal (y = `pad_height + height - 1`) is defended by the blue team.

### Actions
5 valid actions out of the 7-action space: 4 movement directions + stay. Primary action and dig are masked off.

### Ball
- Single tile on the map, rendered as `AGENT_BALL` (tile ID 16).
- When an agent's movement would place them on the ball's tile, the ball is pushed 1 tile in that agent's movement direction.
- If the pushed ball would land on a wall or another agent, the ball stays put **and the pushing agent also stays in place** (cannot push through a blocked ball).
- Staying agents (action = STAY) never interact with the ball.

### Scoring
- Ball entering a goal zone awards `+goal_reward` to the scoring team and `-goal_reward` to the conceding team.
- Red scores when the ball enters the top goal (blue's goal). Blue scores when the ball enters the bottom goal (red's goal).
- After a goal, ball and all agents reset to starting positions.

### Starting Positions
- Red team: bottom quarter of the field, centered horizontally.
- Blue team: top quarter of the field, centered horizontally.
- Ball: center of the field.

### Turn Execution
- All agents act every step. A random permutation determines execution order.
- Agents are processed sequentially (via `jax.lax.fori_loop`) in that order, which resolves ball-push conflicts naturally — only the first agent (in random order) to reach the ball pushes it.

### Termination
Episode ends after `length` steps.

## Configuration

```python
class SoccerConfig(BaseModel):
    env_type: Literal["soccer"] = "soccer"
    team_size: int = 3       # agents per team (total = team_size * 2)
    width: int = 30           # field width (unpadded)
    height: int = 20          # field height (unpadded)
    view_width: int = 11      # agent observation width
    view_height: int = 11     # agent observation height
    goal_width: int = 6       # goal zone width in tiles
    goal_reward: float = 1.0  # reward magnitude per goal
```

## Observations

Each agent gets a `(view_width, view_height, 4)` int8 tensor sliced from the padded tilemap centered on the agent:

| Channel | Meaning | Values |
|---------|---------|--------|
| 0 | Tile type | 0–16 (empty, wall, decor, goal, ball, agent, etc.) |
| 1 | Direction | 0 (unused in soccer) |
| 2 | Team ID | 0 = none, 1 = red, 2 = blue |
| 3 | Health | 0 (unused in soccer) |

## Tile IDs Added

- `TILE_GOAL = 15` — goal zone tiles at field edges
- `AGENT_BALL = 16` — the ball

## Files

| File | Change |
|------|--------|
| `mapox/envs/soccer.py` | New — `SoccerConfig`, `SoccerState`, `SoccerEnv` |
| `mapox/envs/constance.py` | Added `TILE_GOAL`, `AGENT_BALL`, bumped `NUM_TYPES` to 17 |
| `mapox/config.py` | Import, union type, factory registration |
| `mapox/__init__.py` | Export `SoccerConfig` |
| `mapox/renderer.py` | Sprite mappings for new tile types |
