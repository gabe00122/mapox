# Solutions

Hand-written (non-learned) agents for specific mapox environments. They serve
as baselines for RL policies and as probes of what an env's rewards actually
pay for.

Each solution gets its own folder, named after its env, with:

- `README.md`: the approach, how to run it, results, and lessons learned
- `agent.py` (or similar): the agent itself, runnable with `uv run python solutions/<env>/agent.py`

Tests live in the top-level `tests/` as `test_<env>_agent.py`.

| env | folder | summary |
|---|---|---|
| `pacman` | [`pacman/`](pacman/) | plays from ASCII frames with a particle-filter ghost belief and a short lookahead search; ~1030 reward per 1024 turns under default rewards, 99.5% clears in careful mode |
