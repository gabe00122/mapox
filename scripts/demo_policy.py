"""Drive the rust viewer with a python policy: `uv run scripts/demo_policy.py`.

The callable is invoked once per env step with numpy copies of the timestep
and returns one action id per agent; the keyboard still overrides the focused
agent. Press `p` in the window to switch to free-run and watch it act.
"""

import numpy as np

import mapox

rng = np.random.default_rng(0)


def random_policy(
    obs: np.ndarray,  # (num_agents, view_w, view_h, channels) uint16
    reward: np.ndarray,  # (num_agents,) f32
    terminated: np.ndarray,  # (num_agents,) bool
    action_mask: np.ndarray,  # (num_agents, num_actions) bool, True = legal
) -> np.ndarray:
    # uniform over legal actions, the same masked-argmax trick as
    # mapox.agent.RandomAgent
    logits = rng.random(action_mask.shape)
    return np.where(action_mask, logits, -np.inf).argmax(axis=-1).astype(np.uint16)


if __name__ == "__main__":
    mapox.run_demo(policy=random_policy)
