def version() -> str:
    """Returns the version of the compiled extension."""

from collections.abc import Callable

import numpy as np

def run_demo(
    config_json: str | None = None,
    policy: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    | None = None,
) -> None:
    """Opens the viewer window, blocking until it is closed or escape is pressed.

    ``config_json`` is a serialized ``EnvConfig`` (defaults to find_return).
    ``policy`` is called once per env step as
    ``policy(obs, reward, terminated, action_mask)`` and must return an array
    of shape ``(num_agents,)`` castable to uint16; the keyboard overrides the
    focused agent's action. Omitted, agents act uniformly at random; if the
    callable raises, the traceback is printed once and the demo falls back to
    the random policy.

    Must be called from the main thread. Only one window can be opened per
    process; calling this a second time raises ``RuntimeError``.
    """
