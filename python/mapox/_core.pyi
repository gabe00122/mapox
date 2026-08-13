from mapox.envs.rust_env import RustEnv
from typing import Any

def enjoy(
    env: RustEnv,
    length: int,
    seed: int,
    policy: Any | None = None,
) -> None:
    ...
