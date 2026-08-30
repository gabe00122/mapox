from typing import Any

from mapox.envs.rust_env import RustEnv

def enjoy(
    env: RustEnv,
    length: int,
    seed: int,
    policy: Any | None = None,
) -> None: ...
