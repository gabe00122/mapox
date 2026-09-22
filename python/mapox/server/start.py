"""FastAPI server exposing one active environment over HTTP.

Requires the "server" extra:
    uv sync --extra server

Run with:
    uv run python -m mapox.server.start

Endpoints:
    POST /env                    create + reset the active env
    GET  /env                    info: shapes, symbols, ascii legend
    GET  /env/obs/{agent_id}     ascii observation grid for one agent
    POST /env/act/{agent_id}     act as that agent; others act randomly
"""

from pydantic import BaseModel, Field

from mapox.envs.rust_env import (
    RustEnvConfig,
    RustMultiConfig,
    RustVecConfig,
    RustVideoConfig,
)
from mapox.server.session import (
    ActionSymbolError,
    AgentIdError,
    IllegalActionError,
    PlaySession,
)

try:
    import uvicorn
    from fastapi import FastAPI, HTTPException
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The mapox server requires extra dependencies. "
        "Install with: pip install 'mapox[server]' "
        "(or: uv sync --extra server)"
    ) from exc


app = FastAPI(
    title="mapox server",
    version="0.2.0",
)

SESSION_ERROR_MAP: tuple[tuple[type[Exception], int], ...] = (
    (AgentIdError, 404),
    (ActionSymbolError, 400),
    (IllegalActionError, 400),
)

session = None

def get_session() -> PlaySession:
    if session is None:
        raise HTTPException(status_code=404, detail="no active env; POST /env first")
    return session

def _records_video(config: RustEnvConfig) -> bool:
    """Whether the config tree contains a video recorder.

    Checked recursively: rust_vec and rust_multi nest their inner config, so a
    top-level guard would miss a video wrapper hiding one level down.
    """

    match config:
        case RustVideoConfig():
            return True
        case RustVecConfig():
            return _records_video(config.env)
        case RustMultiConfig():
            return any(_records_video(spec.env) for spec in config.envs)
        case _:
            return False


class CreateEnvRequest(BaseModel):
    config: RustEnvConfig = Field(
        description=(
            "environment config, e.g. RustFindReturnConfig"
        )
    )
    length: int = Field(default=512, ge=1)
    seed: int | None = Field(default=None, ge=0)


class ActRequest(BaseModel):
    action: str = Field(
        description="action symbol (e.g. 'move/up') or action id"
    )

@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/env")
async def create_env(req: CreateEnvRequest) -> dict:
    global session

    if _records_video(req.config):
        raise HTTPException(status_code=400, detail="Videos recording isn't supported over the  api")
    # the request model validated the config through the rust union and filled
    # pydantic defaults; forward its json — the rust serde structs want every
    # field present
    try:
        session = PlaySession(req.config.model_dump_json(), req.length, req.seed)
    except Exception as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    return session.info()


@app.get("/env")
async def env_info() -> dict:
    return get_session().info()


@app.get("/env/obs/{agent_id}")
async def env_obs(agent_id: int) -> dict:
    session = get_session()
    try:
        return session.agent_report(agent_id)
    except AgentIdError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err


@app.post("/env/act/{agent_id}")
async def env_act(agent_id: int, req: ActRequest) -> dict:
    session = get_session()
    try:
        return session.act(agent_id, req.action)
    except (AgentIdError, ActionSymbolError, IllegalActionError) as err:
        status = next(
            (code for cls, code in SESSION_ERROR_MAP if isinstance(err, cls)), 400
        )
        raise HTTPException(status_code=status, detail=str(err)) from err


def main() -> None:
    uvicorn.run(
        "mapox.server.start:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()
