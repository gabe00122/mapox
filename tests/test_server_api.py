"""HTTP API tests: one active env, act-as-one-agent, random others.

Builds the real rust env through the session — no mocks. Skipped when the
`server` extra (fastapi/uvicorn) isn't installed.
"""

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from mapox.server.ascii import LEGEND
from mapox.server.start import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        c.post(
            "/env",
            json={
                "config": {
                    "env_type": "rust_find_return",
                    "num_agents": 2,
                    "num_flags": 1,
                    "width": 10,
                    "height": 10,
                    "view_width": 9,
                    "view_height": 9,
                },
                "length": 8,
                "seed": 0,
            },
        )
        yield c


def test_healthz():
    with TestClient(app) as c:
        assert c.get("/healthz").json()["status"] == "ok"


def test_no_env_returns_404():
    with TestClient(app) as c:
        assert c.get("/env").status_code == 404
        assert c.get("/env/obs/0").status_code == 404
        assert c.post("/env/act/0", json={"action": "noop"}).status_code == 404


def test_create_rejects_bad_config():
    with TestClient(app) as c:
        # request validation (pydantic union with env_type discriminator)
        r = c.post("/env", json={"config": {"env_type": "rust_nope"}})
        assert r.status_code == 422
        # rust-side failure, past request validation
        r = c.post(
            "/env",
            json={
                "config": {
                    "env_type": "rust_vec",
                    "num": 0,
                    "env": {"env_type": "rust_find_return"},
                }
            },
        )
        assert r.status_code == 400


def test_create_rejects_video_configs(client):
    # recording writes clips onto the server's filesystem, so rust_video is
    # refused wherever it appears in the config tree
    video = {
        "env_type": "rust_video",
        "env": {"env_type": "rust_find_return", "width": 8, "height": 8},
    }
    for config in [
        video,
        {"env_type": "rust_vec", "num": 1, "env": video},
        {"env_type": "rust_multi", "envs": [{"name": "clips", "num": 1, "env": video}]},
    ]:
        r = client.post("/env", json={"config": config})
        assert r.status_code == 400, r.text

    # a refused request leaves the active env untouched
    assert client.get("/env").json()["num_agents"] == 2
    assert client.post("/env/act/0", json={"action": "noop"}).status_code == 200


def test_info_shape(client):
    info = client.get("/env").json()
    assert info["num_agents"] == 2
    assert info["obs_shape"] == [9, 11]
    assert "tile/wall" in info["ascii_legend"]
    assert "move/up" in info["actions"]
    assert info["ascii_legend"] == {
        k: v for k, v in LEGEND.items() if k != "mask" and k != "ui"
    } | {"mask": " ", "ui": " "}


def test_obs_is_ascii_grid(client):
    obs = client.get("/env/obs/0").json()["obs_ascii"]
    assert isinstance(obs, list)
    assert len(obs) == 11  # view_height 9 + the 2-row UI band
    assert all(isinstance(row, str) and len(row) == 9 for row in obs)


@pytest.mark.xfail(
    reason="agent_report skips _check_agent: out-of-range ids raise "
    "IndexError -> 500 instead of the documented 404"
)
def test_obs_out_of_range_agent(client):
    assert client.get("/env/obs/2").status_code == 404
    assert client.get("/env/obs/-1").status_code == 404


def test_act_returns_agent_report(client):
    report = client.post("/env/act/0", json={"action": "noop"}).json()
    assert report["agent_id"] == 0
    assert report["last_action"] == "noop"
    assert isinstance(report["reward"], float)
    assert isinstance(report["terminated"], bool)
    assert isinstance(report["legal_actions"], list)
    assert len(report["obs_ascii"]) == 11


def test_act_rejects_illegal_action(client):
    # find_return has no dig action: the mask must reject it
    r = client.post("/env/act/0", json={"action": "dig"})
    assert r.status_code == 400
    assert "legal actions" in r.json()["detail"]


def test_act_rejects_unknown_action(client):
    r = client.post("/env/act/0", json={"action": "fly"})
    assert r.status_code == 400


def test_act_rejects_non_string_action(client):
    # actions are symbols only; int ids are a validation error
    assert client.post("/env/act/0", json={"action": 6}).status_code == 422


def test_no_auto_reset_on_terminate(client):
    # length=8: terminated fires on the length-th step; further steps keep
    # the episode running (no auto-reset) — the client POSTs /env to restart
    for _ in range(8):
        report = client.post("/env/act/0", json={"action": "noop"}).json()
    assert report["terminated"] is True
    assert report["time"] == 8

    report = client.post("/env/act/0", json={"action": "noop"}).json()
    assert report["time"] == 9
    assert report["terminated"] is False


def test_ascii_symbols_are_single_chars():
    from mapox.server.ascii import UNKNOWN_CHAR, LEGEND, build_char_table

    table = build_char_table(list(LEGEND))
    assert all(len(c) == 1 for c in LEGEND.values())
    # every listed symbol resolves to its own char, never the unknown fallback
    assert all(c != UNKNOWN_CHAR for c in table)
    # fog of war and the UI band are blank, not question marks
    assert LEGEND["mask"] == LEGEND["ui"] == " "

    # symbols outside the table fall back to ?
    assert build_char_table(["tile/not_a_symbol"]) == [UNKNOWN_CHAR]


def test_obs_ascii_is_verbatim(client):
    # no @ stamp, no synthetic cells: every char comes from the legend
    rows = client.get("/env/obs/0").json()["obs_ascii"]
    assert all(c == " " or c in set(LEGEND.values()) for row in rows for c in row)
    assert not any("@" in row for row in rows)
