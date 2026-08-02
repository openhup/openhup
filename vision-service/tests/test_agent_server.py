"""End-to-end tests for the camera-agent HTTP listener, on an ephemeral port."""

from __future__ import annotations

import httpx

from openhup_vision.agent_server import AgentServer


async def test_agent_server_routes_frames_and_enforces_the_token() -> None:
    received: dict[str, object] = {}

    async def submit(camera_id: str, jpeg: bytes) -> None:
        received["camera_id"] = camera_id
        received["jpeg"] = jpeg

    server = AgentServer(submit, host="127.0.0.1", port=0, token="secret")
    await server.start()
    assert server._server is not None
    port = server._server.sockets[0].getsockname()[1]

    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as client:
        assert (await client.get("/healthz")).status_code == 200

        rejected = await client.post(
            "/agent/frame?camera_id=cam1",
            content=b"jpeg-bytes",
            headers={"Authorization": "Bearer wrong"},
        )
        assert rejected.status_code == 401

        missing = await client.post(
            "/agent/frame", content=b"jpeg-bytes", headers={"Authorization": "Bearer secret"}
        )
        assert missing.status_code == 400

        accepted = await client.post(
            "/agent/frame?camera_id=cam1",
            content=b"jpeg-bytes",
            headers={"Authorization": "Bearer secret"},
        )
        assert accepted.status_code == 200
        assert received == {"camera_id": "cam1", "jpeg": b"jpeg-bytes"}

    await server.stop()


async def test_agent_server_accepts_unauthenticated_when_no_token_is_set() -> None:
    async def submit(camera_id: str, jpeg: bytes) -> None:
        pass

    server = AgentServer(submit, host="127.0.0.1", port=0, token=None)
    await server.start()
    assert server._server is not None
    port = server._server.sockets[0].getsockname()[1]

    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as client:
        response = await client.post("/agent/frame?camera_id=cam1", content=b"jpeg")
        assert response.status_code == 200

    await server.stop()


async def test_agent_server_surfaces_unknown_cameras() -> None:
    async def submit(camera_id: str, jpeg: bytes) -> None:
        raise KeyError(camera_id)

    server = AgentServer(submit, host="127.0.0.1", port=0, token=None)
    await server.start()
    assert server._server is not None
    port = server._server.sockets[0].getsockname()[1]

    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as client:
        response = await client.post("/agent/frame?camera_id=ghost", content=b"jpeg")
        assert response.status_code == 404

    await server.stop()


async def test_agent_server_rejects_an_undecodable_jpeg() -> None:
    async def submit(camera_id: str, jpeg: bytes) -> None:
        raise ValueError("undecodable")

    server = AgentServer(submit, host="127.0.0.1", port=0, token=None)
    await server.start()
    assert server._server is not None
    port = server._server.sockets[0].getsockname()[1]

    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as client:
        response = await client.post("/agent/frame?camera_id=cam1", content=b"jpeg")
        assert response.status_code == 400

    await server.stop()
