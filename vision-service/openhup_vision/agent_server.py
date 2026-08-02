"""HTTP listener for camera-agents pushing JPEG frames in.

Camera-agents run on hosts that own a camera but cannot be reached from the vision service (a Pi
Zero on wifi, anything behind NAT). They POST JPEGs to `POST /agent/frame?camera_id=…`; the vision
service never opens an outbound connection to them, which is what makes `agent_push` a pull-style
design despite the agent being the one doing the posting.

Deliberately a hand-rolled asyncio HTTP/1.1 server rather than a framework: the surface is two
routes and a body, and the vision service should not have to drag in a web framework for that.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from urllib.parse import parse_qs, urlsplit

log = logging.getLogger(__name__)

SubmitFn = Callable[[str, bytes], Awaitable[None]]


class AgentServer:
    """Minimal HTTP server. Routes:

    * `GET /healthz` → 200 (liveness probe)
    * `POST /agent/frame?camera_id=…` → 200, or 400/401/404/405 on error

    `submit` is called with the camera id and raw JPEG bytes. The token check is constant-time-ish
    and, when no token is configured, uploads are accepted (loopback/trusted-network only).
    """

    def __init__(
        self,
        submit: SubmitFn,
        *,
        host: str = "0.0.0.0",
        port: int = 8090,
        token: str | None = None,
    ) -> None:
        self.submit = submit
        self.host = host
        self.port = port
        self.token = token
        self._server: asyncio.AbstractServer | None = None

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle, self.host, self.port)
        log.info("agent listener on %s:%d", self.host, self.port)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request_line = await asyncio.wait_for(reader.readline(), timeout=10)
        except (TimeoutError, asyncio.IncompleteReadError):
            writer.close()
            return
        if not request_line:
            writer.close()
            return

        parts = request_line.decode("latin-1", errors="replace").split()
        if len(parts) != 3:
            await self._respond(writer, 400, "bad request")
            return
        method, target, _version = parts

        headers: dict[str, str] = {}
        while True:
            line = await reader.readline()
            if line in (b"\r\n", b"\n", b""):
                break
            if len(headers) > 64:
                await self._respond(writer, 400, "too many headers")
                return
            name, _, value = line.decode("latin-1", errors="replace").partition(":")
            headers[name.strip().lower()] = value.strip()

        if method.upper() == "GET" and urlsplit(target).path == "/healthz":
            await self._respond(writer, 200, "ok")
            return

        if method.upper() != "POST" or urlsplit(target).path != "/agent/frame":
            await self._respond(writer, 404, "not found")
            return

        if self.token and headers.get("authorization") != f"Bearer {self.token}":
            await self._respond(writer, 401, "unauthorized")
            return

        camera_id = self._camera_id(target)
        if not camera_id:
            await self._respond(writer, 400, "missing camera_id")
            return

        try:
            length = int(headers.get("content-length", "0"))
        except ValueError:
            await self._respond(writer, 400, "bad content-length")
            return
        if length <= 0 or length > 16 * 1024 * 1024:
            await self._respond(writer, 413 if length > 16 * 1024 * 1024 else 400, "bad length")
            return

        body = await reader.readexactly(length)
        try:
            await self.submit(camera_id, body)
        except ValueError:
            await self._respond(writer, 400, "undecodable JPEG")
            return
        except KeyError:
            await self._respond(writer, 404, "unknown camera")
            return
        await self._respond(writer, 200, "ok")

    @staticmethod
    def _camera_id(target: str) -> str | None:
        query = parse_qs(urlsplit(target).query)
        values = query.get("camera_id")
        return values[0] if values else None

    @staticmethod
    async def _respond(writer: asyncio.StreamWriter, status: int, body: str) -> None:
        reason = {
            200: "OK",
            400: "Bad Request",
            401: "Unauthorized",
            404: "Not Found",
            405: "Method Not Allowed",
            413: "Payload Too Large",
        }.get(status, "OK")
        writer.write(f"HTTP/1.1 {status} {reason}\r\n".encode())
        writer.write(b"content-type: text/plain\r\n")
        writer.write(f"content-length: {len(body)}\r\n".encode())
        writer.write(b"connection: close\r\n\r\n")
        writer.write(body.encode())
        await writer.drain()
        writer.close()
        await writer.wait_closed()


__all__ = ["AgentServer"]
