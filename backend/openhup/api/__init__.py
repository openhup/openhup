"""REST + WebSocket API. I/O only: the skill engine runs in a separate process."""

from .main import create_app

__all__ = ["create_app"]
