"""OpenHup backend: API, skill engine, task/alert/metric state, LLM gateway, notifications.

Two entrypoints share this package:

* ``openhup.api.main:app`` - the FastAPI application (REST + WebSocket). I/O only; no CPU work.
* ``openhup.engine:run``   - the singleton worker that consumes observations, ticks the clock,
  runs the skill engine, and dispatches effects.

They are deliberately separate processes: the API must stay responsive while the engine is calling
a local LLM or waiting on a Matrix homeserver.
"""

__version__ = "0.1.0"
