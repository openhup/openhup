# openhup-backend

The API, the skill engine, and everything that decides what to do about what the cameras saw.

```
openhup/
├── api/v1/     REST routers + WebSocket hub. I/O only - no CPU work in this process.
├── core/       config, logging, ids, security, leader election
├── db/         SQLAlchemy 2.0 async models, session management, Alembic migrations
├── skills/     the engine: window → operators → evaluate → compile → fsm
├── tasks/      task + alert engines, micro-step ladder
├── metrics/    rollups, goals, weekly report
├── llm/        provider abstraction, prompts, personality rendering, safety filter
├── notify/     channel plugins
├── bus/        Redis Streams producer/consumer, topics, idempotency
└── engine.py   worker entrypoint
```

## Two processes, one package

| Process | Entrypoint | Job |
|---|---|---|
| API | `openhup.api.main:app` | serve REST + WS, stay responsive |
| Engine | `openhup.engine:run` | consume observations, tick the clock, evaluate skills, dispatch effects |

The engine is a leader-locked singleton (Redis lock): running two would double every task. The API
scales horizontally behind the reverse proxy.

## Development

```sh
uv sync                       # installs openhup-schemas from ../packages as an editable path dep
uv run pytest -q              # the pure engine tests need no Postgres and no Redis
uv run ruff check .
uv run mypy openhup/skills    # the pure core is strict-typed
```

Postgres and Redis are only needed for the API and integration tests:

```sh
docker compose -f ../deploy/compose/docker-compose.yml up -d postgres redis
uv run alembic upgrade head
uv run uvicorn openhup.api.main:app --reload --host 127.0.0.1 --port 8080
```

## Where to look first

`openhup/skills/` is the heart, and it is deliberately layered so the interesting parts are pure
functions with `now` as a parameter:

- `operators.py` — the temporal operators. `held_for` treats a data gap as breaking a run, which is
  what stops a camera outage from looking like a burner left on.
- `evaluate.py` — returns a `Verdict` (matched + per-node reasons + data health), never a bare bool.
- `compile.py` — rejects skills whose trigger and resolve thresholds overlap, because those flap.
- `fsm.py` — per-instance state machine. Absence of data never resolves a task.

`tests/test_operators.py` is the best available documentation of the intended semantics.
