# OpenHup development tasks.
#
# The three Python components have separate virtualenvs on purpose: the vision service pulls in
# onnxruntime and OpenCV, and the backend should not have to.

.DEFAULT_GOAL := help
SHELL := /bin/bash

SCHEMAS  := packages/openhup-schemas
BACKEND  := backend
VISION   := vision-service
FRONTEND := frontend
COMPOSE  := deploy/compose/docker-compose.yml

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------- setup
.PHONY: install
install: ## Sync all Python venvs (vision on CPU inference)
	cd $(SCHEMAS) && uv sync
	cd $(BACKEND) && uv sync
	cd $(VISION) && uv sync

.PHONY: install-vision-openvino install-vision-cuda
install-vision-openvino: ## Vision deps for an Intel iGPU
	cd $(VISION) && uv sync --extra openvino
install-vision-cuda: ## Vision deps for NVIDIA
	cd $(VISION) && uv sync --extra cuda

.PHONY: models
models: ## Download and verify model weights
	cd $(VISION) && uv run python -m openhup_vision.backends --fetch --trust-first-use

.PHONY: config
config: ## Non-interactive: copy config/ from the examples (setup also does this, with secrets)
	mkdir -p config
	cp -n config/config.yaml.example config/config.yaml || true
	cp -n config/vision.yaml.example config/vision.yaml || true
	cp -n examples/cameras/cameras.yaml config/cameras.yaml || true
	cp -n examples/personalities/personalities.yaml config/personalities.yaml || true
	@echo "Now run 'make setup' to generate secrets, ask the questions, and get the start commands."

.PHONY: setup
setup: ## The whole first run: bootstrap config + secrets, voice, AI provider, guided handoff
	cd $(BACKEND) && uv run openhup-cli setup
	@echo "Re-run any time; it merges over the existing config and never overwrites existing files."

# ---------------------------------------------------------------------------- checks
.PHONY: test
test: ## Run every test suite (needs no Postgres, Redis, cameras, or models)
	cd $(SCHEMAS) && uv run pytest -q
	cd $(BACKEND) && uv run pytest -q
	cd $(VISION) && uv run pytest -q

.PHONY: test-cov
test-cov: ## Backend tests with coverage
	cd $(BACKEND) && uv run pytest --cov=openhup --cov-report=term-missing -q

.PHONY: lint
lint: ## ruff check + format --check
	cd $(BACKEND) && uv run ruff check . && uv run ruff format --check .
	cd $(VISION) && uv run ruff check . && uv run ruff format --check .

.PHONY: format
format: ## Auto-fix and format
	cd $(BACKEND) && uv run ruff check --fix . && uv run ruff format .
	cd $(VISION) && uv run ruff check --fix . && uv run ruff format .

.PHONY: typecheck
typecheck: ## mypy strict on the pure engine core
	cd $(BACKEND) && uv run mypy openhup/skills

.PHONY: check
check: lint typecheck test docs-check examples-check ## Everything CI runs

.PHONY: docs-check
docs-check: ## Validate every YAML/JSON block in the docs
	uv run --with pyyaml scripts/check_docs_blocks.py

.PHONY: examples-check
examples-check: ## Compile every shipped example skill
	cd $(BACKEND) && uv run openhup-cli lint ../examples/skills --cameras ../examples/cameras/cameras.yaml

# ---------------------------------------------------------------------------- running
.PHONY: dev-api dev-engine dev-vision
dev-api: ## API with reload on 127.0.0.1:8080
	cd $(BACKEND) && uv run uvicorn openhup.api.main:application --factory --reload \
		--host 127.0.0.1 --port 8080

dev-engine: ## Skill engine in the foreground
	cd $(BACKEND) && uv run python -m openhup.engine --log-level DEBUG

dev-vision: ## Vision service, detecting but publishing nothing
	cd $(VISION) && uv run openhup-vision --config ../config/vision.yaml \
		--config ../config/cameras.yaml --dry-run --log-level DEBUG

.PHONY: dev-deps
dev-deps: ## Start only Postgres and Redis
	docker compose -f $(COMPOSE) up -d postgres redis

.PHONY: migrate
migrate: ## Apply database migrations
	cd $(BACKEND) && uv run alembic upgrade head

.PHONY: revision
revision: ## New migration: make revision m="add widgets"
	cd $(BACKEND) && uv run alembic revision --autogenerate -m "$(m)"

# ---------------------------------------------------------------------------- frontend
.PHONY: ui-install ui-dev ui-build
ui-install: ## Install frontend dependencies
	cd $(FRONTEND) && pnpm install
ui-dev: ## Frontend dev server
	cd $(FRONTEND) && pnpm dev
ui-build: ## Build the frontend into backend/static
	cd $(FRONTEND) && pnpm build

.PHONY: types
types: ## Regenerate TypeScript types from the Pydantic models
	cd $(BACKEND) && uv run openhup-cli export-schemas --out ../$(SCHEMAS)/jsonschema
	cd $(SCHEMAS) && pnpm dlx json-schema-to-typescript -i 'jsonschema/*.json' -o typescript/generated/

# ---------------------------------------------------------------------------- compose
.PHONY: up down logs ps
up: ## Start the stack (cpu + ollama profiles)
	docker compose -f $(COMPOSE) --profile cpu --profile ollama up -d
down: ## Stop the stack
	docker compose -f $(COMPOSE) down
logs: ## Follow logs
	docker compose -f $(COMPOSE) logs -f --tail=100
ps: ## Container status
	docker compose -f $(COMPOSE) ps

.PHONY: health
health: ## Ask a running instance how it is doing
	@curl -fsS localhost:8080/api/v1/system/health | python3 -m json.tool

.PHONY: clean
clean: ## Remove caches and build artefacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name '.pytest_cache' -prune -exec rm -rf {} +
	find . -type d -name '.ruff_cache' -prune -exec rm -rf {} +
	rm -rf $(FRONTEND)/.svelte-kit $(BACKEND)/static
