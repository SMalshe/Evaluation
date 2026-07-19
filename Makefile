# Common tasks. Requires uv: https://docs.astral.sh/uv/
# Targets marked "spends API credit" hit real providers; the rest are offline.

UV ?= $(shell command -v uv 2>/dev/null || echo $(HOME)/.local/bin/uv)
RUN := $(UV) run

HOST ?= 127.0.0.1
PORT ?= 8000
MODEL_A ?= claude-sonnet
MODEL_B ?= llama-70b
MAX_TURNS ?= 30
SCENARIO ?= s01
DEFENSE ?= none
ADVERSARY ?= passive
TRANSCRIPT ?=

.DEFAULT_GOAL := help
.PHONY: help install run run-cli preview eval ping-models test lint format check clean

help: ## List the available targets
	@grep -hE '^[a-z-]+:.*##' $(MAKEFILE_LIST) \
		| awk -F':.*##' '{printf "  \033[1m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Create/update the venv from the lockfile
	$(UV) sync

run: ## Serve the dashboard at HOST:PORT; spends API credit per conversation
	$(RUN) python -m src.server --host $(HOST) --port $(PORT)

run-cli: ## One conversation in the terminal (MODEL_A=opener); spends API credit
	$(RUN) python -m src.smoke --model-a $(MODEL_A) --model-b $(MODEL_B) \
		--max-turns $(MAX_TURNS)

preview: ## Render a scenario's prompts (SCENARIO=, DEFENSE=, ADVERSARY=); offline
	$(RUN) python -m src.preview --scenario $(SCENARIO) --defense $(DEFENSE) \
		--adversary $(ADVERSARY)

eval: ## Evaluate a transcript (TRANSCRIPT=, SCENARIO=, DEFENSE=, ADVERSARY=); spends API credit
	$(RUN) python -m src.evaluate --transcript $(TRANSCRIPT) --scenario $(SCENARIO) \
		--defense $(DEFENSE) --adversary $(ADVERSARY)

ping-models: ## Probe every registry entry with one tiny live call; spends API credit
	$(RUN) python scripts/live_check.py

test: ## Run the offline tests (mock backend, no network)
	$(RUN) pytest -q

lint: ## Check lint + formatting
	$(RUN) ruff check .
	$(RUN) ruff format --check .

format: ## Apply formatting and autofixable lint
	$(RUN) ruff check --fix .
	$(RUN) ruff format .

check: lint test ## Everything that must pass before a commit

clean: ## Remove tool caches (keeps .venv and runs/)
	rm -rf .pytest_cache .ruff_cache
	find src tests scripts -type d -name __pycache__ -prune -exec rm -rf {} +
