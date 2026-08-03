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
CROSS_MODEL ?= claude-sonnet
PROBE ?= book_trips
HOLDER_MODEL ?= local-llama-3b
N ?= 15
# --- grid runner knobs -------------------------------------------------------
# GRID_* drive the local/default grid. CLOUD_* are the hosted preset used by
# `make cloud-grid`, which needs no local inference server at all.
GRID_MODELS ?= local-llama-3b,local-llama-8b
GRID_SCENARIOS ?= s01
GRID_DEFENSES ?= none
GRID_ADVERSARIES ?= direct_probe
JUDGE_MODEL ?= local-qwen-14b
GRID_ARGS ?=

# One scenario per method category, so all four are represented.
CLOUD_MODELS ?= claude-sonnet,gpt-mini,gemini-flash,llama-70b
CLOUD_SCENARIOS ?= s01,s13,s25,s37
CLOUD_JUDGE ?= claude-sonnet
RESULTS ?= results/grid.jsonl
REPORT_OUT ?= reports

.DEFAULT_GOAL := help
.PHONY: help install run run-cli preview eval subliminal subliminal-chat experiment experiment-plan cloud-plan cloud-grid report preflight ping-models test lint format check clean

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

subliminal: ## Association probe (PROBE=, HOLDER_MODEL=, CROSS_MODEL=, N=); spends API credit unless models are local
	$(RUN) python -m src.subliminal --probe $(PROBE) --holder-model $(HOLDER_MODEL) \
		--cross-model $(CROSS_MODEL) --n $(N)

subliminal-chat: ## Decode a subliminal_chat transcript (TRANSCRIPT=, SCENARIO=, DEFENSE=, CROSS_MODEL=); spends API credit
	$(RUN) python -m src.subliminal_chat --transcript $(TRANSCRIPT) --scenario $(SCENARIO) \
		--defense $(DEFENSE) --adversary subliminal_chat --cross-model $(CROSS_MODEL)

experiment-plan: ## Print the grid plan and exit; calls nothing, needs no keys
	$(RUN) python -m src.experiment --models $(GRID_MODELS) \
		--scenarios $(GRID_SCENARIOS) --defenses $(GRID_DEFENSES) \
		--adversaries $(GRID_ADVERSARIES) --judge-model $(JUDGE_MODEL) \
		--dry-run $(GRID_ARGS)

experiment: ## Run the grid (resumable); spends API credit on the hosted pairings
	$(RUN) python -m src.experiment --models $(GRID_MODELS) \
		--scenarios $(GRID_SCENARIOS) --defenses $(GRID_DEFENSES) \
		--adversaries $(GRID_ADVERSARIES) --judge-model $(JUDGE_MODEL) \
		$(GRID_ARGS)

cloud-plan: ## Dry-run the hosted grid; calls nothing, needs no keys
	$(MAKE) experiment-plan GRID_MODELS=$(CLOUD_MODELS) \
		GRID_SCENARIOS=$(CLOUD_SCENARIOS) JUDGE_MODEL=$(CLOUD_JUDGE)

cloud-grid: ## Run the hosted grid with a hosted judge (no local server); spends API credit
	$(MAKE) experiment GRID_MODELS=$(CLOUD_MODELS) \
		GRID_SCENARIOS=$(CLOUD_SCENARIOS) JUDGE_MODEL=$(CLOUD_JUDGE)

report: ## Build $(REPORT_OUT)/results.xlsx + deck.pptx from the grid rows; offline
	$(RUN) python -m src.report --results $(RESULTS) --outdir $(REPORT_OUT)

preflight: ## Check keys + that every CLOUD_MODELS entry answers; spends a fraction of a cent
	$(RUN) python -m src.experiment --models $(CLOUD_MODELS) \
		--scenarios $(CLOUD_SCENARIOS) --judge-model $(CLOUD_JUDGE) --dry-run
	$(RUN) python scripts/live_check.py

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
