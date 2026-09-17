# jarvis-core
#
# `make check` is the gate CI runs. Everything in it works offline: no API
# key, no network, no billed request.

.DEFAULT_GOAL := help
PY ?= python
VENV := .venv
BIN := $(VENV)/bin

.PHONY: help
help: ## Show the available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install: ## Create a virtualenv and install the project with its dev extras
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --quiet --upgrade pip "setuptools>=83.0.0"
	$(BIN)/pip install --quiet -e ".[dev]"

.PHONY: lint
lint: ## ruff check and ruff format --check
	$(BIN)/ruff check src tests examples
	$(BIN)/ruff format --check src tests examples

.PHONY: format
format: ## Apply ruff's formatting
	$(BIN)/ruff format src tests examples
	$(BIN)/ruff check --fix src tests examples

.PHONY: test
test: ## The test suite, offline
	$(BIN)/pytest -q

.PHONY: demo
demo: ## The whole pipeline against the offline provider, no key needed
	$(BIN)/python examples/demo_offline.py

.PHONY: gateway
gateway: ## The tool gateway, with every gate exercised
	$(BIN)/python examples/demo_tool_gateway.py

.PHONY: lab
lab: ## The sandbox and the experiment lab
	$(BIN)/python examples/demo_sandbox.py

.PHONY: overnight
overnight: ## An approval surviving across two real processes
	$(BIN)/python examples/demo_overnight_approval.py

.PHONY: benchmark
benchmark: ## What the cost controls save, measured
	$(BIN)/python examples/benchmark_cost.py
	$(BIN)/python examples/benchmark_gateway.py

.PHONY: security
security: ## bandit and pip-audit
	$(BIN)/bandit -q -r src -ll
	$(BIN)/pip-audit --skip-editable --progress-spinner off

.PHONY: check
check: lint test demo gateway lab overnight benchmark ## The full gate
	@echo "check: ok"

.PHONY: docker
docker: ## Build the image and run the suite inside it
	docker build -t jarvis-core:dev .
	docker run --rm jarvis-core:dev pytest -q

.PHONY: clean
clean: ## Remove caches and build artefacts
	rm -rf .pytest_cache .ruff_cache dist build *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
