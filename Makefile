PYTHON ?= python3
PIP := $(PYTHON) -m pip


.PHONY: help
help:
	@echo "Available commands:"
	@echo "  make setup           Create required directories"
	@echo "  make install         Install project and development dependencies"
	@echo "  make format          Format and automatically fix code"
	@echo "  make lint            Run static code checks"
	@echo "  make type-check      Run type checking"
	@echo "  make test            Run automated tests"
	@echo "  make check           Run lint, type checking and tests"
	@echo "  make audit-data      Run the dataset audit"
	@echo "  make train-baseline  Train the classical baseline model"
	@echo "  make app             Start the Streamlit application"
	@echo "  make clean           Remove temporary Python files"


.PHONY: setup
setup:
	mkdir -p data/raw
	mkdir -p data/interim
	mkdir -p data/processed
	mkdir -p artifacts
	mkdir -p reports/figures
	mkdir -p reports/metrics


.PHONY: install
install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-dev.txt


.PHONY: format
format:
	ruff check --fix .
	ruff format .


.PHONY: lint
lint:
	ruff check .


.PHONY: type-check
type-check:
	mypy


.PHONY: test
test:
	pytest


.PHONY: check
check: lint type-check test


.PHONY: audit-data
audit-data:
	$(PYTHON) scripts/audit_data.py \
		--config configs/baseline.yaml


.PHONY: train-baseline
train-baseline:
	$(PYTHON) scripts/train_baseline.py \
		--config configs/baseline.yaml


.PHONY: app
app:
	streamlit run app/streamlit_app.py


.PHONY: clean
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".coverage" -delete
