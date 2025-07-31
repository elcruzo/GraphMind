# GraphMind Makefile

.PHONY: help install test lint format clean docker run-tests benchmark reproduce-experiments

# Default target
help:
	@echo "GraphMind Development Commands:"
	@echo "  make install              Install dependencies"
	@echo "  make install-dev          Install with development dependencies"
	@echo "  make test                 Run all tests"
	@echo "  make test-coverage        Run tests with coverage report"
	@echo "  make lint                 Run code linting"
	@echo "  make format               Format code with black and isort"
	@echo "  make clean                Clean temporary files"
	@echo "  make docker-build         Build Docker image"
	@echo "  make benchmark            Run benchmarks"
	@echo "  make reproduce-experiments Reproduce paper experiments"

# Installation
install:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev,docs,visualization]"
	pre-commit install

# Testing
test:
	python -m pytest tests/ -v

test-coverage:
	python -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

test-unit:
	python -m pytest tests/ -v -k "not integration and not benchmark"

test-integration:
	python -m pytest tests/ -v -k "integration"

# Code quality
lint:
	flake8 src/ tests/
	mypy src/ --strict
	pylint src/

format:
	black src/ tests/
	isort src/ tests/

check-format:
	black --check src/ tests/
	isort --check-only src/ tests/

# Benchmarking
benchmark:
	@echo "Running consensus benchmarks..."
	python benchmarks/consensus_benchmark.py
	@echo "Running partitioning benchmarks..."
	python benchmarks/partitioning_benchmark.py
	@echo "Running federated learning benchmarks..."
	python benchmarks/federated_benchmark.py

benchmark-byzantine:
	python src/byzantine_simulation.py --nodes 10 --failures 3 --rounds 100

# Reproduce experiments
reproduce-experiments:
	@echo "Reproducing experiments from paper..."
	@mkdir -p results/reproduced
	python scripts/reproduce_experiments.py --config experiments/paper_config.yaml

generate-plots:
	python scripts/generate_plots.py --input results/ --output plots/

statistical-tests:
	python scripts/statistical_tests.py --data results/

# Docker
docker-build:
	docker build -t graphmind:latest .

docker-run:
	docker run -it --rm -v $(PWD)/data:/app/data graphmind:latest

docker-compose-up:
	docker-compose up -d

docker-compose-down:
	docker-compose down

# Distributed training
train-distributed:
	mpirun -np 4 python src/distributed_train.py --config config/training_config.yaml

train-single:
	python src/distributed_train.py --config config/training_config.yaml

# Documentation
docs:
	cd docs && make html

docs-serve:
	cd docs && python -m http.server --directory _build/html

# Cleaning
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .coverage htmlcov .pytest_cache
	rm -rf build dist

clean-data:
	rm -rf data/*
	rm -rf results/*
	rm -rf logs/*

clean-all: clean clean-data
	rm -rf .venv env venv

# Development
dev-setup: install-dev
	@echo "Setting up development environment..."
	@echo "Installing pre-commit hooks..."
	pre-commit install
	@echo "Development environment ready!"

# MPI testing
test-mpi:
	mpirun -np 4 python -m pytest tests/test_distributed.py -v

# Performance profiling
profile:
	python -m cProfile -o profile.stats src/distributed_train.py --config config/small_test.yaml
	python -m pstats profile.stats

# GPU testing
test-gpu:
	python -m pytest tests/ -v -k "gpu" --gpu

# Release
release-patch:
	bump2version patch

release-minor:
	bump2version minor

release-major:
	bump2version major