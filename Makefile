.PHONY: help install test examples lint clean

help:
	@echo "Targets:"
	@echo "  install   - editable install with dev extras (pytest)"
	@echo "  test      - run the test suite"
	@echo "  examples  - run every example script in order"
	@echo "  clean     - remove caches and build artifacts"

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest

examples:
	python examples/run_all.py

clean:
	rm -rf .pytest_cache build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
