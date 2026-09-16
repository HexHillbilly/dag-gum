.PHONY: test clean

# Resolve the interpreter: prefer the repo venv, then python3, then python.
VENV_PYTHON := $(CURDIR)/venv/bin/python
PYTHON := $(shell \
    if [ -x "$(VENV_PYTHON)" ]; then \
        echo "$(VENV_PYTHON)"; \
    elif command -v python3 >/dev/null 2>&1; then \
        echo "python3"; \
    elif command -v python >/dev/null 2>&1; then \
        echo "python"; \
    fi \
)

test:
	@echo "Running AVG v2.4 Test Suite..."
	@if [ -z "$(PYTHON)" ]; then echo "ERROR: No Python interpreter found. Tried: $(VENV_PYTHON), python3, python" >&2; exit 1; fi
	@echo "Using interpreter: $(PYTHON)"
	$(PYTHON) scripts/test_factual_safety.py
	$(PYTHON) scripts/test_degeneracy_smoke.py
	$(PYTHON) scripts/test_path2_code_filter.py
	$(PYTHON) scripts/benchmark_latency.py
	$(PYTHON) -m pytest tests/ -q


clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache/
