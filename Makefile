.PHONY: install lint format test train run score clean

VENV := notebooks/.venv
PY := $(VENV)/bin/python
RUFF := $(VENV)/bin/ruff

install:
	$(PY) -m pip install -e ".[dev]"

lint:
	$(RUFF) check src tests
	$(RUFF) format --check src tests

format:
	$(RUFF) check --fix src tests
	$(RUFF) format src tests

test:
	$(PY) -m pytest tests/ -q

train:
	$(PY) -m src.modeling.train

run:
	$(PY) -m src.api.entry

score:
	$(PY) -m src.batch.score data/raw/Telco_customer_churn.xlsx

clean:
	rm -rf .pytest_cache .ruff_cache src/*.egg-info
	find . -type d -name __pycache__ -not -path "./notebooks/.venv/*" -exec rm -rf {} +
