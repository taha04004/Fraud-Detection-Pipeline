.PHONY: install test lint api etl-pandas etl-spark

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest

lint:
	python -m ruff check .

api:
	python -m uvicorn src.api.main:app --reload

etl-pandas:
	python -m src.data.pandas_baseline

etl-spark:
	python -m src.data.spark_etl

