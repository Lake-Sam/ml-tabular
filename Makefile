.PHONY: setup data split train eval serve test lint format ci docker-build docker-run clean

setup:
	poetry install
	pre-commit install

data:
	python -m src.data.download
	python -m src.data.prepare
	python -m src.data.split

split:
	python -m src.data.split

train:
	python -m src.pipelines.train_pipeline

eval:
	python -m src.pipelines.evaluate_pipeline

serve:
	uvicorn src.api.app:app --host 0.0.0.0 --port 8000

test:
	pytest -q --maxfail=1 --disable-warnings --cov=src

lint:
	flake8 .
	isort --check-only .
	black --check .

format:
	isort .
	black .

docker-build:
	docker build -t ml-cookiecutter-tabular .

docker-run:
	docker run -p 8000:8000 ml-cookiecutter-tabular

clean:
	rm -rf data/processed/* models/* mlruns/*
