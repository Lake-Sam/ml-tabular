# ml-cookiecutter-tabular

End-to-end template for tabular classification with **Hydra + DVC + MLflow + FastAPI**.

## Quickstart

```bash
make setup
make data
make train
make eval
make serve
```

### Example prediction

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
 -d '{"records":[{"age":30,"job":"admin.","marital":"single","education":"university.degree",
 "default":"no","housing":"yes","loan":"no","contact":"cellular","month":"may","day_of_week":"mon",
 "duration":100,"campaign":1,"pdays":999,"previous":0,"poutcome":"nonexistent",
 "emp.var.rate":1.1,"cons.price.idx":93.2,"cons.conf.idx":-34.6,"euribor3m":4.9,"nr.employed":5200.0}]}'
```

## Reproducibility

- **Data & pipeline:** [DVC](https://dvc.org/) (`dvc repro`)
- **Experiments:** [MLflow](https://mlflow.org/) (`mlruns/`)
- **Configs:** [Hydra](https://hydra.cc/) (`configs/`)
- **CI:** GitHub Actions (`.github/workflows/ci.yml`)
- **API:** FastAPI at `/predict` and `/health`

## Project structure

See `configs/`, `src/`, `dvc.yaml`, and `Makefile` for details.

## License

MIT
