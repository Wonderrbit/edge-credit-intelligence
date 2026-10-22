.PHONY: install data features train eval export demo test lint pipeline rag-pairs rag-train-reranker autopilot autopilot-dry eval-judgment mcp smoke-ai assets-demo smoke docker-build

install:
	python -m venv .venv --system-site-packages
	.venv/Scripts/python -m pip install -r requirements-dev.txt

data:
	.venv/Scripts/python scripts/generate_data.py

features:
	.venv/Scripts/python scripts/build_features.py

train: features
	.venv/Scripts/python scripts/run_training.py

eval:
	.venv/Scripts/python scripts/run_evaluation.py

export:
	.venv/Scripts/python scripts/run_export.py

pipeline:
	.venv/Scripts/python scripts/run_pipeline.py

rag-pairs:
	.venv/Scripts/python scripts/build_reranker_pairs.py --questions data/rag_questions.jsonl --mine-rerank --negatives 4 --candidates 8 --output data/reranker_pairs_hard.jsonl

rag-train-reranker: rag-pairs
	.venv/Scripts/python scripts/train_reranker.py --pairs data/reranker_pairs_hard.jsonl --output artifacts/rag/reranker_ft_hard --epochs 2

demo:
	.venv/Scripts/python scripts/run_api.py

test:
	.venv/Scripts/python -m pytest

lint:
	.venv/Scripts/python -m ruff check src scripts tests

autopilot:
	.venv/Scripts/python scripts/run_autopilot.py --mode full

autopilot-dry:
	.venv/Scripts/python scripts/run_autopilot.py --mode full --dry-run --limit 50

eval-judgment:
	.venv/Scripts/python scripts/eval_judgment.py

mcp:
	.venv/Scripts/python scripts/run_mcp.py

smoke-ai:
	.venv/Scripts/python scripts/smoke_ai.py

assets-demo:
	.venv/Scripts/python scripts/make_demo_assets.py

smoke: assets-demo
	.venv/Scripts/python -m pytest -q

docker-build:
	docker build -t edge-credit-intelligence:local .
