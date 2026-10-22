from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from mlops.config import ensure_dirs, resolve_paths


class PairDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, str]:
        return self.rows[index]


def _load_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        rows.append({"query": record["query"], "passage": record["positive"], "label": 1})
        for negative in record.get("negatives", []):
            rows.append({"query": record["query"], "passage": negative, "label": 0})
    return rows


def _split_queries(rows: list[dict[str, str]], validation_ratio: float, seed: int):
    queries = sorted({row["query"] for row in rows})
    rng = random.Random(seed)
    rng.shuffle(queries)
    validation_size = max(1, int(len(queries) * validation_ratio))
    validation_queries = set(queries[:validation_size])
    train_rows = [row for row in rows if row["query"] not in validation_queries]
    validation_rows = [row for row in rows if row["query"] in validation_queries]
    return train_rows, validation_rows


def _collate(tokenizer, max_length: int):
    def collate(batch: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        encoded = tokenizer(
            [row["query"] for row in batch],
            [row["passage"] for row in batch],
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor([[float(row["label"])] for row in batch])
        return encoded

    return collate


@torch.no_grad()
def _evaluate(
    model,
    rows: list[dict[str, str]],
    tokenizer,
    batch_size: int,
    max_length: int,
    device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0
    pairwise_correct = 0
    pairwise_total = 0
    by_query: dict[str, list[tuple[float, int]]] = {}
    loss_fct = torch.nn.BCEWithLogitsLoss()
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        encoded = tokenizer(
            [row["query"] for row in batch],
            [row["passage"] for row in batch],
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        labels = torch.tensor([[float(row["label"])] for row in batch]).to(device)
        logits = model(**{key: value.to(device) for key, value in encoded.items()}).logits
        total_loss += float(loss_fct(logits, labels)) * labels.size(0)
        predictions = (torch.sigmoid(logits) >= 0.5).float()
        correct += int((predictions == labels).sum())
        count += labels.size(0)
        for row, logit, label in zip(batch, logits.view(-1), labels.view(-1), strict=True):
            by_query.setdefault(row["query"], []).append((float(logit), int(label)))
    for scores in by_query.values():
        positives = [score for score, label in scores if label == 1]
        negatives = [score for score, label in scores if label == 0]
        if positives and negatives:
            pairwise_total += 1
            pairwise_correct += int(max(positives) > max(negatives))
    return {
        "loss": round(total_loss / max(count, 1), 6),
        "accuracy": round(correct / max(count, 1), 4),
        "pairwise_accuracy": round(pairwise_correct / max(pairwise_total, 1), 4),
    }


def train(
    pairs_path: Path,
    output_dir: Path,
    base_model: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
    validation_ratio: float,
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)
    rows = _load_rows(pairs_path)
    train_rows, validation_rows = _split_queries(rows, validation_ratio, seed)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    collate = _collate(tokenizer, max_length)
    train_loader = DataLoader(
        PairDataset(train_rows),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    steps = max(1, len(train_loader) * epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(steps * 0.1)),
        num_training_steps=steps,
    )
    loss_fct = torch.nn.BCEWithLogitsLoss()

    best = {"loss": float("inf"), "epoch": 0}
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            logits = model(**batch).logits
            loss = loss_fct(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += float(loss.detach()) * labels.size(0)
        train_loss = running / max(len(train_rows), 1)
        metrics = _evaluate(
            model,
            validation_rows,
            tokenizer,
            batch_size,
            max_length,
            device,
        )
        metrics["train_loss"] = round(train_loss, 6)
        metrics["epoch"] = epoch
        history.append(metrics)
        print(json.dumps(metrics), flush=True)
        if metrics["loss"] < best["loss"]:
            best = {"loss": metrics["loss"], "epoch": epoch, "metrics": metrics}
            output_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

    meta = {
        "base_model": base_model,
        "output_dir": str(output_dir),
        "pairs": len(rows),
        "train_pairs": len(train_rows),
        "validation_pairs": len(validation_rows),
        "queries": len({row["query"] for row in rows}),
        "best_epoch": best["epoch"],
        "best_validation": best.get("metrics", {}),
        "history": history,
        "seed": seed,
    }
    (output_dir / "training_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune do cross-encoder com pares de domínio.")
    parser.add_argument("--pairs", default="data/reranker_pairs.jsonl")
    parser.add_argument("--output", default="artifacts/rag/reranker_ft")
    parser.add_argument("--base-model", default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=320)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = ensure_dirs(resolve_paths())
    pairs = Path(args.pairs)
    output = Path(args.output)
    if not pairs.is_absolute():
        pairs = paths.project_root / pairs
    if not output.is_absolute():
        output = paths.project_root / output
    meta = train(
        pairs,
        output,
        args.base_model,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.max_length,
        args.validation_ratio,
        args.seed,
    )
    print(json.dumps({key: meta[key] for key in ("pairs", "best_epoch", "best_validation")}, indent=2))


if __name__ == "__main__":
    main()
