"""Deterministic spam/quality text-classification smoke demo.

This intentionally avoids the legacy notebook dependencies and trains a small
Multinomial Naive Bayes model with the Python standard library.
"""

from __future__ import annotations

import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOKEN_RE = re.compile(r"[A-Za-z0-9_#@']+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def load_rows(limit: int | None = None) -> list[dict[str, str]]:
    with (ROOT / "train.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows if limit is None else rows[:limit]


def split_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    train: list[dict[str, str]] = []
    valid: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        (valid if index % 5 == 0 else train).append(row)
    return train, valid


def train_model(rows: list[dict[str, str]]) -> tuple[dict[str, Counter[str]], Counter[str], set[str]]:
    token_counts: dict[str, Counter[str]] = defaultdict(Counter)
    label_counts: Counter[str] = Counter()
    vocabulary: set[str] = set()
    for row in rows:
        label = row["Type"]
        label_counts[label] += 1
        tokens = tokenize(row["Tweet"])
        token_counts[label].update(tokens)
        vocabulary.update(tokens)
    return token_counts, label_counts, vocabulary


def predict(
    text: str,
    token_counts: dict[str, Counter[str]],
    label_counts: Counter[str],
    vocabulary: set[str],
) -> str:
    total_docs = sum(label_counts.values())
    vocab_size = len(vocabulary) or 1
    scores: dict[str, float] = {}
    for label, doc_count in label_counts.items():
        total_tokens = sum(token_counts[label].values())
        score = math.log(doc_count / total_docs)
        for token in tokenize(text):
            score += math.log((token_counts[label][token] + 1) / (total_tokens + vocab_size))
        scores[label] = score
    return max(scores, key=scores.get)


def evaluate(rows: list[dict[str, str]]) -> str:
    train, valid = split_rows(rows)
    model = train_model(train)
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for row in valid:
        confusion[row["Type"]][predict(row["Tweet"], *model)] += 1

    labels = sorted(confusion)
    correct = sum(confusion[label][label] for label in labels)
    total = sum(sum(counts.values()) for counts in confusion.values())
    lines = [
        "# Spam Detection Demo",
        "",
        f"Rows used: {len(rows)}",
        f"Training rows: {len(train)}",
        f"Validation rows: {len(valid)}",
        f"Validation accuracy: {correct / total:.3f}",
        "",
        "Confusion matrix (actual -> predicted counts):",
    ]
    for label in labels:
        cells = ", ".join(f"{predicted}={confusion[label][predicted]}" for predicted in sorted(confusion[label]))
        lines.append(f"- {label}: {cells}")
    return "\n".join(lines) + "\n"


def main() -> None:
    output = evaluate(load_rows(limit=2500))
    output_path = ROOT / "outputs" / "demo_summary.md"
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
