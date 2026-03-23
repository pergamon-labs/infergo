#!/usr/bin/env python3
"""Generate a public-safe Transformers text-classification reference file."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import __version__ as transformers_version


DEFAULT_MODEL_ID = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
DEFAULT_INPUT_PATH = Path("testdata/reference/text-classification/sst2-inputs.json")
DEFAULT_OUTPUT_PATH = Path("testdata/reference/text-classification/distilbert-sst2-reference.json")


@dataclass
class InputCase:
    id: str
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face model id")
    parser.add_argument("--input", default=str(DEFAULT_INPUT_PATH), help="path to the public input json")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="path to the generated reference json")
    parser.add_argument("--max-length", type=int, default=128, help="max tokenizer length")
    return parser.parse_args()


def softmax(logits: list[float]) -> list[float]:
    tensor = torch.tensor(logits, dtype=torch.float32)
    return torch.softmax(tensor, dim=-1).tolist()


def load_inputs(path: Path) -> tuple[str, list[InputCase]]:
    payload = json.loads(path.read_text())
    cases = [InputCase(id=item["id"], text=item["text"]) for item in payload["cases"]]
    return payload["name"], cases


def build_reference(model_id: str, input_name: str, cases: list[InputCase], max_length: int) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval()

    labels = [label for _, label in sorted(model.config.id2label.items())]
    output_cases: list[dict[str, Any]] = []

    with torch.no_grad():
        for case in cases:
            encoded = tokenizer(case.text, return_tensors="pt", truncation=True, max_length=max_length)
            result = model(**encoded)
            logits = result.logits[0].detach().cpu().tolist()
            probabilities = softmax(logits)
            predicted_idx = max(range(len(probabilities)), key=probabilities.__getitem__)

            output_cases.append(
                {
                    "id": case.id,
                    "text": case.text,
                    "tokens": tokenizer.convert_ids_to_tokens(encoded["input_ids"][0].tolist()),
                    "input_ids": encoded["input_ids"][0].tolist(),
                    "attention_mask": encoded["attention_mask"][0].tolist(),
                    "expected_logits": logits,
                    "expected_probabilities": probabilities,
                    "expected_label": labels[predicted_idx],
                }
            )

    return {
        "name": f"{input_name} ({model_id})",
        "source": "huggingface-transformers",
        "model_id": model_id,
        "task": "text-classification",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "transformers_version": transformers_version,
        "torch_version": torch.__version__,
        "labels": labels,
        "cases": output_cases,
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    input_name, cases = load_inputs(input_path)
    reference = build_reference(args.model_id, input_name, cases, args.max_length)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(reference, indent=2) + "\n")

    print(f"wrote reference file to {output_path}")


if __name__ == "__main__":
    main()
