#!/usr/bin/env python3
"""Run a TorchScript text-classification artifact against a reference input set."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


DEFAULT_REFERENCE_PATH = Path("testdata/reference/text-classification/distilbert-sst2-reference.json")
DEFAULT_BUNDLE_DIR = Path("dist/torchscript/distilbert-sst2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default=str(DEFAULT_REFERENCE_PATH), help="reference json with token ids and masks")
    parser.add_argument("--bundle-dir", default=str(DEFAULT_BUNDLE_DIR), help="directory containing model.torchscript.pt and metadata.json")
    parser.add_argument("--output", default="", help="optional candidate output path")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def pad_to_length(values: list[int], target_length: int, pad_value: int) -> list[int]:
    if len(values) >= target_length:
        return values[:target_length]
    return values + [pad_value] * (target_length - len(values))


def softmax(logits: list[float]) -> list[float]:
    tensor = torch.tensor(logits, dtype=torch.float32)
    return torch.softmax(tensor, dim=-1).tolist()


def main() -> None:
    args = parse_args()
    reference_path = Path(args.reference)
    bundle_dir = Path(args.bundle_dir)
    metadata_path = bundle_dir / "metadata.json"
    artifact_path = bundle_dir / "model.torchscript.pt"
    output_path = Path(args.output) if args.output else bundle_dir / "candidate.json"

    reference = load_json(reference_path)
    metadata = load_json(metadata_path)

    model = torch.jit.load(str(artifact_path))
    model.eval()

    cases = []
    with torch.no_grad():
        for case in reference["cases"]:
            input_ids = pad_to_length(case["input_ids"], metadata["sequence_length"], metadata["pad_token_id"])
            attention_mask = pad_to_length(case["attention_mask"], metadata["sequence_length"], 0)

            logits_tensor = model(
                torch.tensor([input_ids], dtype=torch.long),
                torch.tensor([attention_mask], dtype=torch.long),
            )

            logits = logits_tensor[0].detach().cpu().tolist()
            probabilities = softmax(logits)
            predicted_idx = max(range(len(probabilities)), key=probabilities.__getitem__)

            cases.append(
                {
                    "id": case["id"],
                    "text": case["text"],
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "observed_logits": logits,
                    "observed_probabilities": probabilities,
                    "observed_label": metadata["labels"][predicted_idx],
                }
            )

    payload = {
        "name": f"{reference['name']} TorchScript candidate",
        "source": "torchscript",
        "model_id": metadata["model_id"],
        "task": metadata["task"],
        "artifact": str(artifact_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "labels": metadata["labels"],
        "cases": cases,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote candidate file to {output_path}")


if __name__ == "__main__":
    main()
