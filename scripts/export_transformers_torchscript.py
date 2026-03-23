#!/usr/bin/env python3
"""Export a Hugging Face sequence-classification model to TorchScript."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import __version__ as transformers_version


DEFAULT_MODEL_ID = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
DEFAULT_REFERENCE_PATH = Path("testdata/reference/text-classification/distilbert-sst2-reference.json")
DEFAULT_OUTPUT_DIR = Path("dist/torchscript/distilbert-sst2")


class LogitsOnlyWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face model id")
    parser.add_argument("--reference", default=str(DEFAULT_REFERENCE_PATH), help="reference json used to define the tracing shape")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="output directory for the TorchScript bundle")
    return parser.parse_args()


def load_reference(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def pad_to_length(values: list[int], target_length: int, pad_value: int) -> list[int]:
    if len(values) >= target_length:
        return values[:target_length]
    return values + [pad_value] * (target_length - len(values))


def main() -> None:
    args = parse_args()
    reference = load_reference(Path(args.reference))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_id)
    model.eval()

    wrapper = LogitsOnlyWrapper(model)
    max_seq_len = max(len(case["input_ids"]) for case in reference["cases"])
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    example_case = reference["cases"][0]
    example_input_ids = torch.tensor(
        [pad_to_length(example_case["input_ids"], max_seq_len, pad_token_id)],
        dtype=torch.long,
    )
    example_attention_mask = torch.tensor(
        [pad_to_length(example_case["attention_mask"], max_seq_len, 0)],
        dtype=torch.long,
    )

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (example_input_ids, example_attention_mask), strict=False)

    artifact_path = output_dir / "model.torchscript.pt"
    traced.save(str(artifact_path))

    metadata = {
        "name": f"{reference['name']} TorchScript export",
        "source": "huggingface-transformers-export",
        "model_id": args.model_id,
        "task": "text-classification",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "transformers_version": transformers_version,
        "torch_version": torch.__version__,
        "artifact": artifact_path.name,
        "labels": [label for _, label in sorted(model.config.id2label.items())],
        "pad_token_id": pad_token_id,
        "sequence_length": max_seq_len,
    }

    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote torchscript bundle to {output_dir}")


if __name__ == "__main__":
    main()
