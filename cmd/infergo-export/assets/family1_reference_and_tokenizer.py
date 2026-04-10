#!/usr/bin/env python3
"""Generate a family-1 source reference and tokenizer assets."""

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


@dataclass
class InputCase:
    id: str
    text: str
    text_pair: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True, help="Hugging Face model id or local model directory")
    parser.add_argument("--input", required=True, help="path to the public input json")
    parser.add_argument("--reference-output", required=True, help="path to the generated reference json")
    parser.add_argument("--tokenizer-dir", required=True, help="directory to write tokenizer assets into")
    parser.add_argument("--max-length", type=int, default=128, help="max tokenizer length")
    return parser.parse_args()


def softmax(logits: list[float]) -> list[float]:
    tensor = torch.tensor(logits, dtype=torch.float32)
    return torch.softmax(tensor, dim=-1).tolist()


def load_inputs(path: Path) -> tuple[str, list[InputCase]]:
    payload = json.loads(path.read_text())
    cases = [
        InputCase(
            id=item["id"],
            text=item["text"],
            text_pair=item.get("text_pair"),
        )
        for item in payload["cases"]
    ]
    return payload["name"], cases


def tokenizer_runtime_capabilities(tokenizer_json_path: Path) -> tuple[bool, bool]:
    spec = json.loads(tokenizer_json_path.read_text())
    normalizer = spec.get("normalizer") if isinstance(spec.get("normalizer"), dict) else {}
    pre_tokenizer = spec.get("pre_tokenizer") if isinstance(spec.get("pre_tokenizer"), dict) else {}
    post_processor = spec.get("post_processor") if isinstance(spec.get("post_processor"), dict) else {}
    model = spec.get("model") if isinstance(spec.get("model"), dict) else {}

    wordpiece_supported = (
        normalizer.get("type") == "BertNormalizer"
        and pre_tokenizer.get("type") == "BertPreTokenizer"
        and post_processor.get("type") == "TemplateProcessing"
        and model.get("type") == "WordPiece"
        and isinstance(model.get("vocab"), dict)
    )
    if wordpiece_supported:
        return True, bool(post_processor.get("pair"))

    roberta_bpe_supported = (
        pre_tokenizer.get("type") == "ByteLevel"
        and post_processor.get("type") == "RobertaProcessing"
        and model.get("type") == "BPE"
        and isinstance(model.get("vocab"), dict)
        and isinstance(model.get("merges"), list)
    )
    if roberta_bpe_supported:
        return True, True

    return False, False


def save_tokenizer_assets(model_id: str, target_dir: Path, *, pair_text_requested: bool) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(target_dir)

    files: dict[str, str] = {}
    for key, filename in (
        ("tokenizer_json", "tokenizer.json"),
        ("tokenizer_config", "tokenizer_config.json"),
        ("special_tokens_map", "special_tokens_map.json"),
        ("vocab", "vocab.txt"),
        ("merges", "merges.txt"),
    ):
        if (target_dir / filename).exists():
            files[key] = filename

    if "tokenizer_json" in files:
        kind = "hf-tokenizer-json"
    elif "vocab" in files:
        kind = "wordpiece"
    else:
        raise SystemExit("tokenizer save_pretrained() did not produce a supported tokenizer file set")

    raw_text_supported = False
    pair_text_supported = False
    if "tokenizer_json" in files:
        raw_text_supported, pair_text_supported = tokenizer_runtime_capabilities(target_dir / files["tokenizer_json"])
        pair_text_supported = pair_text_requested and pair_text_supported

    special_tokens = {}
    for key, value in tokenizer.special_tokens_map.items():
        if isinstance(value, str):
            special_tokens[key] = value

    manifest = {
        "kind": kind,
        "raw_text_supported": raw_text_supported,
        "pair_text_supported": pair_text_supported,
        "special_tokens": special_tokens,
        "files": files,
    }
    (target_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def build_reference(model_id: str, input_name: str, cases: list[InputCase], max_length: int) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval()

    labels = [label for _, label in sorted(model.config.id2label.items())]
    output_cases: list[dict[str, Any]] = []

    with torch.no_grad():
        for case in cases:
            encoded = tokenizer(
                case.text,
                case.text_pair,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            result = model(**encoded)
            logits = result.logits[0].detach().cpu().tolist()
            probabilities = softmax(logits)
            predicted_idx = max(range(len(probabilities)), key=probabilities.__getitem__)

            output_cases.append(
                {
                    "id": case.id,
                    "text": case.text,
                    "text_pair": case.text_pair,
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
    tokenizer_dir = Path(args.tokenizer_dir)
    reference_output = Path(args.reference_output)

    input_name, cases = load_inputs(input_path)
    pair_text_requested = any(case.text_pair for case in cases)

    save_tokenizer_assets(args.model_id, tokenizer_dir, pair_text_requested=pair_text_requested)
    reference = build_reference(args.model_id, input_name, cases, args.max_length)

    reference_output.parent.mkdir(parents=True, exist_ok=True)
    reference_output.write_text(json.dumps(reference, indent=2) + "\n")

    print(f"wrote tokenizer assets to {tokenizer_dir}")
    print(f"wrote reference file to {reference_output}")


if __name__ == "__main__":
    main()
