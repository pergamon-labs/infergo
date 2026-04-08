#!/usr/bin/env python3
"""Export a family-1 text-classification model into an InferGo alpha bundle."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_SCRIPT = REPO_ROOT / "scripts" / "transformers_text_classification_reference.py"
NATIVE_BUNDLE_TOOL = "./internal/tools/nativebundlegen"

SUPPORTED_FEATURE_MODES = {
    "token-id-bag",
    "embedding-avg-pool",
    "embedding-masked-avg-pool",
}

COMMON_POSITIVE_LABELS = {
    "positive",
    "match",
    "entailment",
    "duplicate",
}
COMMON_NEGATIVE_LABELS = {
    "negative",
    "non_match",
    "contradiction",
    "not_duplicate",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="Hugging Face model id or local model directory",
    )
    parser.add_argument(
        "--model-id",
        help="canonical model id to record in bundle metadata; defaults to --model for non-path inputs",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="path to the public-safe text-classification input set",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="output bundle directory",
    )
    parser.add_argument(
        "--feature-mode",
        default="embedding-masked-avg-pool",
        choices=sorted(SUPPORTED_FEATURE_MODES),
        help="native BIOnet feature mode to export",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="max tokenizer length passed to the source reference generator",
    )
    parser.add_argument(
        "--bundle-version",
        default="1.0",
        help="alpha bundle version written into metadata.json",
    )
    parser.add_argument(
        "--reference-output",
        help="optional path to persist the generated source reference json",
    )
    parser.add_argument(
        "--positive-label",
        help="optional positive label override for binary bundles",
    )
    parser.add_argument(
        "--negative-label",
        help="optional negative label override for binary bundles",
    )
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def resolve_model_id(model_arg: str, model_id_override: str | None) -> str:
    if model_id_override:
        return model_id_override

    model_path = Path(model_arg)
    if model_path.exists():
        raise SystemExit("--model-id is required when --model points to a local path")

    return model_arg


def detect_repo_url(model_arg: str) -> str | None:
    if Path(model_arg).exists():
        return None
    if "/" not in model_arg:
        return None
    return f"https://huggingface.co/{model_arg}"


def infer_label_overrides(labels: list[str], positive: str | None, negative: str | None) -> tuple[str | None, str | None]:
    if positive and positive not in labels:
        raise SystemExit(f"--positive-label {positive!r} is not present in labels: {labels}")
    if negative and negative not in labels:
        raise SystemExit(f"--negative-label {negative!r} is not present in labels: {labels}")

    if positive or negative:
        return positive, negative

    normalized = {label.lower(): label for label in labels}
    inferred_positive = next((normalized[key] for key in COMMON_POSITIVE_LABELS if key in normalized), None)
    inferred_negative = next((normalized[key] for key in COMMON_NEGATIVE_LABELS if key in normalized), None)
    return inferred_positive, inferred_negative


def save_tokenizer_assets(model: str, target_dir: Path) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model)
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

    special_tokens = {}
    for key, value in tokenizer.special_tokens_map.items():
        if isinstance(value, str):
            special_tokens[key] = value

    manifest = {
        "kind": kind,
        "raw_text_supported": False,
        "pair_text_supported": False,
        "special_tokens": special_tokens,
        "files": files,
    }
    write_json(target_dir / "manifest.json", manifest)
    return manifest


def build_alpha_metadata(
    *,
    bundle_version: str,
    model_id: str,
    model_arg: str,
    max_length: int,
    legacy_metadata: dict[str, Any],
    labels: list[str],
    positive_label: str | None,
    negative_label: str | None,
) -> dict[str, Any]:
    source: dict[str, Any] = {
        "framework": "pytorch",
        "ecosystem": "transformers",
    }
    repo_url = detect_repo_url(model_arg)
    if repo_url:
        source["repo_url"] = repo_url

    outputs: dict[str, Any] = {
        "kind": "label_logits",
        "labels_artifact": "labels.json",
    }
    if positive_label:
        outputs["positive_label"] = positive_label
    if negative_label:
        outputs["negative_label"] = negative_label

    backend_config: dict[str, Any] = {
        "feature_mode": legacy_metadata["feature_mode"],
        "feature_token_ids": legacy_metadata["feature_token_ids"],
    }
    if legacy_metadata.get("embedding_artifact"):
        backend_config["embedding_artifact"] = legacy_metadata["embedding_artifact"]

    metadata: dict[str, Any] = {
        "bundle_format": "infergo-native",
        "bundle_version": bundle_version,
        "family": "encoder-text-classification",
        "task": "text-classification",
        "backend": "bionet",
        "backend_artifact": legacy_metadata["artifact"],
        "model_id": model_id,
        "source": source,
        "inputs": {
            "raw_text_supported": False,
            "pair_text_supported": False,
            "tokenized_input_supported": True,
            "max_sequence_length": max_length,
        },
        "tokenizer": {
            "manifest": "tokenizer/manifest.json",
        },
        "outputs": outputs,
        "backend_config": backend_config,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_by": {
            "tool": "scripts/export_encoder_text_bundle.py",
            "version": "0.1.0-alpha",
        },
    }

    if len(labels) == 2 and positive_label and negative_label:
        metadata["outputs"]["threshold"] = 0.5

    return metadata


def main() -> None:
    args = parse_args()
    model_id = resolve_model_id(args.model, args.model_id)
    output_dir = Path(args.out).resolve()
    input_path = Path(args.input).resolve()

    if not input_path.exists():
        raise SystemExit(f"input set does not exist: {input_path}")

    with tempfile.TemporaryDirectory(prefix="infergo-family1-export-") as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        source_reference_path = temp_dir / "source-reference.json"
        legacy_bundle_dir = temp_dir / "legacy-bundle"

        run(
            [
                sys.executable,
                str(REFERENCE_SCRIPT),
                "--model-id",
                args.model,
                "--input",
                str(input_path),
                "--output",
                str(source_reference_path),
                "--max-length",
                str(args.max_length),
            ]
        )

        run(
            [
                "go",
                "run",
                NATIVE_BUNDLE_TOOL,
                "-reference",
                str(source_reference_path),
                "-output-dir",
                str(legacy_bundle_dir),
                "-mode",
                args.feature_mode,
            ]
        )

        reference = load_json(source_reference_path)
        labels = reference.get("labels", [])
        if not labels:
            raise SystemExit("generated source reference does not define labels")

        positive_label, negative_label = infer_label_overrides(labels, args.positive_label, args.negative_label)

        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(legacy_bundle_dir / "model.gob", output_dir / "model.gob")
        embeddings_path = output_dir / "embeddings.gob"
        if (legacy_bundle_dir / "embeddings.gob").exists():
            shutil.copy2(legacy_bundle_dir / "embeddings.gob", embeddings_path)
        elif embeddings_path.exists():
            embeddings_path.unlink()

        tokenizer_dir = output_dir / "tokenizer"
        shutil.rmtree(tokenizer_dir, ignore_errors=True)
        save_tokenizer_assets(args.model, tokenizer_dir)

        write_json(output_dir / "labels.json", {"labels": labels})

        legacy_metadata = load_json(legacy_bundle_dir / "metadata.json")
        alpha_metadata = build_alpha_metadata(
            bundle_version=args.bundle_version,
            model_id=model_id,
            model_arg=args.model,
            max_length=args.max_length,
            legacy_metadata=legacy_metadata,
            labels=labels,
            positive_label=positive_label,
            negative_label=negative_label,
        )
        write_json(output_dir / "metadata.json", alpha_metadata)

        if args.reference_output:
            reference_output = Path(args.reference_output).resolve()
            reference_output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_reference_path, reference_output)

    print(f"wrote family-1 alpha bundle to {output_dir}")


if __name__ == "__main__":
    main()
