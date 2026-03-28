#!/usr/bin/env python3
"""Build one or more public-safe token-classification reference packs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST_PATH = REPO_ROOT / "testdata/reference/token-classification/model-packs.json"
REFERENCE_SCRIPT = REPO_ROOT / "scripts/transformers_token_classification_reference.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST_PATH),
        help="path to the token-classification model pack manifest",
    )
    parser.add_argument(
        "--pack-key",
        action="append",
        dest="pack_keys",
        help="build only the selected pack key; repeat to build multiple",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list pack keys and exit",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="max tokenizer length passed to the reference generator",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if not payload.get("packs"):
        raise SystemExit(f"manifest {path} does not define any packs")
    return payload


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    manifest = load_manifest(manifest_path)

    if args.list:
        for pack in manifest["packs"]:
            print(f'{pack["key"]}: {pack["model_id"]}')
        return

    selected = manifest["packs"]
    if args.pack_keys:
        selected_keys = set(args.pack_keys)
        selected = [pack for pack in manifest["packs"] if pack["key"] in selected_keys]
        missing = selected_keys.difference({pack["key"] for pack in selected})
        if missing:
            raise SystemExit(f"unknown pack key(s): {', '.join(sorted(missing))}")

    input_path = manifest["input_set_path"]
    for pack in selected:
        print(f'building token-classification pack {pack["key"]} ({pack["model_id"]})')
        run(
            [
                sys.executable,
                str(REFERENCE_SCRIPT),
                "--model-id",
                pack["model_id"],
                "--input",
                input_path,
                "--output",
                pack["reference_path"],
                "--max-length",
                str(args.max_length),
            ]
        )
        run(
            [
                "go",
                "run",
                "./internal/tools/nativetokenbundlegen",
                "-reference",
                pack["reference_path"],
                "-output-dir",
                pack["native_bundle_dir"],
                "-mode",
                pack["native_mode"],
            ]
        )


if __name__ == "__main__":
    main()
