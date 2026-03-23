# Transformers Text Classification Reference

This directory holds the first external reference path for InferGo parity work.

- `sst2-inputs.json` is the public-safe text input set.
- `distilbert-sst2-reference.json` is generated from a Hugging Face Transformers model.
- TorchScript export bundles are generated under `dist/torchscript/distilbert-sst2/` and are intentionally not committed.

Generate or refresh the reference file from the repo root with:

```bash
uv run --with torch==2.10.0 --with transformers==5.3.0 python ./scripts/transformers_text_classification_reference.py
```

Export the reference model to TorchScript and run the local candidate path with:

```bash
uv run --with torch==2.10.0 --with transformers==5.3.0 python ./scripts/export_transformers_torchscript.py
uv run --with torch==2.10.0 python ./scripts/run_torchscript_text_classification.py
go run ./cmd/infergo-parity \
  -reference ./testdata/reference/text-classification/distilbert-sst2-reference.json \
  -candidate ./dist/torchscript/distilbert-sst2/candidate.json \
  -tolerance 1e-4
```
