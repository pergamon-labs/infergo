# Transformers Text Classification Reference

This directory holds the first external reference path for InferGo parity work.

- `sst2-inputs.json` is the public-safe text input set.
- `distilbert-sst2-reference.json` is generated from a Hugging Face Transformers model.

Generate or refresh the reference file from the repo root with:

```bash
uv run --with torch==2.10.0 --with transformers==5.3.0 python ./scripts/transformers_text_classification_reference.py
```
