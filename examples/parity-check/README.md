# Parity Check Example

This example demonstrates the first public-safe parity workflow using a checked-in
native pack:

```bash
go run ./cmd/infergo-parity \
  -reference ./testdata/reference/token-classification/distilbert-ner-reference.json \
  -infergo-bundle-dir ./testdata/native/token-classification/distilbert-ner-windowed-embedding-linear \
  -tolerance 1e-4
```

List the supported token-classification packs with:

```bash
uv run python ./scripts/build_token_classification_reference_pack.py --list
```

Regenerate them all, including multilingual XLM-R packs, with:

```bash
uv run --with torch==2.10.0 --with transformers==5.3.0 \
  --with sentencepiece --with protobuf --with tiktoken \
  python ./scripts/build_token_classification_reference_pack.py
```

Or list the supported text-classification packs with:

```bash
uv run python ./scripts/build_text_classification_reference_pack.py --list
```
