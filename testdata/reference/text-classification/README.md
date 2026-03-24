# Transformers Text Classification Reference

This directory holds the first external reference path for InferGo parity work.

- `sst2-inputs.json` is the public-safe text input set.
- `distilbert-sst2-reference.json` is generated from a Hugging Face Transformers model.
- `../../native/text-classification/distilbert-sst2-token-id-bag/` contains the first InferGo-native bundle generated from the same reference set.
- TorchScript export bundles are generated under `dist/torchscript/distilbert-sst2/` and are intentionally not committed.
- The native Go candidate path requires `CGO_ENABLED=1`, `-tags torchscript_native`, and a libtorch install exposed through `CGO_CXXFLAGS` and `CGO_LDFLAGS`.

Generate or refresh the reference file from the repo root with:

```bash
uv run --with torch==2.10.0 --with transformers==5.3.0 python ./scripts/transformers_text_classification_reference.py
```

Export the reference model to TorchScript and run the local candidate path with:

```bash
uv run --with torch==2.10.0 --with transformers==5.3.0 python ./scripts/export_transformers_torchscript.py
source ./scripts/setup_libtorch_local.sh
go run -tags torchscript_native ./cmd/infergo-parity \
  -reference ./testdata/reference/text-classification/distilbert-sst2-reference.json \
  -torchscript-bundle-dir ./dist/torchscript/distilbert-sst2 \
  -candidate-output ./dist/torchscript/distilbert-sst2/candidate.json \
  -tolerance 1e-4
```

Generate or refresh the default InferGo-native `embedding-avg-pool` bundle and
run the Go-only native candidate path with:

```bash
go run ./internal/tools/nativebundlegen \
  -reference ./testdata/reference/text-classification/distilbert-sst2-reference.json \
  -output-dir ./testdata/native/text-classification/distilbert-sst2-embedding-avg-pool

go run ./cmd/infergo-parity \
  -reference ./testdata/reference/text-classification/distilbert-sst2-reference.json \
  -infergo-bundle-dir ./testdata/native/text-classification/distilbert-sst2-embedding-avg-pool \
  -candidate-output ./testdata/native/text-classification/distilbert-sst2-embedding-avg-pool/candidate.json \
  -tolerance 1e-4
```

Regenerate the earlier `token-id-bag` baseline explicitly with:

```bash
go run ./internal/tools/nativebundlegen \
  -mode token-id-bag \
  -reference ./testdata/reference/text-classification/distilbert-sst2-reference.json \
  -output-dir ./testdata/native/text-classification/distilbert-sst2-token-id-bag
```
