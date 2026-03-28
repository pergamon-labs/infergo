# Transformers Text Classification Reference

This directory holds the first external reference path for InferGo parity work.

- `sst2-inputs.json` is the default English public-safe text input set.
- `multilingual-sentiment-inputs.json` is the first public-safe non-English
  text input set used for multilingual sentiment parity.
- `model-packs.json` is the supported public text-classification pack manifest.
- one generated `*-reference.json` file exists per supported pack in the manifest.
- `../../native/text-classification/distilbert-sst2-token-id-bag/` contains the first InferGo-native bundle generated from the same reference set.
- the default native embedding bundles now use compact dense token embeddings derived from the fitted pooled classifier instead of identity-sized embedding tables.
- the native bundle generator has an experimental `-use-layernorm` flag for the masked-pooling path, but the checked-in supported bundles still use the default head.
- TorchScript export bundles are generated under `dist/torchscript/distilbert-sst2/` and are intentionally not committed.
- The native Go candidate path requires `CGO_ENABLED=1`, `-tags torchscript_native`, and a libtorch install exposed through `CGO_CXXFLAGS` and `CGO_LDFLAGS`.

Generate or refresh the supported text-classification packs from the repo root with:

```bash
uv run --with torch==2.10.0 --with transformers==5.3.0 \
  --with sentencepiece --with protobuf --with tiktoken \
  python ./scripts/build_text_classification_reference_pack.py
```

List the supported text-classification pack keys with:

```bash
uv run python ./scripts/build_text_classification_reference_pack.py --list
```

That manifest currently includes a first non-English text-classification pack:

- `twitter-xlm-roberta-sentiment-multilingual`

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

Regenerate a single supported pack explicitly with:

```bash
uv run --with torch==2.10.0 --with transformers==5.3.0 \
  --with sentencepiece --with protobuf --with tiktoken \
  python ./scripts/build_text_classification_reference_pack.py \
  --pack-key distilbert-sst2
```

Run the Go-only native candidate path with:

```bash
go run ./cmd/infergo-parity \
  -reference ./testdata/reference/text-classification/distilbert-sst2-reference.json \
  -infergo-bundle-dir ./testdata/native/text-classification/distilbert-sst2-embedding-masked-avg-pool \
  -candidate-output ./testdata/native/text-classification/distilbert-sst2-embedding-masked-avg-pool/candidate.json \
  -tolerance 1e-4
```

The lower-level generators are still available when you want to experiment
outside the supported pack workflow. For example, regenerate the earlier
`embedding-avg-pool` bundle explicitly with:

```bash
go run ./internal/tools/nativebundlegen \
  -mode embedding-avg-pool \
  -reference ./testdata/reference/text-classification/distilbert-sst2-reference.json \
  -output-dir ./testdata/native/text-classification/distilbert-sst2-embedding-avg-pool
```

Regenerate the earlier `token-id-bag` baseline explicitly with:

```bash
go run ./internal/tools/nativebundlegen \
  -mode token-id-bag \
  -reference ./testdata/reference/text-classification/distilbert-sst2-reference.json \
  -output-dir ./testdata/native/text-classification/distilbert-sst2-token-id-bag
```
