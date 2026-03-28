# InferGo

InferGo is a Go-native inference and model-serving toolkit for backend services.

This repository is the early public home for InferGo under the Pergamon Labs GitHub organization. The initial v1 story is intentionally narrow:

- CPU-first inference for small models
- a stable Go-facing serving API
- BIOnet as the first real backend path
- parity-driven validation against reference implementations for future exported model support

What InferGo is not in v1:

- a training framework
- a blanket loader for arbitrary `.pt` files
- an LLM platform
- a GPU-first system

## Quickstart

If you want the fastest path from clone to a real parity run, use the checked-in
native token-classification assets.

1. Clone the repo and enter it.
2. Run the test suite:

```bash
go test ./...
```

3. Run a first parity check with the checked-in DistilBERT NER reference and
   native bundle:

```bash
go run ./cmd/infergo-parity \
  -reference ./testdata/reference/token-classification/distilbert-ner-reference.json \
  -infergo-bundle-dir ./testdata/native/token-classification/distilbert-ner-windowed-embedding-linear \
  -tolerance 1e-4
```

4. If you want to see more checked-in token-classification packs, list the
   supported pack manifest:

```bash
uv run python ./scripts/build_token_classification_reference_pack.py --list
```

5. If you want to regenerate every checked-in token-classification reference
   pack from the shared public-safe NER corpus, install `uv`, then run:

```bash
uv run --with torch==2.10.0 --with transformers==5.3.0 \
  python ./scripts/build_token_classification_reference_pack.py
```

6. If you want to regenerate just one pack, use its manifest key:

```bash
uv run --with torch==2.10.0 --with transformers==5.3.0 \
  python ./scripts/build_token_classification_reference_pack.py \
  --pack-key distilbert-ner
```

7. The text-classification parity path now uses the same contributor workflow:

```bash
uv run python ./scripts/build_text_classification_reference_pack.py --list

uv run --with torch==2.10.0 --with transformers==5.3.0 \
  python ./scripts/build_text_classification_reference_pack.py
```

8. If you want a direct library usage example, try:

```bash
go run ./examples/bionet-classifier
```

9. If you want a tiny HTTP serving example, try:

```bash
go run ./examples/http-server
```

Current scaffold highlights:

- [`infer/`](./infer) is the stable public API layer
- [`backends/bionet/`](./backends/bionet) is the first implementation path
- [`backends/bionet/runtime/`](./backends/bionet/runtime) now contains the first extracted BIOnet runtime core
- [`backends/bionet/text_classification_bundle.go`](./backends/bionet/text_classification_bundle.go) defines the first InferGo-native bundle formats for text classification
- [`backends/bionet/token_classification_bundle.go`](./backends/bionet/token_classification_bundle.go) defines the first InferGo-native bundle format for token classification
- [`backends/torchscript/`](./backends/torchscript) is reserved for a narrow, parity-tested backend path
- [`docs/parity-spike-01.md`](./docs/parity-spike-01.md) defines the first concrete parity spike
- [`cmd/infergo-parity/`](./cmd/infergo-parity) runs the current parity harness
- [`testdata/parity/text-classification/`](./testdata/parity/text-classification) contains the first public-safe artifact fixture
- [`scripts/transformers_text_classification_reference.py`](./scripts/transformers_text_classification_reference.py) generates the first external Transformers reference file
- [`scripts/build_text_classification_reference_pack.py`](./scripts/build_text_classification_reference_pack.py) regenerates checked-in text-classification reference packs from the manifest
- [`scripts/transformers_token_classification_reference.py`](./scripts/transformers_token_classification_reference.py) generates a single external Transformers token-classification reference file
- [`scripts/build_token_classification_reference_pack.py`](./scripts/build_token_classification_reference_pack.py) regenerates checked-in token-classification reference packs from the manifest
- [`scripts/export_transformers_torchscript.py`](./scripts/export_transformers_torchscript.py) plus the native `torchscript` backend define the first TorchScript export/run path
- [`internal/tools/nativebundlegen/`](./internal/tools/nativebundlegen) generates InferGo-native text-classification bundles from the public reference set, including compact dense token embeddings for the default native path
- [`internal/tools/nativetokenbundlegen/`](./internal/tools/nativetokenbundlegen) generates InferGo-native token-classification bundles from the public NER reference set
- [`testdata/native/text-classification/`](./testdata/native/text-classification) contains checked-in native bundles used by the Go-only parity path
- [`testdata/native/token-classification/`](./testdata/native/token-classification) contains checked-in native token-classification bundles, including the current windowed local-context path
- [`testdata/reference/text-classification/model-packs.json`](./testdata/reference/text-classification/model-packs.json) and [`testdata/reference/token-classification/model-packs.json`](./testdata/reference/token-classification/model-packs.json) are the contributor-facing source of truth for supported public packs
- [`scripts/setup_libtorch_local.sh`](./scripts/setup_libtorch_local.sh) prepares a local libtorch install and exports the native build flags
- [`COMPATIBILITY.md`](./COMPATIBILITY.md) keeps public support claims narrow and explicit
- the public text-classification packs are now validated against an English DistilBERT SST-2 path, an English RoBERTa sentiment path, and a first non-English multilingual sentiment path
- the native `bionet` path is now validated on the supported token-classification model packs listed in [`testdata/reference/token-classification/model-packs.json`](./testdata/reference/token-classification/model-packs.json), without `libtorch`
- the text-classification parity path now follows the same manifest-backed contributor workflow as token classification
- [`examples/bionet-classifier`](./examples/bionet-classifier) and [`examples/http-server`](./examples/http-server) now show honest, runnable usage with checked-in bundles
- the current native token-classification path uses a tiny local-context window rather than pretending to support transformer attention
- layer normalization is now available as a BIOnet runtime activation and can be explored through the native bundle generator without changing the supported default parity path

Next milestone:

1. add a first non-English token-classification pack
2. expand examples beyond checked-in case IDs only when the public tokenization story is documented end to end
3. keep the optional TorchScript bridge healthy without letting it drive the core roadmap
