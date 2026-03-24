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

Current scaffold highlights:

- [`infer/`](./infer) is the stable public API layer
- [`backends/bionet/`](./backends/bionet) is the first implementation path
- [`backends/bionet/runtime/`](./backends/bionet/runtime) now contains the first extracted BIOnet runtime core
- [`backends/bionet/text_classification_bundle.go`](./backends/bionet/text_classification_bundle.go) defines the first InferGo-native bundle formats for text classification
- [`backends/torchscript/`](./backends/torchscript) is reserved for a narrow, parity-tested backend path
- [`docs/parity-spike-01.md`](./docs/parity-spike-01.md) defines the first concrete parity spike
- [`cmd/infergo-parity/`](./cmd/infergo-parity) runs the current parity harness
- [`testdata/parity/text-classification/`](./testdata/parity/text-classification) contains the first public-safe artifact fixture
- [`scripts/transformers_text_classification_reference.py`](./scripts/transformers_text_classification_reference.py) generates the first external Transformers reference file
- [`scripts/export_transformers_torchscript.py`](./scripts/export_transformers_torchscript.py) plus the native `torchscript` backend define the first TorchScript export/run path
- [`internal/tools/nativebundlegen/`](./internal/tools/nativebundlegen) generates InferGo-native text-classification bundles from the public reference set
- [`testdata/native/text-classification/`](./testdata/native/text-classification) contains checked-in native bundles used by the Go-only parity path
- [`scripts/setup_libtorch_local.sh`](./scripts/setup_libtorch_local.sh) prepares a local libtorch install and exports the native build flags
- [`COMPATIBILITY.md`](./COMPATIBILITY.md) keeps public support claims narrow and explicit

Next milestone:

1. widen the native bundle beyond the current token-id-bag spike and prove it on a larger public reference set
2. move from the current `embedding -> avg pool -> linear` native shape toward a less hand-fit and more reusable exported-model path
3. keep the optional TorchScript bridge working without letting it define the core product story
