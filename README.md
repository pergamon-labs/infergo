# InferGo

InferGo is a Go-native inference and model-serving toolkit for backend services.

This local scaffold exists to make the extraction plan concrete before a public GitHub repository is created. The initial v1 story is intentionally narrow:

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
- [`backends/torchscript/`](./backends/torchscript) is reserved for a narrow, parity-tested backend path
- [`docs/parity-spike-01.md`](./docs/parity-spike-01.md) defines the first concrete parity spike
- [`cmd/infergo-parity/`](./cmd/infergo-parity) runs the current parity harness
- [`testdata/parity/text-classification/`](./testdata/parity/text-classification) contains the first public-safe artifact fixture
- [`scripts/transformers_text_classification_reference.py`](./scripts/transformers_text_classification_reference.py) generates the first external Transformers reference file
- [`scripts/export_transformers_torchscript.py`](./scripts/export_transformers_torchscript.py) plus the native `torchscript` backend define the first TorchScript export/run path
- [`scripts/setup_libtorch_local.sh`](./scripts/setup_libtorch_local.sh) prepares a local libtorch install and exports the native build flags
- [`COMPATIBILITY.md`](./COMPATIBILITY.md) keeps public support claims narrow and explicit

Next milestone:

1. validate the native TorchScript backend on a machine with `libtorch` available and promote it from scaffolded to supported
2. widen the reference input set and tolerance coverage beyond four fixed cases
3. decide whether the first public backend story stays TorchScript-first or shifts toward an InferGo-native artifact format
