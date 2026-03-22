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
- [`backends/torchscript/`](./backends/torchscript) is reserved for a narrow, parity-tested backend path
- [`docs/parity-spike-01.md`](./docs/parity-spike-01.md) defines the first concrete parity spike
- [`COMPATIBILITY.md`](./COMPATIBILITY.md) keeps public support claims narrow and explicit

Next milestone:

1. extract the BIOnet-safe runtime packages under `backends/bionet/runtime`
2. define the first public-safe example artifact set
3. build the parity harness for the first small text-classification model
