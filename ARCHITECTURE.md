# Architecture

InferGo is structured around a small public API and explicit backend boundaries.

## Public API

The `infer` package is the stable entrypoint for loading models, selecting backends, and running inference from Go code.

The design goal is to keep end-user code focused on serving concerns rather than low-level runtime details.

## Backend boundaries

### `backends/bionet`

This is the first real implementation path. It will absorb the reusable runtime pieces extracted from:

- `bionet/tensor`
- `bionet/functional`
- `bionet/module`
- `bionet/initializer`
- `bionet/embeddings`
- `bionet/tokenizer` after cleanup or rename

The low-level runtime stays under the BIOnet backend boundary until we know which internals are worth stabilizing as public APIs.

### `backends/torchscript`

This package is reserved for a narrow, parity-tested exported-model path. It should remain a backend-specific boundary around libtorch/TorchScript details instead of leaking into the public API.

## Parity tooling

Parity is a first-class part of the architecture, not a side script.

The `internal/parity` package and `cmd/infergo-parity` command are intended to support:

- fixed public input sets
- reference-output capture
- tolerance-based comparisons
- layer-by-layer debugging when needed

## Extraction mapping from screening

Extract early:

- `bionet/tensor` -> `backends/bionet/runtime/tensor`
- `bionet/functional` -> `backends/bionet/runtime/functional`
- `bionet/module` -> `backends/bionet/runtime/module`
- `bionet/initializer` -> `backends/bionet/runtime/initializer`
- `bionet/embeddings` -> `backends/bionet/runtime/embeddings`

Extract later or only after cleanup:

- `bionet/tokenizer`
- `cmd/bionet.go`
- `torch/*`

Do not extract as-is:

- `services/advmedia/**`
- `cmd/api.go`
- `constants/files.go`
- `services/entres/model.go`
- private `data/**` artifacts
