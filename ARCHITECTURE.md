# Architecture

InferGo is structured around a small public API, explicit backend boundaries,
and parity-backed artifact contracts.

## Public surfaces

### `infer`

The `infer` package is the stable entrypoint for loading bundles and running
inference from Go code.

Key surfaces:

- `infer.LoadTextClassifier(...)`
- `infer.LoadTextBundle(...)`
- `infer.LoadTokenClassifier(...)`
- `TextInput` / `TextPrediction`
- `TokenInput` / `TokenPrediction`

This is the primary product surface.

### `infer/packs`

This is the curated convenience layer for checked-in proof packs.

Use it when you want:

- a repo-shipped pack by key
- curated examples
- sample NER flows built on validated public-safe fixtures

### `infer/httpserver`

This is the stable HTTP adapter layer.

It exists for teams who want:

- a standalone HTTP process
- quick smoke tests
- a simple REST boundary around a loaded bundle

It is optional. Most Go teams can embed InferGo directly in their existing
service code instead.

### CLI entrypoints

- `cmd/infergo-export`
  - installable family-1 BYOM exporter
- `cmd/infergo-serve`
  - installable standalone HTTP server
- `cmd/infergo-parity`
  - parity check tool
- `cmd/infergo-packs`
  - curated-pack discovery

## Backends

### `backends/bionet`

This is the primary native backend.

It currently powers:

- native text-classification bundles
- native token-classification bundles
- exported family-1 text bundles

It is the main Go-native runtime path for alpha.

### `backends/torchscript`

This is an experimental backend-specific bridge.

It exists for the internal family-2 `entres` dogfood path and should not be
treated as the default InferGo runtime story.

## Artifact model

InferGo is centered on exported bundle directories, not raw training artifacts.

For the public family-1 path, a bundle contains:

- `metadata.json`
- `labels.json`
- BIOnet runtime artifacts such as `model.gob`
- tokenizer assets when raw-text runtime support is available

The relevant user story is:

1. export a supported model into a bundle
2. load the bundle in Go
3. call it directly or expose it over HTTP
4. verify parity against the source model

## Family split

### Family 1

Public alpha BYOM contract:

- PyTorch-origin
- Hugging Face Transformers-style
- encoder text classification
- paired-text classification

### Family 2

Internal-first bridge:

- numeric-feature TorchScript scorer
- current `entres` use case
- experimental and not part of the public alpha BYOM claim

## Parity

Parity is part of the product, not a side tool.

InferGo uses parity to make support claims trustworthy:

- exported bundles are compared against a source reference implementation
- curated proof packs are tied to explicit reference data
- new support claims should come with parity or golden-test evidence

## Practical interpretation

If you are adopting InferGo today:

- use `infer` first
- use `infer/httpserver` or `infergo-serve` only when you want a standalone
  HTTP process
- treat `bionet` as the primary runtime path
- treat TorchScript as experimental and internal-first
