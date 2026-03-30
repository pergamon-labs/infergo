# Architecture

InferGo is structured around a small public API and explicit backend boundaries.

## Public API

The `infer` package is the stable entrypoint for loading models and running
inference from Go code.

The design goal is to keep end-user code focused on serving concerns rather than low-level runtime details.

Current stable surfaces:

- `infer.LoadTextClassifier(...)`
- `infer.LoadTokenClassifier(...)`
- typed `TextInput` / `TextPrediction`
- typed `TokenInput` / `TokenPrediction`

Those APIs intentionally expose checked-in native bundle behavior without
forcing callers to know about `backends/bionet`.

The `infer/packs` package is the curated convenience layer above that stable
bundle API. It is where checked-in manifests, reference-aware helpers,
piece-aware prediction paths, and the first truly native raw-text text pack
live. This keeps repo-shipped pack workflows out of the lower-level stable
surface.

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

The first InferGo-native external comparison paths also live here today:

- a simple `token-id-bag -> linear` bundle
- a more expressive `embedding -> avg pool -> linear` bundle with compact dense token embeddings
- a sequence-aware `embedding -> masked avg pool -> linear` bundle with compact dense token embeddings
- a narrow `windowed token embedding -> linear` bundle for token classification against a widened public NER reference set

Those keep the bundle format and Go-only serving loop moving forward without
claiming general transformer support too early.

The new token-classification path is intentionally scoped to score non-special,
non-punctuation tokens from a public-safe reference set. It uses only a tiny
prev/current/next local window, which is enough to fix BIO and nearby-context
conflicts in the widened NER set without claiming contextual encoder support
the runtime does not yet have.

Layer normalization is now available in the BIOnet runtime as an activation
primitive, but it remains experimental in the native bundle generator until it
meets the same parity bar as the default masked-pooling path.

### `backends/torchscript`

This package is reserved for a narrow, parity-tested exported-model path. It should remain a backend-specific boundary around libtorch/TorchScript details instead of leaking into the public API.

## Parity tooling

Parity is a first-class part of the architecture, not a side script.

The `internal/parity` package and `cmd/infergo-parity` command are intended to support:

- fixed public input sets
- reference-output capture
- tolerance-based comparisons
- layer-by-layer debugging when needed

The checked-in pack manifests are the public-facing contract for supported
proof packs. They drive both contributor workflows and the curated `infer/packs`
helper layer, so pack-specific conveniences stay tied to explicit parity-backed
artifacts instead of leaking into generic runtime claims.

The first raw-text-capable text pack uses a native BasicTokenizer projection of
the SST-2 proof set. That is intentionally narrower than claiming generic raw
text support across the whole manifest, but it gives InferGo one fully honest
end-to-end raw-text serving path.

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
