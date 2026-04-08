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
- `infer/httpserver.NewTextPackMux(...)`
- `infer/httpserver.NewTokenPackMux(...)`

Those APIs intentionally expose checked-in native bundle behavior without
forcing callers to know about `backends/bionet`.

The `infer/packs` package is the curated convenience layer above that stable
bundle API. It is where checked-in manifests, reference-aware helpers,
piece-aware prediction paths, and the first truly native raw-text text pack
live. This keeps repo-shipped pack workflows out of the lower-level stable
surface.

The `infer/httpserver` package is the first stable REST serving layer above the
curated pack API. It keeps the current HTTP contract small and explicit:

- `GET /healthz`
- `GET /metadata`
- `POST /predict`

It also owns the first stable operational serving knobs:

- structured JSON API errors
- request logging hooks
- reusable `ServerConfig` for read, write, idle, and shutdown timeouts

The `cmd/infergo-serve` command is a thin wrapper on top of that package. It is
the current supported command-line entrypoint for serving curated text and token
packs over HTTP without forcing callers to copy the example binaries directly.
It now supports env-driven defaults and graceful shutdown on common process
signals.

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

The text-classification loader now has an explicit migration path from the
older checked-in bundle metadata to the newer alpha-era artifact contract. For
text bundles, the loader can validate a versioned `metadata.json`, external
`labels.json`, and tokenizer manifest references while still keeping legacy
fixtures working during the transition.

The first concrete family-1 exporter path sits above that loader contract as a
Python-first orchestration workflow. It currently:

- generates a source Transformers reference over a supplied public-safe input
  set
- fits the current BIOnet native bundle shape against that reference
- writes the alpha bundle metadata, labels artifact, and tokenizer asset
  manifest

That exporter is intentionally projection-based for now. It is enough to prove
export -> load -> parity for the first family-1 milestone without claiming a
full native encoder runtime yet.

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

It now also carries the first **experimental family-2 bridge** for internal
entity-resolution scoring:

- metadata-validated numeric-feature TorchScript bundles
- fixed `vectors + message -> scores` contract
- explicit separation from the public family-1 alpha contract

That bridge is allowed to exist because it helps dogfood a real internal use
case, but it must stay clearly marked as experimental/internal-first.

### `infer/experimental/entres`

This package is the first family-2 experimental API surface.

It exists to:

- load the numeric-feature TorchScript bridge bundle
- expose a narrow Go API for `vectors + message -> scores`
- support a small experimental HTTP serving path for internal dogfooding
- preserve explicitly documented bridge-compatibility details such as the
  legacy message projection used by the current screening runtime

It should not be treated as part of the stable public `infer` API yet.

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

`cmd/infergo-packs` sits on top of that same contract. It gives developers one
discovery path for the curated pack surface, including which text packs support
raw text today, without forcing them to inspect manifests by hand.

`cmd/infergo-serve` sits beside it as the first supported serving path for the
same curated pack surface. That makes InferGo's REST story more product-like
than the old example-only approach while still keeping the HTTP contract narrow
and parity-backed.

`cmd/infergo-entres-serve` is the parallel experimental command for family 2.
It is intentionally separate from `cmd/infergo-serve` so the internal dogfood
bridge does not silently broaden the stable public serving claim.

`cmd/infergo-entres-parity` is the parallel experimental parity command for
family 2. It exists so InferGo can compare the bridge output against fixtures
captured from the current screening runtime for the real individual and
organization ER models without broadening the public parity contract.

The first raw-text-capable text pack uses a native BasicTokenizer projection of
the SST-2 proof set. That is intentionally narrower than claiming generic raw
text support across the whole manifest, but it gives InferGo one fully honest
end-to-end raw-text serving path.

The first raw-text-capable token pack follows the same pattern through a
BasicTokenizer projection of the French NER proof set. That keeps the token
serving story honest too: raw text is supported where the checked-in tokenizer
behavior is native and validated, not as a blanket promise across every pack.

The benchmark story follows that same principle. The repo benchmarks the stable
public `infer/packs` surface and the stable `infer/httpserver` surface for the
currently honest raw-text-capable text and token paths, rather than claiming
generalized runtime performance across every possible backend or artifact
shape.

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
