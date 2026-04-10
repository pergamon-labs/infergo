# Compatibility

InferGo supports only the artifact types, backends, tasks, and usage paths
documented here and in the README.

InferGo does **not** claim blanket support for:

- arbitrary `.pt` files
- arbitrary Hugging Face models
- arbitrary PyTorch models

## Public alpha support

### Library usage

| Surface | Status |
| --- | --- |
| `infer.LoadTextClassifier(...)` | Supported |
| `infer.LoadTextBundle(...)` | Supported |
| `infer.LoadTokenClassifier(...)` | Supported |
| `infer/packs.LoadTextPack(...)` | Supported |
| `infer/packs.LoadTokenPack(...)` | Supported |

### HTTP usage

| Surface | Status |
| --- | --- |
| `infer/httpserver` | Supported |
| `cmd/infergo-serve` for curated text packs | Supported |
| `cmd/infergo-serve` for curated token packs | Supported |
| `cmd/infergo-serve -bundle ...` for exported family-1 text bundles | Experimental |

### BYOM

| Surface | Status |
| --- | --- |
| `cmd/infergo-export` for family-1 text classification | Experimental |
| Exported family-1 bundles loaded in Go without Python runtime | Experimental |
| Exported family-1 bundles served over HTTP | Experimental |
| Token-classification BYOM export/import | Not part of alpha |

## Supported task shape

### Family 1

Public BYOM family:

- PyTorch-origin
- Hugging Face Transformers-style
- encoder text classification
- paired-text classification

Current raw-text runtime boundary:

- validated BERT-style `hf-tokenizer-json` WordPiece subset only
- if a model falls outside that tokenizer subset, export can still produce a
  tokenized-input bundle, but it will not embed raw-text tokenizer metadata

### Curated token classification

Supported today through checked-in packs and examples:

- native token classification bundles
- raw-text-capable curated token packs where tokenizer behavior is validated
- sample NER extraction service via
  [`examples/ner-service/`](./examples/ner-service)

This is **not** yet a public BYOM export/import claim.

## Backends

| Backend | Status |
| --- | --- |
| `bionet` | Primary native backend |
| `torchscript` | Experimental, backend-specific, optional compatibility bridge that depends on `libtorch` |

## Runtime posture

- CPU-first
- Go-native runtime for supported `bionet` bundles
- no Python required at runtime for exported family-1 bundles
- Python/Transformers tooling required only at export time for family 1

## Not supported in alpha

- arbitrary `.pt` loading without a documented export path
- direct runtime loading from arbitrary Hugging Face repositories
- general transformer execution in the native backend
- token-classification BYOM export/import
- stable character-offset spans for NER extraction
- a first-class stable entity-extraction API in `infer/`
- ONNX runtime support
- gRPC serving
- training or fine-tuning
- GPU-first runtime work

## Practical reading guide

If you are evaluating InferGo today:

1. start with [README.md](./README.md)
2. use [cmd/infergo-export/README.md](./cmd/infergo-export/README.md) for BYOM
3. use [cmd/infergo-serve/README.md](./cmd/infergo-serve/README.md) only if you
   want a standalone HTTP process
4. use [docs/alpha-family-1-walkthrough.md](./docs/alpha-family-1-walkthrough.md)
   for the end-to-end supported alpha path
