# InferGo Alpha Native Artifact Contract

This document defines the **native artifact contract** for InferGo's first
supported alpha model family.

It is the bundle-level companion to:

- [`docs/alpha-supported-model-family.md`](./alpha-supported-model-family.md)
- [`docs/alpha-roadmap.md`](./alpha-roadmap.md)
- [`docs/alpha-family-2-entres-bridge.md`](./alpha-family-2-entres-bridge.md)

The goal is to make the alpha roadmap operational:

- a supported model can be exported into an InferGo-native bundle
- that bundle can be loaded in Go
- that bundle can be served over HTTP
- loader behavior and compatibility checks are explicit

## Scope

This contract applies to the first supported alpha family only:

- **PyTorch-origin, Hugging Face Transformers-style, encoder-only text
  classification and pair-scoring models**

This contract is intentionally narrower than:

- arbitrary `.pt` or `.pth` loading
- arbitrary PyTorch graph execution
- token classification
- embeddings-only families
- ONNX as a primary format

It does **not** apply to the internal family-2 numeric-feature TorchScript
bridge.

## Contract status

- stage: alpha planning
- intended use: export/import contract for supported-family bundles
- current repo relation: current checked-in bundles are useful prior art, but
  they should be treated as pre-alpha prototypes rather than the final alpha
  contract

## Design goals

The native artifact contract should be:

- **versioned**
- **self-describing**
- **loadable without Python at runtime**
- **explicit about tokenizer behavior**
- **explicit about prediction semantics**
- **strict enough that loader failures are understandable**

## Bundle layout

The alpha bundle should be a directory with this shape:

```text
my-model-bundle/
  metadata.json
  model.gob
  labels.json
  tokenizer/
    manifest.json
    tokenizer.json
    tokenizer_config.json
    special_tokens_map.json
    vocab.txt
    merges.txt
```

Not every tokenizer file will be required for every tokenizer kind, but
`tokenizer/manifest.json` must exist whenever the bundle claims raw-text
support.

## Required top-level files

### `metadata.json`

This is the canonical bundle manifest.

It describes:

- bundle version
- backend/runtime contract
- supported task
- supported model family
- artifact file names
- tokenizer behavior
- prediction semantics
- backend-specific projection config

### `model.gob`

This is the InferGo-native model artifact consumed by the Go runtime.

For alpha, the contract assumes:

- one native model artifact per bundle
- no external PyTorch runtime dependency

### `labels.json`

This is the canonical ordered label list for the bundle.

For the first alpha family, label metadata is required even for pairwise
entity-resolution style models. If the source model is conceptually a scorer,
alpha should still export it in a label-oriented way, for example:

- `["non_match", "match"]`

That keeps the serving and API contract simple.

### `tokenizer/`

This directory contains tokenizer assets and tokenizer metadata.

It is required whenever:

- the bundle claims raw-text support
- the bundle claims paired-text support through raw-text encoding

## Canonical `metadata.json` fields

The alpha contract should require at least these fields:

```json
{
  "bundle_format": "infergo-native",
  "bundle_version": "1.0",
  "family": "encoder-text-classification",
  "task": "text-classification",
  "backend": "bionet",
  "backend_artifact": "model.gob",
  "model_id": "example/model",
  "source": {
    "framework": "pytorch",
    "ecosystem": "transformers",
    "weights_format": "safetensors"
  },
  "inputs": {
    "raw_text_supported": true,
    "pair_text_supported": true,
    "tokenized_input_supported": true,
    "max_sequence_length": 256
  },
  "tokenizer": {
    "manifest": "tokenizer/manifest.json"
  },
  "outputs": {
    "kind": "label_logits",
    "labels_artifact": "labels.json",
    "positive_label": "match"
  },
  "backend_config": {
    "feature_mode": "embedding-masked-avg-pool",
    "feature_token_ids": [101, 102, 2023],
    "embedding_artifact": "embeddings.gob"
  },
  "created_at": "2026-04-08T00:00:00Z",
  "created_by": {
    "tool": "infergo-export",
    "version": "0.1.0-alpha"
  }
}
```

## Field requirements

### Bundle identity and versioning

Required:

- `bundle_format`
- `bundle_version`
- `family`
- `task`
- `backend`
- `backend_artifact`

Rules:

- `bundle_format` must equal `infergo-native`
- `bundle_version` must use `major.minor`
- alpha loaders may accept newer minor versions within the same major version
- alpha loaders must reject unsupported major versions

### Source metadata

Required:

- `model_id`
- `source.framework`

Recommended:

- `source.ecosystem`
- `source.weights_format`
- `source.repo_url`
- `source.revision`

This is not required for runtime correctness, but it is required for
traceability and parity documentation.

### Input contract

Required:

- `inputs.raw_text_supported`
- `inputs.pair_text_supported`
- `inputs.tokenized_input_supported`
- `inputs.max_sequence_length`

Rules:

- `raw_text_supported=true` requires `tokenizer.manifest`
- `pair_text_supported=true` means the tokenizer/export path must define how two
  raw strings are encoded into one sequence
- `tokenized_input_supported=true` means callers may provide explicit
  `input_ids` and `attention_mask`

### Tokenizer reference

Required when `raw_text_supported=true`:

- `tokenizer.manifest`

Optional but recommended:

- `tokenizer.kind`
- `tokenizer.raw_text_normalization`

The canonical source of tokenizer behavior should still be the tokenizer
manifest file.

### Output contract

Required:

- `outputs.kind`
- `outputs.labels_artifact`

Optional:

- `outputs.positive_label`
- `outputs.negative_label`
- `outputs.threshold`

Rules:

- for the first alpha family, `outputs.kind` should be `label_logits`
- binary entity-resolution models should still export labels, not a free-form
  scalar-only output
- `positive_label` is recommended for binary classification and pair scoring

### Backend config

Required:

- `backend_config.feature_mode`
- `backend_config.feature_token_ids`

Required when the feature mode uses embeddings:

- `backend_config.embedding_artifact`

For the current BIOnet alpha runtime, this backend config is not optional. The
generic bundle contract must still tell the runtime how token ids are projected
into feature vectors before the native model head runs.

## Current BIOnet backend config assumptions

For the first alpha family, BIOnet currently expects one of:

- `token-id-bag`
- `embedding-avg-pool`
- `embedding-masked-avg-pool`

That backend-specific projection config is part of the alpha contract until the
native runtime evolves to absorb more of that structure directly into the model
artifact.

## `labels.json` contract

`labels.json` should contain an ordered label list:

```json
{
  "labels": ["non_match", "match"]
}
```

Rules:

- label order must match output-logit order
- labels must be unique
- labels must be stable across model revisions unless a breaking output change
  is intended

## Tokenizer asset contract

The tokenizer contract should be explicit and file-backed.

### `tokenizer/manifest.json`

This file should define:

```json
{
  "kind": "wordpiece",
  "raw_text_supported": true,
  "pair_text_supported": true,
  "special_tokens": {
    "cls_token": "[CLS]",
    "sep_token": "[SEP]",
    "pad_token": "[PAD]",
    "unk_token": "[UNK]"
  },
  "files": {
    "vocab": "vocab.txt",
    "tokenizer_config": "tokenizer_config.json",
    "special_tokens_map": "special_tokens_map.json"
  }
}
```

or, for tokenizers that support it:

```json
{
  "kind": "hf-tokenizer-json",
  "raw_text_supported": true,
  "pair_text_supported": true,
  "files": {
    "tokenizer_json": "tokenizer.json",
    "tokenizer_config": "tokenizer_config.json",
    "special_tokens_map": "special_tokens_map.json"
  }
}
```

### Alpha tokenizer rules

For alpha:

- tokenizer behavior must be reconstructable from committed tokenizer assets
- no Python-only hidden tokenizer behavior should be required at runtime
- tokenizer support should stay limited to the families required by the first
  supported model family

The most likely alpha tokenizer targets are:

- WordPiece-style tokenizers
- BPE-style Hugging Face tokenizers used by common encoder-only models

## Prediction semantics

The first alpha family should use one prediction contract:

- **ordered class logits**

That keeps:

- `infer` APIs simple
- HTTP serving simple
- parity comparison simple

Entity-resolution or pair-scoring models should therefore be represented as:

- binary classification
- or small-label classification

instead of introducing a separate scalar-regression contract in alpha.

## Versioning rules

### Bundle format versioning

Use `bundle_version` in `major.minor` form.

Rules:

- increment `minor` for additive metadata or non-breaking field extensions
- increment `major` for incompatible layout or semantic changes
- alpha loaders should:
  - accept exact supported major versions
  - reject unsupported major versions with a clear error
  - allow newer minor versions only when compatibility is explicitly designed

### Artifact compatibility

The loader should not guess compatibility.

If a bundle was generated for a different:

- family
- task
- backend
- unsupported major format version

the loader should fail fast.

## Loader compatibility checks

When loading a bundle, InferGo should validate at minimum:

1. `metadata.json` exists and parses
2. `bundle_format == infergo-native`
3. `bundle_version` major is supported
4. `family == encoder-text-classification`
5. `task == text-classification`
6. `backend == bionet` for the alpha native path
7. `backend_artifact` exists
8. `labels_artifact` exists and parses
9. label count matches the output dimension expected by the model head
10. `backend_config` is valid for the selected backend
11. if `raw_text_supported=true`, tokenizer manifest exists
12. tokenizer manifest references only files that exist in the bundle
13. if `pair_text_supported=true`, tokenizer manifest and input contract both
    support paired encoding
14. if `max_sequence_length` is invalid or missing, fail with a clear message

The loader should also reject:

- missing required fields
- unknown prediction kinds
- duplicate labels
- contradictory input claims

## Error behavior expectations

Loader failures should be:

- deterministic
- specific
- actionable

Good examples:

- `load bundle: unsupported bundle format "infergo-beta"`
- `load bundle: unsupported bundle version major 2`
- `load bundle: missing tokenizer manifest for raw-text-capable bundle`
- `load bundle: labels count does not match model output dimension`

## Recommended export behavior

The first alpha exporter should:

1. validate the source model belongs to the supported family
2. materialize a complete bundle directory
3. write all referenced tokenizer assets into `tokenizer/`
4. write `labels.json`
5. write `metadata.json`
6. write the native model artifact
7. optionally emit parity fixtures or parity instructions alongside the bundle

The exporter may begin as:

- a documented script
- a narrower CLI command

It does not need to begin life as a universal export framework.

The first concrete exporter contract now lives in:

- [`docs/alpha-family-1-exporter-contract.md`](./alpha-family-1-exporter-contract.md)

Current implementation note:

- the first exporter path is projection-based and single-text-only
- it writes tokenizer assets into the bundle today
- it does **not yet** claim generic raw-text serving for exported bundles

## Relation to current checked-in bundles

The current checked-in native bundles under `testdata/native/` are still useful
and should remain as parity fixtures and prior art.

But they are not yet the full alpha artifact contract because they do not
consistently carry:

- bundle format versioning
- tokenizer asset manifests
- explicit input contract fields
- explicit label-artifact separation

This document defines the direction the exporter and loader should move toward.

## What comes after this contract

Once this contract is accepted, the next implementation steps are:

1. update the loader to understand the new alpha bundle metadata
2. define the first exporter workflow for the supported family
3. choose the first private entity-resolution source model to dogfood against
4. prove parity and serving on that exported bundle
