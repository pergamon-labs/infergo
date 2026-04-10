# Alpha Gaps and Missing Primitives

Maintainer-facing note:

If you are evaluating InferGo as a user, start with
[`README.md`](../README.md) and
[`alpha-family-1-walkthrough.md`](./alpha-family-1-walkthrough.md).

This document captures the remaining gaps between the current public
pre-alpha state and a credible alpha release.

It is intentionally scoped to the current alpha strategy:

- **family 1**
  - public alpha contract
  - PyTorch-origin, Transformers-style, encoder text and paired-text
    classification export/import

The goal is to make the remaining work concrete without broadening support
claims beyond what InferGo is actually proving.

## Current validated state

The following are already true:

- family-1 export now has an installable `infergo-export` CLI
- family-1 export can start from a generated single-text or paired-text input
  template instead of repo-local fixture files
- family-1 exported bundles can load in Go
- family-1 exported bundles can serve raw text for the current validated
  tokenizer-json subsets:
  - BERT-style WordPiece + `TemplateProcessing`
  - RoBERTa-style ByteLevel BPE + `RobertaProcessing`
- family-1 export, load, serve, and parity are documented in
  [`alpha-family-1-walkthrough.md`](./alpha-family-1-walkthrough.md)
- token classification is now validated in a realistic sample Go service via
  [`examples/ner-service/`](../examples/ner-service)

## Current alpha decision

For alpha:

- family 1 remains the only public BYOM family
- token classification remains a curated-pack and sample-service story
- token classification export/import is deferred to a later public family

This keeps the alpha claim narrow while preserving the realistic NER validation
path we already have.

## Must close before alpha

These are the gaps that still matter for calling InferGo a credible alpha
toolkit.

### 1. Family-1 export needs a cleaner developer UX

Current state:

- the family-1 exporter is now an installable CLI
- it no longer requires a repo checkout
- it can write starter input templates for single-text and paired-text flows

Needed:

- one more pass on failure messaging from a stranger's point of view
- clearer troubleshooting for missing `uv` or unsupported source-model layouts
- confirmation that the supported path feels obvious outside current maintainers

Why this matters:

- the remaining risk is now usability polish, not exporter viability

### 2. Tokenizer support is still narrow

Current state:

- exported family-1 raw-text serving currently supports two validated
  tokenizer-json subsets:
  - BERT-style WordPiece + `TemplateProcessing`
  - RoBERTa-style ByteLevel BPE + `RobertaProcessing`

Needed:

- broader tokenizer support for the first family beyond those explicit
  tokenizer-json subsets

Why this matters:

- tokenizer assumptions are part of the product contract, not an implementation
  footnote

### 3. Token classification intentionally does not have a self-serve BYOM path yet

Current state:

- token classification is proven through curated packs and a realistic sample
  service
- token classification is **not** yet part of the public export/import contract

Needed:

- keep this boundary explicit in the docs and support matrix
- avoid language that implies token-classification BYOM is already part of alpha

Why this matters:

- outside users need to know whether token classification is "supported via
  checked-in examples" or "supported via self-serve export"

### 4. Private internal validation must stay separate from the public story

Current state:

- private internal validation exists outside the public family-1 contract

Needed:

- confirm the public docs never imply support for those private paths
- keep private runbooks and commands out of the public user path

Why this matters:

- the project is more credible when the public story stays narrow

## Important but acceptable after alpha

These are worth doing, but they should not force alpha scope expansion.

### 1. Broader tokenizer families

Examples:

- SentencePiece
- richer Hugging Face tokenizer-json support beyond the current subset

### 2. First-class gRPC surface

HTTP is the current alpha transport.
gRPC can follow once the serving contract settles further.

### 3. More public BYOM families

Examples:

- token classification export/import
- embeddings
- carefully scoped ONNX import paths

### 4. First-class NER/entity abstraction in `infer/`

The sample service proves a workable path, but alpha intentionally leaves entity
grouping at the sample/service layer until repeated usage patterns justify a
stable helper in `infer/` or `infer/packs`.

## Explicit non-goals for alpha

These should stay out of alpha planning unless the roadmap changes on purpose.

- arbitrary `.pt` loading
- blanket Hugging Face runtime loading
- general transformer execution
- training
- GPU-first runtime work
- turning TorchScript into the default InferGo story
- platform-style serving features before the toolkit path is clean

## Decision checkpoints before alpha

Before cutting alpha, InferGo should answer these clearly:

1. Is the current tokenizer support boundary documented clearly enough that
   users will not over-assume?
2. Are the public docs and examples cleanly separated from private internal
   validation paths?
