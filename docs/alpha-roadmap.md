# InferGo Alpha Roadmap

Maintainer-facing note:

If you are evaluating InferGo as a user, start with
[`README.md`](../README.md),
[`COMPATIBILITY.md`](../COMPATIBILITY.md), and
[`alpha-family-1-walkthrough.md`](./alpha-family-1-walkthrough.md).

This document defines the next phase for InferGo from the current public
pre-alpha state to a credible alpha release.

InferGo is already public as a narrow pre-alpha. This roadmap is about how we
move from "curated proof packs and parity demos" to "a backend engineer can
bring a supported model family, load it in Go, serve it over HTTP, and trust
the result without needing Python in production."

## North star

InferGo should become a Go-native inference toolkit for backend services where:

- a supported model can be exported into an InferGo-native artifact
- that artifact can be loaded directly in Go
- it can be served over HTTP through a stable runtime surface
- parity against a reference implementation is documented and reproducible
- Python, PyTorch, or TensorFlow are not required at runtime

This is still narrower than:

- a blanket `.pt` loader
- a general transformer runtime
- a training framework
- a Kubernetes serving platform

InferGo's public alpha track is:

- **Family 1**
  - the primary public alpha contract
  - self-serve export/import for a documented supported model family

## Product rule

InferGo should support **bring your own model for documented model families**.

It should not claim:

- "load any PyTorch model"
- "supports Hugging Face models" in the broad sense
- "supports arbitrary `.pt` files"

The alpha goal is not universal model compatibility. The alpha goal is a
credible, repeatable export/import workflow for a narrow but useful set of
PyTorch-origin inference model families.

## Hugging Face position

Hugging Face can be part of the alpha roadmap, but only in a constrained way.

Good alpha interpretation:

- support export/import for supported PyTorch/Hugging Face-origin model families
- document a workflow for converting supported Hugging Face-style models into
  InferGo-native artifacts
- keep Hugging Face as a source ecosystem, not a blanket runtime support claim

Bad alpha interpretation:

- "InferGo directly loads any Hugging Face repo"
- "InferGo supports Hugging Face models" without explicit constraints

So yes, Hugging Face can be included in alpha, but the wording should remain:

- documented export/import for supported Hugging Face-origin model families

and not:

- general direct runtime support for arbitrary Hugging Face models

## Anchor use cases

### 1. Adjacent internal validation: NER in Go services

Secondary internal validation target:

- use InferGo to load NER into Go services that currently depend on
  `legacy-core`, `core-svc`, or other Python-coupled paths
- prove that the serving/runtime contract is reusable beyond one model

Why this matters:

- it tests tokenization and sequence-labeling pressure
- it validates that InferGo is becoming a toolkit, not a one-off entity
  resolution wrapper

Alpha scope decision:

- token classification remains a curated-pack and sample-service validation path
  for alpha
- it is not the second public BYOM family yet
- alpha NER scope now includes stable byte/character offsets for curated
  raw-text flows, with entity grouping kept in sample/service code
- a first-class entity helper is not part of alpha and should be reconsidered
  only if repeated usage patterns emerge
- a later public token-classification export family can be evaluated after the
  current family-1 alpha hardening work is complete

### 3. Public alpha target: supported PyTorch-origin model family

Public alpha should expose:

- one documented export/import workflow
- one documented supported model family for text-style inference
- stable HTTP serving and Go loading for that family

Private internal dogfooding can continue separately while the public alpha
story remains narrowly centered on family 1.

## Alpha definition

InferGo should be called "alpha" when all of the following are true:

1. A user can export a supported PyTorch-origin model into an InferGo-native
   artifact with a documented workflow.
2. That artifact can be loaded from Go without PyTorch at runtime.
3. That artifact can be served through `infergo-serve`.
4. Parity against the reference implementation is reproducible and documented.
5. At least one real internal use case has been validated with the workflow.
6. The support contract is explicit enough that outside users know what will
   and will not work.

## Main workstreams

### A. Supported model-family definition

We need to explicitly choose the first family instead of drifting toward
"arbitrary model loading."

The first alpha family should likely be something close to:

- encoder-style classification / scoring models
- token classification / sequence labeling models
- simple text-pair scoring models

This family should be chosen based on:

- internal usefulness
- exportability from existing PyTorch code
- ability to reproduce parity in Go
- ability to document tokenizer/runtime constraints honestly

That family is now documented in:

- [`docs/alpha-supported-model-family.md`](./alpha-supported-model-family.md)

### B. Export / import workflow

This is the main product unlock.

We need:

- a documented exporter contract
- versioned native bundle metadata
- a loader contract in Go
- reproducible parity checks against the source model

Desired user story:

1. export a supported model using an InferGo tool or documented script
2. load the produced bundle in Go
3. serve it with `infergo-serve`
4. run parity checks against the source implementation

This workstream is about **family 1**.

The first concrete exporter contract for that workstream now lives in:

- [`docs/alpha-family-1-exporter-contract.md`](./alpha-family-1-exporter-contract.md)
- [`docs/alpha-family-1-walkthrough.md`](./alpha-family-1-walkthrough.md)

Current implementation note:

- the first family-1 exporter is now an installable `infergo-export` command
  with a projection-based, Python-first export bridge for **single-text and
  paired-text classification**
- it emits alpha-format bundles that load and parity-check in Go
- exported text bundles can now be served through `cmd/infergo-serve -bundle`
  using tokenizer-backed raw text when the staged tokenizer assets match the
  currently validated tokenizer-json runtime subsets

Adjacent NER validation is now covered in a realistic sample Go service via
[`examples/ner-service/`](../examples/ner-service). The remaining public alpha
gaps are tracked in
[`alpha-gaps-and-missing-primitives.md`](./alpha-gaps-and-missing-primitives.md).

### C. Runtime and backend work

We should keep investing in:

- native artifact format
- bundle versioning
- tokenizer support for the first supported model family
- enough runtime primitives to support the chosen family

We should avoid:

- adding broad new primitives without a model-family reason
- pushing TorchScript into the center of the product story

TorchScript can remain a bridge and debugging aid, but not the main alpha path.

### D. Serving and ops

The serving work is now on the right track. For alpha we should keep:

- stable REST surface
- structured JSON errors
- request logging and timeouts
- graceful shutdown
- benchmark guidance

We can add gRPC later, but it should not block alpha.

### E. Internal dogfooding

InferGo should continue to be validated on real internal use cases, but those
private paths should not define the public support contract unless they are
safe and ready to support openly.

### F. Public docs and examples

Alpha needs:

- install docs
- export/import docs
- library usage docs
- serving docs
- troubleshooting docs
- at least one public-safe model walkthrough end to end

## Phased roadmap

### Phase 0: Current state

Current state already achieved:

- public pre-alpha repo
- stable `infer` APIs
- curated pack helpers
- stable HTTP serving surface
- parity tooling
- benchmark story

### Phase 1: Choose the alpha family

Goal:

- narrow the first self-serve supported family

Deliverables:

- written model-family definition
- required runtime primitives list
- tokenizer/export requirements
- clear "supported vs not supported" decision for alpha
- clear "public contract vs private validation" boundary

### Phase 2: Self-serve export/import

Goal:

- replace "checked-in curated packs only" with "documented supported export path"

Deliverables:

- exporter tool or documented script path
- versioned native bundle schema
- loader compatibility checks
- parity workflow for exported artifacts

This phase applies to **family 1**.

### Phase 3: Internal dogfood validation

Goal:

- prove the workflow on real internal models

Deliverables:

- NER validation in a Go service or sample service
- documented findings on gaps, failure modes, and missing primitives
- private internal validation that does not muddy the public family-1 story

Current status:

- NER is validated in a public-safe sample Go service
- remaining alpha gaps are now documented in
  [`alpha-gaps-and-missing-primitives.md`](./alpha-gaps-and-missing-primitives.md)

### Phase 4: Public alpha hardening

Goal:

- make the public repo credible for outside adoption

Deliverables:

- one or more public-safe example models using the same export/import contract
- end-to-end walkthrough docs
- updated benchmark and support docs
- issue backlog aligned to the alpha support contract
- alpha release notes

## What we should not do

- Do not turn InferGo into a vague "supports PyTorch" project.
- Do not broaden support claims faster than parity and docs.
- Do not optimize for pack count over model-family clarity.
- Do not make TorchScript/libtorch the default runtime dependency story.
- Do not chase platform breadth before toolkit breadth is credible.

## Immediate next steps

1. Close or consciously defer the gaps listed in
   [`alpha-gaps-and-missing-primitives.md`](./alpha-gaps-and-missing-primitives.md)
   before cutting alpha.
2. Keep token classification as a curated-pack/sample-service alpha story and
   only revisit a second public BYOM family after the current alpha hardening
   work lands.
3. Tighten the family-1 exporter and tokenizer story until a stranger can
   follow the supported path without repo archaeology.
