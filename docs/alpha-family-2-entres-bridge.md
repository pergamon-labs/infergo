# InferGo Alpha Family 2: Numeric-Feature TorchScript ER Bridge

This document defines the **second alpha family track** for InferGo:

- a **numeric-feature TorchScript scorer bridge**
- aimed first at the internal `entres` entity-resolution use case
- explicitly treated as an **internal dogfood track**, not the primary public
  alpha contract

It exists so InferGo can solve a real Minerva/Pergamon deployment problem while
the public alpha contract stays clean and narrow.

## Status

- stage: alpha planning / internal dogfood implementation
- support level: experimental, internal-first
- primary use case: entity-resolution scoring over engineered numeric features
- public posture: do not describe this as the first supported public family

## Why this family exists

InferGo's public alpha family remains:

- encoder-style text and pair classification with a native InferGo artifact path

But the strongest real internal dogfood target today is different:

- the current `entres` entity-resolution model is a TorchScript scorer
- it consumes engineered numeric feature vectors plus a message vector
- it already avoids Python at runtime, but still depends on TorchScript runtime
  bindings

That makes it a very good internal proof target, but a poor candidate for the
first public model-family claim.

## Family definition

Family 2 is:

- **TorchScript-origin, numeric-feature scoring models**
- with a fixed engineered feature vector input
- plus an optional shared message vector input
- returning one scalar or score per candidate vector

This is intentionally narrower than:

- arbitrary TorchScript graph execution
- general `.pt` loading
- generic tabular model serving
- the first public InferGo alpha family

## Intended alpha role

This family should be used to:

- prove InferGo can host a real internal entity-resolution model in Go
- remove Python from the production runtime story for that workload
- exercise loading, serving, errors, config, and deployment shape on a real
  model family

This family should **not** be used to:

- redefine the public alpha support contract
- imply broad TorchScript support
- turn TorchScript into the center of InferGo's product identity

## Input and output shape

The first internal bridge should assume:

- `vectors`: `[][]float64`
- `message`: `[]float64`
- `scores`: `[]float64`

Where:

- each row in `vectors` is one candidate comparison vector
- `message` is the shared message/consensus vector used by the scorer
- each returned score corresponds to one input vector row

## Bundle assumptions

The first bridge bundle should be a directory with:

```text
my-entres-bridge/
  metadata.json
  model.torchscript.pt
```

The metadata contract should describe:

- bundle format and version
- family identifier
- task identifier
- backend (`torchscript`)
- expected vector dimension
- expected message dimension
- score semantics
- source model traceability

The first implementation uses:

```json
{
  "bundle_format": "infergo-torchscript-bridge",
  "bundle_version": "1.0",
  "family": "numeric-feature-scoring",
  "task": "entity-resolution-scoring",
  "backend": "torchscript",
  "artifact": "model.torchscript.pt",
  "model_id": "pergamon/entres-individual",
  "profile_kind": "individual",
  "source": {
    "framework": "pytorch",
    "format": "torchscript"
  },
  "inputs": {
    "vector_size": 268,
    "message_size": 268,
    "input_layout": "stacked_sample_message_channels",
    "message_strategy": "caller_supplied_consensus_vector",
    "message_projection": "legacy_first_value_broadcast"
  },
  "outputs": {
    "kind": "score_vector",
    "interpretation": "confidence"
  }
}
```

## Recommended family identifier

Use:

- `family = "numeric-feature-scoring"`
- `task = "entity-resolution-scoring"`

This keeps the family clearly separate from:

- `encoder-text-classification`

## Alpha success criteria for family 2

InferGo should consider family 2 useful for internal alpha only when:

1. the bridge bundle can be loaded through a documented InferGo path
2. the model can be served over HTTP from Go
3. the serving contract is stable enough for internal integration
4. the scoring behavior matches the existing TorchScript path closely enough to
   be trusted internally
5. the family is still clearly labeled experimental/internal in docs

The first implementation path for this is:

- bundle loader in `backends/torchscript`
- experimental Go API in `infer/experimental/entres`
- experimental serving command in `cmd/infergo-entres-serve`
- experimental bundle scaffolding command in `cmd/infergo-entres-bundle`

The first local dogfood validation has already proven:

- a real individual `entres` TorchScript artifact can be wrapped in a family-2
  bundle
- that bundle can be loaded through InferGo
- that bundle can be served from Go and return real scores
- parity can be captured against the current screening runtime and reproduced
  locally through `cmd/infergo-entres-parity`
- the same bridge contract now passes parity for both the individual and
  organization `core-svc` ER models

## Bridge metadata required for parity

The bridge metadata must carry a little more than just dimensions:

- `source.framework`
- `source.format`
- `inputs.input_layout`
- `inputs.message_strategy`
- `inputs.message_projection`
- `outputs.interpretation`

Why these fields matter:

- parity needs to know the expected tensor assembly contract
- internal users need to know whether InferGo expects a supplied message vector
  or computes one
- the current screening runtime has a legacy quirk where the message channel is
  built by broadcasting `message[0]` across the second channel, and the bridge
  needs to declare that explicitly instead of hiding it
- score interpretation needs to be explicit before we compare or threshold
  outputs

The first implementation now emits those fields in bundle metadata.

## Non-goals for family 2

Do not claim:

- support for arbitrary TorchScript models
- support for all numeric-feature models
- support for arbitrary preprocessing pipelines
- support for training or fine-tuning
- public alpha support parity with family 1

## Relationship to family 1

The intended split is:

- **Family 1**
  - public alpha contract
  - self-serve export/import
  - encoder text/pair classification
  - no PyTorch runtime dependency in production

- **Family 2**
  - internal dogfood bridge
  - numeric-feature TorchScript scorer
  - real internal entity-resolution validation
  - allowed to rely on TorchScript as a bridge while family 1 stays the public
    product story

## Recommendation

InferGo should run these tracks in parallel:

1. keep family 1 as the clean public alpha contract
2. implement family 2 as an explicit internal bridge for entity resolution
3. use family 2 to validate serving/runtime ergonomics on a real internal
   workload
4. do not let family 2 broaden the public support claims until its value and
   boundaries are proven
