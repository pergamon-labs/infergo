# InferGo Alpha Supported Model Family

This document turns the alpha roadmap into an implementation contract.

It defines the **first model family InferGo will support for self-serve
export/import**, the source ecosystem assumptions around that family, what is
explicitly out of scope for alpha, and what families we should evaluate next.

## Status

- stage: alpha planning
- support level: design target, not broad runtime claim
- primary internal anchor: entity resolution
- public alpha posture: bring-your-own-model for one documented family

## First supported family

InferGo alpha should first support:

- **PyTorch-origin, Hugging Face Transformers-style, encoder-only text
  classification and pair-scoring models**

In practical terms, this means models that:

- are authored in PyTorch
- follow the encoder-only Transformer pattern
- produce one pooled representation per input or input pair
- feed that pooled representation into a simple prediction head

This family is broad enough to cover:

- binary text classification
- multi-class text classification
- text-pair classification
- pairwise entity-resolution scoring framed as classification or scoring

This family is narrow enough to avoid claiming:

- arbitrary `.pt` support
- arbitrary PyTorch graph execution
- general-purpose Transformer runtime coverage

## Why this family goes first

This is the best alpha family because it aligns with both the product north
star and the internal dogfood target:

- it supports the long-term product shape for entity resolution in Go without
  Python in production
- it matches common Hugging Face and PyTorch workflows closely enough to be
  useful to outside developers
- it is simpler to document and validate than starting with token-level or
  generative model families
- it keeps InferGo focused on credible toolkit breadth, not platform breadth

Private internal validation paths may exist outside this public contract, but
they are not part of this first supported family.

## Source ecosystem in scope

For alpha, InferGo should intentionally focus on the source ecosystem that most
directly serves the first supported family:

- **PyTorch** as the authoring/training ecosystem
- **Transformers** as the model/config/tokenizer ecosystem
- **Safetensors** as a preferred weight format when available

Alpha may also accept equivalent PyTorch-origin weights that are not stored in
Safetensors, but the docs should prefer Safetensors because it is clearer and
easier to reason about as a deployment input.

This means Hugging Face can be part of alpha in a useful way:

- a user starts from a supported Hugging Face-style encoder model
- they export it with InferGo's documented workflow
- they deploy the resulting InferGo-native bundle in Go

It does **not** mean:

- InferGo directly loads arbitrary Hugging Face repos at runtime
- InferGo supports every Hugging Face library listed on the models page
- InferGo should start with TensorFlow, JAX, Diffusers, GGUF, or other
  unrelated model/runtime ecosystems

## Family shape and task shape

### Supported task shapes in this family

- single-text classification
- paired-text classification
- paired-text scoring when the score can be represented by a simple output head

### Supported model shape assumptions

The first family should assume something close to:

- tokenizer -> encoder -> pooled sequence representation -> linear or small MLP
  prediction head

Alpha should allow either:

- a classifier head that returns logits over labels
- a scorer head that returns a score or a small set of logits

### Typical examples

- sentiment classifier
- topic classifier
- entity-resolution pair classifier
- duplicate-detection pair scorer

## Tokenizer assumptions for alpha

Alpha should support tokenization only as needed for the first family.

Initial tokenizer assumptions should be:

- documented, explicit tokenizer assets come with the exported bundle
- raw-text serving depends on tokenizer support being part of the bundle
- text-pair encoding must be supported for the entity-resolution path

The initial tokenizer families we should prioritize are:

- WordPiece-style tokenization
- BPE-style tokenization used by common Hugging Face encoder models

If one tokenizer family needs to be first for implementation simplicity, start
with the family required by the first internal entity-resolution export target.

Alpha should not claim:

- every tokenizer family on Hugging Face
- arbitrary custom tokenizers
- opaque Python-only tokenizer behavior reproduced by guesswork

## Export assumptions for alpha

InferGo alpha needs a documented export/import workflow for this family.

The bundle-level contract for that workflow is defined in:

- [`docs/alpha-native-artifact-contract.md`](./alpha-native-artifact-contract.md)

The exporter contract should assume:

- the source model is PyTorch-origin
- the source model belongs to the supported family
- the source model has enough configuration/tokenizer/label metadata to produce
  a complete InferGo-native bundle

The exported InferGo-native bundle should include at minimum:

- versioned bundle metadata
- model weights in InferGo-native form
- tokenizer metadata and tokenizer assets when the exported bundle supports the
  current raw-text runtime subset
- label metadata or score metadata
- enough task metadata for Go loading and REST serving

The alpha exporter is now an installable `infergo-export` command. It is not a
universal exporter, but the user workflow is now documented and reproducible
without requiring a repo checkout.

The first concrete exporter contract now lives in:

- [`docs/alpha-family-1-exporter-contract.md`](./alpha-family-1-exporter-contract.md)

Current implementation note:

- the first exporter milestone now implements:
  - single-text classification
  - paired-text classification
- pair-scoring remains part of family 1 when it can be represented by the same
  sequence-classification style output contract
- exported family-1 bundles can now serve through tokenizer-backed raw text
  when the staged tokenizer assets match the current supported BERT-style
  WordPiece tokenizer-json subset

## Runtime assumptions for alpha

The runtime only needs to grow far enough to support the first family
end-to-end.

Alpha runtime work should therefore prioritize:

- encoder-family primitives required by the chosen exported model shape
- pooled output handling
- classifier/scorer heads
- tokenizer-backed raw-text input for the supported family
- stable Go loading and HTTP serving

TorchScript may still be useful for debugging or comparison, but it should
remain:

- optional
- backend-specific
- secondary to the native export/import path

## Parity contract for alpha

For this first family, parity should mean:

- the exported InferGo-native bundle can be compared against the source
  PyTorch/Transformers model
- the comparison workflow is documented
- the comparison inputs and tolerance are reproducible

Toolkit maintainers are responsible for:

- proving the export/runtime path works for the supported family
- keeping public-safe parity fixtures and walkthroughs

Users exporting their own models are responsible for:

- validating parity on their model version before production rollout

InferGo should make that easy, but it should not hide the need for validation
when users bring their own fine-tuned models.

## What is explicitly out of scope for alpha

InferGo alpha should not claim support for:

- arbitrary `.pt` or `.pth` files
- arbitrary PyTorch graphs
- TensorFlow-authored models
- JAX-authored models
- Diffusers
- GGUF
- PEFT/adapter combinations as a general feature
- ONNX as the primary alpha path
- sentence-transformers as a separate product track unless they fit the same
  documented encoder-family export contract
- token classification as part of the **first** supported family
- generative decoding or text generation
- GPU-first runtime support

## What we should evaluate next

After the first family is working end to end, the most likely next families are:

### 1. Token classification / sequence labeling

Why next:

- it is already a strong adjacent validation path in the repo
- it supports internal NER use cases
- it pressures tokenizer and sequence-labeling support honestly

### 2. Sentence-embedding style encoder outputs

Why next:

- it broadens the usefulness of the same encoder-family foundation
- it can support retrieval and matching-style backend workflows

### 3. Additional import sources for the same family

Possible future directions:

- broader Hugging Face-origin model coverage within the same encoder family
- ONNX as an import or bridge format if it helps portability
- carefully scoped TensorFlow support only if it becomes important enough to
  justify the complexity

These should be sequenced only after the first family is real and self-serve.

## Recommendation on the Hugging Face library list

For alpha, do **not** treat the Hugging Face models-page library list as a
feature checklist.

The right alpha focus is:

- PyTorch
- Transformers
- Safetensors

The next layer to evaluate later is:

- sentence-transformers
- ONNX

Everything else should stay outside alpha unless a concrete supported-family use
case forces it in.

## Alpha acceptance check for this family

InferGo should only say this family is supported when all of the following are
true:

1. A user can follow a documented export workflow for this family.
2. The exported bundle loads in Go without Python at runtime.
3. The exported bundle serves through `infergo-serve`.
4. The parity workflow is documented and reproducible.
5. The support matrix says exactly what assumptions and limits apply.

If any of those are missing, the family is still experimental rather than
supported.
