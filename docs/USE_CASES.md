# InferGo Use Cases

This guide helps you decide when InferGo is a good fit and when it is not.

## Good fits

### 1. Add inference directly to an existing Go service

This is the most natural InferGo use case.

Example:

- your API is already written in Go
- you want to run a small text classifier or scoring model in-process
- you do not want a Python sidecar in production

Why InferGo fits:

- load a bundle in Go
- call it directly from handlers or service code
- keep one runtime and one deployable service

### 2. Export a supported model once and deploy it as a Go-native artifact

Example:

- your model originates from a PyTorch / Transformers-style workflow
- you want a documented deployment format for Go
- you need a repeatable export -> load -> parity path

Why InferGo fits:

- the supported family-1 BYOM path is designed exactly for this
- export-time Python tooling does not carry into production
- the runtime story stays Go-native

### 3. Serve a supported bundle through a simple HTTP boundary

Example:

- you want a small standalone model process
- another service is not written in Go
- you want a smoke-test surface around a bundle quickly

Why InferGo fits:

- `infergo-serve` provides a stable HTTP surface
- it is easy to stand up for testing or separation
- it uses the same bundles as the embedded library path

### 4. Replace Python runtime dependencies in internal backend systems

Example:

- a model is already useful internally
- the current runtime path depends on Python, TorchScript bindings, or fragile
  sidecar logic
- you want a narrower Go-native deployment story

Why InferGo fits:

- this is one of the main reasons the project exists
- private internal validation can use InferGo without broadening the public
  support claim

### 5. Validate NER-style service flows in Go

Example:

- you need a small entity-extraction service in Go
- curated token packs are enough for the current need
- you want a realistic service shape without claiming full token-classification
  BYOM support yet

Why InferGo fits:

- the sample NER service already exercises that path
- token classification is validated as a sample-service story in alpha
- the current alpha scope includes stable token/entity offsets plus
  service-owned entity grouping, not a first-class entity-helper API

## Use with caution

### 1. Broad Hugging Face compatibility

InferGo is not yet the right tool if your requirement is:

- "load any Hugging Face repo"
- "support any tokenizer"
- "run arbitrary models without an export contract"

That is outside the current alpha scope.

### 2. Training or fine-tuning

InferGo is a runtime and deployment toolkit, not a training framework.

If your main need is:

- training loops
- fine-tuning infrastructure
- experiment management

InferGo is not the right primary tool.

### 3. Platform-scale model serving

InferGo is also not trying to be:

- a Kubernetes-native serving platform
- a high-scale batching/orchestration system
- a general multi-tenant model-hosting layer

It is better understood as a toolkit that can be embedded or served simply.

### 4. Arbitrary ONNX or TorchScript loading

InferGo may grow bridge paths over time, but today it should not be chosen for:

- arbitrary ONNX import expectations
- arbitrary TorchScript compatibility
- generic graph-runtime promises

## Current alpha recommendation

Use InferGo today if you want:

- a Go-native inference toolkit
- a documented family-1 export/import path
- library-first runtime integration
- optional HTTP serving
- narrow, parity-backed support claims

Start with:

- [`GETTING_STARTED.md`](./GETTING_STARTED.md)
- [`alpha-family-1-walkthrough.md`](./alpha-family-1-walkthrough.md)
- [`../COMPATIBILITY.md`](../COMPATIBILITY.md)
