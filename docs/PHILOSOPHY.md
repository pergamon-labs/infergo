# InferGo Philosophy

InferGo is a Go-native inference toolkit for backend services.

Its purpose is simple:

- export a supported model once
- load it in Go
- run predictions without Python in production

## Core principles

### 1. Production-first

InferGo is built for runtime use inside backend systems.

That means:

- library-first integration
- predictable deployment
- CPU-first posture
- narrow support claims

The default question is not "can we run this model somehow?"

It is:

"Can a backend engineer use this model safely and clearly in a Go service?"

### 2. Export once, run in Go

InferGo treats source-model ecosystems and runtime ecosystems as separate
concerns.

For the current public alpha path:

- models originate in PyTorch / Hugging Face Transformers-style workflows
- export happens with dedicated tooling
- production inference happens in Go

Python may be needed at export time. It should not be needed in production once
the InferGo bundle exists.

### 3. Parity before support claims

InferGo should not claim support for a path unless it can be compared against a
reference implementation and validated reproducibly.

This is why parity is part of the product surface instead of just test
scaffolding.

The goal is not "it seems to work."

The goal is:

- the exported bundle matches the documented source path
- the validation steps are repeatable
- the support boundary is explicit

### 4. Narrow families over vague compatibility

InferGo does not aim to load arbitrary `.pt` files or arbitrary Hugging Face
models.

Instead, it grows through documented supported families.

That keeps the project honest and useful:

- users know what is supported
- maintainers know what must be validated
- the runtime can stay small and understandable

### 5. Library-first, HTTP optional

The main InferGo story is embedding inference directly inside existing Go
services.

`infergo-serve` exists for cases where a standalone HTTP process is helpful:

- smoke testing
- process isolation
- non-Go callers
- simple deployment boundaries

HTTP is a deployment mode, not the main identity of the project.

### 6. Internal dogfooding without public overclaiming

InferGo should solve real internal problems, but it should not broaden public
claims prematurely while doing so.

Private internal validation can exist, but it should live outside the public
product contract unless and until it is safe and ready to support openly.

## What InferGo is not

InferGo is not:

- a training framework
- a general transformer runtime
- a model zoo
- a blanket `.pt` loader
- a platform-scale serving system

Those may sound adjacent, but they are not the same product.

## What success looks like

InferGo is succeeding when a backend engineer can:

1. understand the supported path quickly
2. export a supported model into an InferGo bundle
3. load that bundle in Go without Python at runtime
4. validate parity when needed
5. use it directly in a service without repo archaeology
