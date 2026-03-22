# Parity Spike 01: Small Text Classification

## Goal

Prove that InferGo can load and run one small, exported text-classification model and match the Python reference implementation on a fixed public input set.

This is the first parity spike because the output contract is simpler than token classification or NER, which makes debugging and tolerance-setting much easier.

## Why this spike first

- the output surface is small and easy to compare
- it gives us a clean CPU-first inference story
- it forces us to define the parity harness early
- it creates a reusable path for later token-classification / NER work

## Scope

In scope:

- one small public text-classification model
- a documented export path into the InferGo backend artifact
- a fixed public input set
- reference outputs from Python
- InferGo output comparison with documented tolerances

Out of scope:

- multiple architectures
- direct Hugging Face repository loading
- generation tasks
- GPU acceleration

## Candidate model profile

Choose a model that is:

- small enough to run comfortably on CPU
- permissively usable for a public demo
- easy to export through a documented path
- stable enough to use in repeatable parity tests

Good model class candidates:

- small transformer text-classification models
- DistilBERT-class or BERT-tiny-class models

The exact model can stay open until the export path is selected.

## Deliverables

1. A documented export path into the backend artifact InferGo will load
2. A public fixture set of representative text inputs
3. A Python reference runner that records expected outputs
4. An InferGo parity runner that compares outputs and reports diffs
5. A short report showing pass/fail and tolerance details

## Acceptance criteria

- the model can be loaded by InferGo through a documented path
- the same public input set runs in both Python and InferGo
- output differences stay within the chosen tolerance
- repeated InferGo runs are deterministic for the same inputs
- the parity command is easy enough to demo live

## Suggested output comparison

Prefer comparing logits or probabilities directly, depending on which form is stable and easiest to expose from both implementations.

Suggested initial tolerance:

- start with `max_abs_diff <= 1e-4`
- relax only if the backend/export path justifies it and the reason is documented

## Follow-on spike

After this succeeds, move to:

- token classification / NER

That will be a better showcase for Go backend teams, but it is a more complex parity problem and should come second.
