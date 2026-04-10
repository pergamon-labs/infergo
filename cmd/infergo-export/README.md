# `infergo-export`

`infergo-export` is the installable family-1 BYOM exporter for InferGo alpha.

It lets a user:

1. write a small public-safe validation input set
2. export a supported PyTorch/Transformers sequence-classification model into
   an InferGo-native bundle
3. carry that bundle into a Go service without cloning this repo

## Install

```bash
go install github.com/pergamon-labs/infergo/cmd/infergo-export@latest
```

Export still needs Python-side model tooling at export time.

The default path is:

```bash
uv run --with torch==2.10.0 --with transformers==5.3.0
```

You do **not** need Python or PyTorch in production after the bundle is built.

## Write an input template

Single-text:

```bash
infergo-export template -kind single -out ./family1-inputs.json
```

Paired-text:

```bash
infergo-export template -kind pair -out ./family1-pairs.json
```

Edit that JSON so it contains a few representative, public-safe examples from
your task.

## Export a bundle

Single-text:

```bash
infergo-export export \
  -model distilbert/distilbert-base-uncased-finetuned-sst-2-english \
  -input ./family1-inputs.json \
  -out ./artifacts/distilbert-sst2-alpha \
  -reference-output ./artifacts/distilbert-sst2-reference.json
```

Paired-text:

```bash
infergo-export export \
  -model textattack/bert-base-uncased-MRPC \
  -input ./family1-pairs.json \
  -out ./artifacts/mrpc-alpha \
  -reference-output ./artifacts/mrpc-reference.json
```

The command prints:

- bundle path
- tokenizer kind
- whether raw text is supported
- whether paired text is supported
- the supported request/input modes

## What it supports today

- family 1 only
- PyTorch-origin, Hugging Face Transformers-style sequence classification
- single-text classification
- paired-text classification
- tokenized input always
- raw-text serving only for the current supported `hf-tokenizer-json`
  WordPiece subset

## What it does not support

- arbitrary Hugging Face models
- token classification export/import
- arbitrary tokenizer families
- raw `.pt` loading

## Common export failures

### `uv was not found in PATH`

The default exporter path uses:

```bash
uv run --with torch==2.10.0 --with transformers==5.3.0
```

Fixes:

- install `uv`
- or rerun with `-python-runner=python` and a Python environment that already
  has the required dependencies installed

### `-model-id is required when -model points to a local path`

If `-model` is a local directory, also provide a stable canonical id for the
bundle metadata:

```bash
infergo-export export \
  -model ./local-model-dir \
  -model-id myorg/my-model \
  -input ./family1-inputs.json \
  -out ./artifacts/my-model-alpha
```

### Python export step fails after the model loads

The current alpha path is narrow. Check these first:

- the source model is a Hugging Face Transformers-style sequence-classification
  model
- your input set shape matches the task
- the tokenizer stays within the current alpha runtime subset

If you need broader tokenizer or model-family support, that is likely outside
the current alpha contract rather than a simple local setup problem.

### Raw-text support is false after export

That means the bundle can still be used through tokenized input, but InferGo
did not embed tokenizer metadata into the bundle because the staged tokenizer
assets fell outside the current raw-text runtime subset.

This is expected for some models in alpha.

## Related docs

- [`docs/alpha-family-1-exporter-contract.md`](../../docs/alpha-family-1-exporter-contract.md)
- [`docs/alpha-family-1-walkthrough.md`](../../docs/alpha-family-1-walkthrough.md)
- [`docs/alpha-native-artifact-contract.md`](../../docs/alpha-native-artifact-contract.md)
