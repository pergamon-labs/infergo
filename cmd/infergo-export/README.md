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
- raw-text serving only for the current supported tokenizer subset

## What it does not support

- arbitrary Hugging Face models
- token classification export/import
- arbitrary tokenizer families
- raw `.pt` loading

## Related docs

- [`docs/alpha-family-1-exporter-contract.md`](../../docs/alpha-family-1-exporter-contract.md)
- [`docs/alpha-family-1-walkthrough.md`](../../docs/alpha-family-1-walkthrough.md)
- [`docs/alpha-native-artifact-contract.md`](../../docs/alpha-native-artifact-contract.md)
