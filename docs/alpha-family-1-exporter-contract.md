# InferGo Alpha Family-1 Exporter Contract

This document defines the **first concrete exporter workflow** for InferGo's
family-1 alpha track.

It turns the higher-level family and bundle contracts into an actual
implementation shape that we can run today:

- choose a supported source model
- generate a source reference from Transformers
- fit a native BIOnet bundle against that reference
- write the alpha bundle metadata and artifacts
- load the bundle in Go and verify parity

## Status

- stage: first concrete implementation
- support level: experimental, family-1 only
- implementation form: installable CLI with a Python-first export bridge
- current task branch: **single-text and paired-text classification**

This is the first operational exporter for family 1, but it is **not yet the
final alpha export story**.

The most important constraint is:

- the current exporter is **projection-based**, not direct-weight transformer
  execution in Go

That means the exporter does not translate an arbitrary encoder model into a
native transformer runtime. Instead, it builds an InferGo-native bundle by
capturing source-model outputs on a supplied input set and fitting the current
native BIOnet bundle shape against those examples.

## Why this exporter exists first

This exporter is the smallest honest bridge from today's repo reality to the
family-1 alpha contract.

It lets us prove:

- a supported PyTorch-origin model can be exported into an InferGo-native
  bundle
- the resulting bundle loads through the alpha loader
- parity can be checked reproducibly against the source model

without pretending that InferGo already contains a full native encoder runtime.

## Implemented scope

The first exporter implementation supports:

- PyTorch-origin models loaded through Hugging Face `transformers`
- encoder-style **single-text classification**
- encoder-style **paired-text classification**
- alpha-format bundle output for the `bionet` backend
- tokenized-input parity and loading in Go
- tokenizer-backed raw-text serving for supported exported bundles
- paired-text HTTP requests through `{"text":"...","text_pair":"..."}`
- tokenizer asset staging inside the exported bundle

It does **not yet** implement:

- direct transformer-weight execution in the native runtime
- tokenizer-backed serving for arbitrary tokenizer families beyond the current
  BERT-style WordPiece tokenizer-json subset

Pair-scoring is supported in this milestone only when the source model can be
represented as a paired-text sequence-classification head with ordered label
logits.

## Exporter entrypoint

The first exporter entrypoint is:

- [`cmd/infergo-export`](../cmd/infergo-export)
- end-to-end usage is documented in
  [`docs/alpha-family-1-walkthrough.md`](./alpha-family-1-walkthrough.md)

It is intentionally an installable CLI instead of a repo-only script so users
can export without cloning this repository. The actual source-model work is
still Python-first because the source ecosystem is still PyTorch/Transformers.

## Input contract

The exporter currently expects:

1. a supported source model id or local model directory
2. a public-safe input set in the existing text-classification input format
3. an output bundle directory

Current input-set shape:

```json
{
  "name": "example input set",
  "cases": [
    {
      "id": "case-1",
      "text": "This product is excellent and reliable."
    },
    {
      "id": "pair-1",
      "text": "The customer asked for a refund after being charged twice.",
      "text_pair": "A customer requested a refund because they were billed two times."
    }
  ]
}
```

## CLI shape

Install the exporter:

```bash
go install github.com/pergamon-labs/infergo/cmd/infergo-export@latest
```

Write a starter input template:

```bash
infergo-export template -kind single -out ./family1-inputs.json
```

Single-text export:

```bash
infergo-export export \
  -model distilbert/distilbert-base-uncased-finetuned-sst-2-english \
  -input ./family1-inputs.json \
  -out ./artifacts/distilbert-sst2-alpha
```

Paired-text usage:

```bash
infergo-export template -kind pair -out ./family1-pairs.json

infergo-export export \
  -model textattack/bert-base-uncased-MRPC \
  -input ./family1-pairs.json \
  -out ./artifacts/mrpc-alpha
```

Important flags:

- `template`
  - writes a public-safe starter JSON file for either `single` or `pair`
    inputs
- `export`
  - exports a supported model into an InferGo-native bundle without a repo
    checkout
- `-model`
  - Hugging Face model id or local model directory
- `-model-id`
  - optional canonical model id to record in the exported bundle metadata
- `-input`
  - public-safe calibration/parity input set
  - supports optional `text_pair` per case
- `-out`
  - output bundle directory
- `-feature-mode`
  - current BIOnet native bundle mode
- `-max-length`
  - max tokenizer length passed to the source reference generator
- `-bundle-version`
  - alpha bundle version written into `metadata.json`
- `-python-runner`
  - `uv` by default, or `python` if the caller wants to manage Python deps
    themselves

## Exporter workflow

The current exporter does the following:

1. optionally write a starter input template for the user
2. load the source model and source tokenizer through `transformers`
3. generate a source reference JSON over the supplied input set
4. save tokenizer assets into `tokenizer/`
5. fit the BIOnet native bundle against the generated reference
6. write `labels.json`
7. write alpha-format `metadata.json`
8. print the supported input modes for the exported bundle

Internally, it uses:

- an embedded Python helper for `transformers` reference generation and
  tokenizer asset export
- [`internal/nativebundlegen`](../internal/nativebundlegen) for the BIOnet
  bundle fit

## Output contract

The exporter writes a bundle that follows:

- [`docs/alpha-native-artifact-contract.md`](./alpha-native-artifact-contract.md)

Current output shape:

```text
my-bundle/
  metadata.json
  labels.json
  model.gob
  embeddings.gob
  tokenizer/
    manifest.json
    tokenizer.json
    tokenizer_config.json
    special_tokens_map.json
    vocab.txt
    merges.txt
```

Not every tokenizer file will exist for every model family. The exporter writes
only the files produced by `save_pretrained(...)` and references those exact
files from `tokenizer/manifest.json`.

## Metadata posture for the first exporter

The first implementation keeps one boundary explicit:

- exported bundles remain **tokenized-input capable**
- exported bundles can now also support tokenizer-backed raw text when the
  staged tokenizer assets match the current supported runtime subset
- paired-text export is supported and exported bundles can now accept paired
  HTTP requests through `text` + `text_pair`
- the first runtime-backed raw-text path is intentionally narrow:
  BERT-style `hf-tokenizer-json` assets with a WordPiece model and Template
  Processing layout

So the first exporter currently writes:

- `inputs.tokenized_input_supported = true`
- `inputs.raw_text_supported = true` when the staged tokenizer assets match the
  supported runtime subset
- `inputs.pair_text_supported = true` when the input set and tokenizer assets
  both support paired text
- `tokenizer.manifest` only when the staged tokenizer assets stay inside the
  supported runtime subset; otherwise the bundle remains tokenized-input-only

This is intentional. It keeps the first milestone honest while enabling a real
raw-text serving path only for the tokenizer subset we actually support today.

## Loader and parity expectations

An exported bundle is considered valid for this milestone when:

1. `infer.LoadTextClassifier(...)` can load it
2. `cmd/infergo-parity -infergo-bundle-dir ...` passes against the generated
   source reference within the documented tolerance
3. `cmd/infergo-serve -task text -bundle ...` can serve the bundle over HTTP
   using raw text when `inputs.raw_text_supported=true`
4. paired-text bundles accept `{"text":"...","text_pair":"..."}` requests when
   `inputs.pair_text_supported=true`
5. a user can reach those steps through `infergo-export` and
   `alpha-family-1-walkthrough.md` without needing repo-local fixture paths

This is the first concrete family-1 success bar.

## Failure conditions

The exporter should fail clearly when:

- the source model cannot be loaded through `transformers`
- the source task is not compatible with sequence classification
- the input set is empty or malformed
- the native bundle generator fails
- tokenizer assets cannot be saved
- the final bundle metadata would violate the alpha contract

## First real target model

The first real targets validated with this exporter are:

- `distilbert/distilbert-base-uncased-finetuned-sst-2-english`
- `textattack/bert-base-uncased-MRPC`

Why these first:

- stable and widely used sequence-classification models
- cover both single-text and paired-text branches
- good fit for the first projection-based exporter milestone

## What comes next

Once this exporter path is working and validated, the next family-1 steps
should be:

1. pair-scoring export for entity-resolution-style family-1 models with
   clearer output semantics
2. expand tokenizer-backed raw-text loading/serving beyond the current
   BERT-style tokenizer-json subset
3. tighten the exported-bundle library path so raw-text serving is not only an
   HTTP concern
4. move from a script-first exporter toward a more polished public CLI
