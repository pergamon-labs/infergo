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
- implementation form: documented Python-first script
- current task branch: **single-text classification**

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
- alpha-format bundle output for the `bionet` backend
- tokenized-input parity and loading in Go
- tokenizer asset staging for future raw-text support

It does **not yet** implement:

- paired-text classification export
- pair-scoring export
- direct raw-text serving from exported bundles
- direct transformer-weight execution in the native runtime

Those remain part of the family-1 roadmap, but not this first exporter
milestone.

## Exporter entrypoint

The first exporter entrypoint is:

- [`scripts/export_encoder_text_bundle.py`](../scripts/export_encoder_text_bundle.py)

It is intentionally a Python-first script because the source ecosystem is still
PyTorch/Transformers.

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
    }
  ]
}
```

## CLI shape

Current usage:

```bash
uv run --with torch==2.10.0 --with transformers==5.3.0 \
  python ./scripts/export_encoder_text_bundle.py \
  --model distilbert/distilbert-base-uncased-finetuned-sst-2-english \
  --input ./testdata/reference/text-classification/sst2-inputs.json \
  --out ./dist/family1/distilbert-sst2-alpha
```

Important flags:

- `--model`
  - Hugging Face model id or local model directory
- `--model-id`
  - optional canonical model id to record in the exported bundle metadata
- `--input`
  - public-safe calibration/parity input set
- `--out`
  - output bundle directory
- `--feature-mode`
  - current BIOnet native bundle mode
- `--max-length`
  - max tokenizer length passed to the source reference generator
- `--bundle-version`
  - alpha bundle version written into `metadata.json`

## Exporter workflow

The current exporter does the following:

1. load the source model and source tokenizer through `transformers`
2. generate a source reference JSON over the supplied input set
3. run the existing native BIOnet bundle generator against that reference
4. save tokenizer assets into `tokenizer/`
5. write `labels.json`
6. write alpha-format `metadata.json`
7. materialize the final bundle directory

Internally, it builds on these existing repo tools:

- [`scripts/transformers_text_classification_reference.py`](../scripts/transformers_text_classification_reference.py)
- [`internal/tools/nativebundlegen`](../internal/tools/nativebundlegen)

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

- exported bundles are currently **tokenized-input capable**
- tokenizer assets are staged in the bundle
- raw-text serving through the generic public surfaces is **not claimed yet**

So the first exporter currently writes:

- `inputs.tokenized_input_supported = true`
- `inputs.raw_text_supported = false`
- `inputs.pair_text_supported = false`

This is intentional. It keeps the first milestone honest while preserving the
tokenizer assets we will need for later raw-text support.

## Loader and parity expectations

An exported bundle is considered valid for this milestone when:

1. `infer.LoadTextClassifier(...)` can load it
2. `cmd/infergo-parity -infergo-bundle-dir ...` passes against the generated
   source reference within the documented tolerance

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

The first target we should validate with this exporter is:

- `distilbert/distilbert-base-uncased-finetuned-sst-2-english`

Why first:

- stable and widely used
- already present in the repo's parity/reference flow
- simple single-text classification shape
- good fit for the first exporter milestone

## What comes next

Once this exporter path is working and validated, the next family-1 steps
should be:

1. paired-text classification input support
2. pair-scoring export for entity-resolution-style family-1 models
3. tokenizer-backed raw-text loading/serving for exported bundles
4. move from a script-first exporter toward a more polished public CLI
