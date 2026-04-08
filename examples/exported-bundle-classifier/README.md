# Exported Bundle Classifier

This example shows the non-HTTP Go library path for an exported family-1 text
bundle.

Single-text example:

```bash
go run ./examples/exported-bundle-classifier \
  -bundle ./dist/family1/distilbert-sst2-alpha \
  -text "This product is excellent and reliable."
```

Paired-text example:

```bash
go run ./examples/exported-bundle-classifier \
  -bundle ./dist/family1/mrpc-alpha \
  -text "The company said the deal closed." \
  -text-pair "The acquisition has been completed, the company said."
```

This example relies on:

- `infer.LoadTextBundle(...)`
- tokenizer-backed raw-text support from the exported bundle metadata
- the current family-1 alpha export path documented in
  [`docs/alpha-family-1-walkthrough.md`](../../docs/alpha-family-1-walkthrough.md)
