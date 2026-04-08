# Exported Bundle Classifier

This example shows the non-HTTP Go library path for an exported family-1 text
bundle.

For most Go teams, this is the normal InferGo usage mode: load a bundle in your
existing service and call it directly.

Single-text example:

```bash
go run ./examples/exported-bundle-classifier \
  -bundle ./artifacts/distilbert-sst2-alpha \
  -text "This product is excellent and reliable."
```

Paired-text example:

```bash
go run ./examples/exported-bundle-classifier \
  -bundle ./artifacts/mrpc-alpha \
  -text "The company said the deal closed." \
  -text-pair "The acquisition has been completed, the company said."
```

This example relies on:

- `infer.LoadTextBundle(...)`
- tokenizer-backed raw-text support from the exported bundle metadata
- the current family-1 alpha export path documented in
  [`docs/alpha-family-1-walkthrough.md`](../../docs/alpha-family-1-walkthrough.md)
