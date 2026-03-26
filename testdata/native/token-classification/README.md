# Native Token Classification Bundles

This directory contains checked-in InferGo-native bundles for narrow,
public-safe token-classification parity checks.

Current bundle:

- `distilbert-ner-embedding-linear`: compact token-embedding plus linear-head
  artifact derived from the public `dslim/distilbert-NER` reference set

Generation:

```bash
go run ./internal/tools/nativetokenbundlegen
```

Notes:

- this bundle is intentionally narrow and reference-driven
- it is a parity spike for sequence labeling, not a claim of full contextual NER
  support
