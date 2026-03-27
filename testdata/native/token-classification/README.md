# Native Token Classification Bundles

This directory contains checked-in InferGo-native bundles for narrow,
public-safe token-classification parity checks.

Current bundle:

- `distilbert-ner-windowed-embedding-linear`: compact prev/current/next token
  embedding plus linear-head artifact derived from the public
  `dslim/distilbert-NER` reference set
- `bert-base-ner-windowed-embedding-linear`: the same native bundle contract
  validated against `dslim/bert-base-NER`
- `distilbert-ner-embedding-linear`: earlier token-only baseline kept as a
  simple comparison point

Generation:

```bash
go run ./internal/tools/nativetokenbundlegen
```

Notes:

- this bundle is intentionally narrow and reference-driven
- it is a parity spike for sequence labeling, not a claim of full contextual NER
  support
- the windowed bundle is the current supported native token-classification path
