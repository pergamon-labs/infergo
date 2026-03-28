# Native Token Classification Bundles

This directory contains checked-in InferGo-native bundles for narrow,
public-safe token-classification parity checks.

Current bundle:

- one checked-in `*-windowed-embedding-linear` native bundle per supported
  model pack in `testdata/reference/token-classification/model-packs.json`
- `distilbert-ner-embedding-linear`: earlier token-only baseline kept as a
  simple comparison point
- the multilingual `xlm-roberta-ner-hrl-windowed-embedding-linear/` bundle is
  the first checked-in non-English token-classification native path

Generation:

```bash
uv run --with torch==2.10.0 --with transformers==5.3.0 \
  --with sentencepiece --with protobuf --with tiktoken \
  python ./scripts/build_token_classification_reference_pack.py
```

Notes:

- this bundle is intentionally narrow and reference-driven
- it is a parity spike for sequence labeling, not a claim of full contextual NER
  support
- the windowed bundle is the current supported native token-classification path
