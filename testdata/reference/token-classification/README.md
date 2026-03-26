# Token Classification References

This directory contains public-safe token-classification reference assets used
by InferGo parity checks.

Current reference set:

- `ner-inputs.json`: a small, reproducible NER smoke set
- `distilbert-ner-reference.json`: generated from `dslim/distilbert-NER`

Generation:

```bash
uv run --with torch==2.10.0 --with transformers==5.3.0 \
  python ./scripts/transformers_token_classification_reference.py
```

Scoring rules:

- InferGo compares only tokens where `scoring_mask` is `1`
- special tokens and punctuation are intentionally excluded from scoring
- the goal is to validate a narrow, public-safe native token-labeling path, not
  to claim general transformer-equivalent NER support
