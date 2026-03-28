# Token Classification References

This directory contains public-safe token-classification reference assets used
by InferGo parity checks.

Current reference set:

- `ner-inputs.json`: a widened, reproducible NER parity set with repeated
  tokens, multi-token entities, punctuation-heavy examples, subword splits,
  acronyms, quoted entities, slash-separated organizations, and title-heavy
  cases
- `model-packs.json`: the supported public token-classification model pack
  manifest
- one generated `*-reference.json` file per supported pack in the manifest

Generation:

```bash
uv run --with torch==2.10.0 --with transformers==5.3.0 \
  python ./scripts/build_token_classification_reference_pack.py
```

Scoring rules:

- InferGo compares only tokens where `scoring_mask` is `1`
- special tokens and punctuation are intentionally excluded from scoring
- the goal is to validate a narrow, public-safe native token-labeling path, not
  to claim general transformer-equivalent NER support
