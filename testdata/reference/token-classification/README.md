# Token Classification References

This directory contains public-safe token-classification reference assets used
by InferGo parity checks.

Current reference set:

- `ner-inputs.json`: a widened, reproducible NER parity set with repeated
  tokens, multi-token entities, punctuation-heavy examples, subword splits,
  acronyms, quoted entities, slash-separated organizations, and title-heavy
  cases
- `multilingual-ner-inputs.json`: the first public-safe non-English NER corpus
  covering Spanish, French, German, Italian, and Portuguese examples, with
  diacritics, apostrophes, quoted entities, slash-separated organizations, and
  mixed-language cases
- `french-ner-inputs.json`: a French-specific public-safe NER corpus with
  France and Canada-relevant examples, including Montréal, Québec,
  Hydro-Québec, Radio-Canada, and Université de Montréal
- `model-packs.json`: the supported public token-classification model pack
  manifest
- one generated `*-reference.json` file per supported pack in the manifest

Generation:

```bash
uv run --with torch==2.10.0 --with transformers==5.3.0 \
  --with sentencepiece --with protobuf --with tiktoken \
  python ./scripts/build_token_classification_reference_pack.py
```

That manifest now includes a first non-English token-classification pack:

- `xlm-roberta-ner-hrl`
- `distilcamembert-french-ner`

Scoring rules:

- InferGo compares only tokens where `scoring_mask` is `1`
- special tokens and punctuation are intentionally excluded from scoring
- token-classification pass/fail is based on label agreement plus probability
  agreement within tolerance; the checked-in native token path is currently
  exercised at `1e-3`, and raw logit deltas are still reported as diagnostics
- the goal is to validate a narrow, public-safe native token-labeling path, not
  to claim general transformer-equivalent NER support
