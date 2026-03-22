# Synthetic Text Classification Fixture

This fixture is intentionally public-safe.

- The texts are generic and non-proprietary.
- The feature vectors are synthetic.
- The BIOnet model artifact is generated from a tiny deterministic linear + softmax classifier.

Regenerate the fixture files from the repo root with:

```bash
go run ./internal/tools/textfixturegen
```
