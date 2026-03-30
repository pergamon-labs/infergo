# HTTP Server Example

This example demonstrates the curated `infer/packs` API for serving a checked-in
native text-classification pack without overclaiming raw-text tokenization
support.

It will:

- load a checked-in InferGo-native text-classification pack
- expose a tiny `/predict` HTTP endpoint
- accept either a known `case_id`, a token-piece array, or raw text when the
  chosen pack honestly supports a checked-in tokenizer helper

Run it from the repo root:

```bash
go run ./examples/http-server
```

Then call it with token pieces:

```bash
curl -s -X POST http://127.0.0.1:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{"tokens":["this","product","is","excellent","and","reliable","."]}'
```

Or call it with a checked-in case:

```bash
curl -s -X POST http://127.0.0.1:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{"case_id":"positive-review"}'
```
