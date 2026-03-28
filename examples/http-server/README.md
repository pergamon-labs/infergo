# HTTP Server Example

This example demonstrates the core InferGo shape today without overclaiming raw
text tokenization support.

It will:

- load a checked-in InferGo-native text-classification bundle
- expose a tiny `/predict` HTTP endpoint
- accept either a known `case_id` from a checked-in reference file or explicit
  `input_ids` plus `attention_mask`

Run it from the repo root:

```bash
go run ./examples/http-server
```

Then call it with a checked-in case:

```bash
curl -s -X POST http://127.0.0.1:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{"case_id":"positive-review"}'
```
