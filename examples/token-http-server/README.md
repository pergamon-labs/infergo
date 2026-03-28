# Token Classification HTTP Server Example

This example demonstrates the stable public `infer` API for serving a checked-in
native token-classification bundle behind a tiny HTTP endpoint.

It will:

- load a checked-in InferGo-native token-classification bundle through `infer`
- expose a `/predict` HTTP endpoint
- accept either a known `case_id` from a checked-in reference file or explicit
  `input_ids` plus `attention_mask`

Run it from the repo root:

```bash
go run ./examples/token-http-server
```

Then call it with a checked-in French demo case:

```bash
curl -s -X POST http://127.0.0.1:8081/predict \
  -H 'Content-Type: application/json' \
  -d '{"case_id":"frca-003"}'
```
