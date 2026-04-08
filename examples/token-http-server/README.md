# Token Classification HTTP Server Example

This example demonstrates the stable `infer/httpserver` package for serving a
checked-in native token-classification pack.

If you want an entity-oriented sample service instead of raw token-label JSON,
see [`examples/ner-service/`](../ner-service).

For the supported CLI entrypoint, prefer:

```bash
go run ./cmd/infergo-serve -task token
```

It will:

- load a checked-in InferGo-native token-classification pack through `infer/packs`
- expose `/healthz`, `/metadata`, and `/predict`
- accept raw text for packs that support native tokenization
- accept either a known `case_id` from a checked-in reference file or explicit
  tokens that match the chosen pack

Run it from the repo root:

```bash
go run ./examples/token-http-server
```

Then call it with raw text:

```bash
curl -s -X POST http://127.0.0.1:8081/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"Sophie Tremblay a parlé avec Hydro-Québec à Montréal."}'
```

Or call it with a checked-in French demo case:

```bash
curl -s -X POST http://127.0.0.1:8081/predict \
  -H 'Content-Type: application/json' \
  -d '{"case_id":"frca-003"}'
```

Or call it with token pieces directly:

```bash
curl -s -X POST http://127.0.0.1:8081/predict \
  -H 'Content-Type: application/json' \
  -d '{"tokens":["jean","dupont","a","rencontré","airbus","à","paris"]}'
```
