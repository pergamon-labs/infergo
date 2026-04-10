# NER Service Example

This example demonstrates a more application-shaped use of InferGo token
classification: load a checked-in NER pack, accept raw text or token input, and
return extracted entities instead of only token labels.

It intentionally shows what a backend service can build today on top of:

- `infer/packs.LoadTokenPack(...)`
- `TokenPack.TokenizeText(...)`
- `TokenPack.PredictTokens(...)`

Run it from the repo root:

```bash
go run ./examples/ner-service
```

Then call it with raw text:

```bash
curl -s -X POST http://127.0.0.1:8082/extract \
  -H 'Content-Type: application/json' \
  -d '{"text":"Sophie Tremblay a parlé avec Hydro-Québec à Montréal."}'
```

Or call it with token pieces directly:

```bash
curl -s -X POST http://127.0.0.1:8082/extract \
  -H 'Content-Type: application/json' \
  -d '{"tokens":["sophie","tremblay","a","parlé","avec","hydro","-","québec","à","montréal"]}'
```

The response includes:

- model metadata
- scored tokens
- token labels
- grouped entities with token ranges

Important limitation:

- this sample groups entity spans by token index only
- it does not yet expose first-class character offsets from the stable InferGo
  API

That limitation is tracked in the alpha gap docs.
