# NER Service Example

This example demonstrates a more application-shaped use of InferGo token
classification: load a checked-in NER pack, accept raw text or token input, and
return extracted entities instead of only token labels.

It intentionally shows what a backend service can build today on top of:

- `infer/packs.LoadTokenPack(...)`
- `TokenPack.TokenizeText(...)`
- `TokenPack.TokenizeTextWithOffsets(...)`
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
- token spans with stable byte and character offsets for raw-text requests
- token labels
- grouped entities with token ranges and stable byte/character offsets for
  raw-text requests

Important limitation:

- the entity grouping logic intentionally lives in this sample service, not in a
  first-class stable helper under `infer/`

That limitation is tracked in the alpha gap docs.
