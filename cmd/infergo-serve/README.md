# InferGo Serve

`infergo-serve` is the current first-class HTTP serving entrypoint for InferGo.

To benchmark the current handler path:

```bash
go test ./infer/httpserver -run '^$' -bench . -benchmem
```

Run a text-classification pack:

```bash
go run ./cmd/infergo-serve -task text
```

Run a token-classification pack:

```bash
go run ./cmd/infergo-serve -task token
```

Useful flags:

```bash
go run ./cmd/infergo-serve \
  -task token \
  -pack infergo-basic-french-ner \
  -addr :8081 \
  -log-requests=true \
  -read-timeout 5s \
  -write-timeout 10s \
  -idle-timeout 30s \
  -shutdown-timeout 10s
```

The same defaults can be set through environment variables:

- `INFERGO_SERVE_ADDR`
- `INFERGO_SERVE_TASK`
- `INFERGO_SERVE_PACK`
- `INFERGO_SERVE_BUNDLE`
- `INFERGO_SERVE_LOG_REQUESTS`
- `INFERGO_SERVE_READ_TIMEOUT`
- `INFERGO_SERVE_READ_HEADER_TIMEOUT`
- `INFERGO_SERVE_WRITE_TIMEOUT`
- `INFERGO_SERVE_IDLE_TIMEOUT`
- `INFERGO_SERVE_SHUTDOWN_TIMEOUT`

Check health and metadata:

```bash
curl -s http://127.0.0.1:8080/healthz
curl -s http://127.0.0.1:8080/metadata
```

Run prediction:

```bash
curl -s -X POST http://127.0.0.1:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"This product is excellent and reliable."}'
```

Serve an exported family-1 text bundle directly without using curated pack
manifests:

```bash
go run ./cmd/infergo-serve \
  -task text \
  -bundle ./dist/family1/mrpc-alpha \
  -addr 127.0.0.1:8080
```

Then call it with raw text when the exported bundle advertises
`supports_raw_text=true`:

```bash
curl -s -X POST http://127.0.0.1:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"This product is excellent and reliable."}'
```

Paired-text bundles can accept `text` plus `text_pair`:

```bash
curl -s -X POST http://127.0.0.1:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"The company said the deal closed.","text_pair":"The acquisition has been completed, the company said."}'
```

The bundle path and curated pack path are mutually exclusive:

- use `-pack` for curated checked-in packs
- use `-bundle` for directly exported family-1 text bundles

Exported bundles may still accept explicit tokenized input through:

```bash
curl -s -X POST http://127.0.0.1:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{"input_ids":[101,2023,4031,2003,6581,1998,10539,1012,102],"attention_mask":[1,1,1,1,1,1,1,1,1]}'
```

When `-task token` is used, the default pack is `infergo-basic-french-ner` and
the default raw-text example is:

```bash
curl -s -X POST http://127.0.0.1:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"Sophie Tremblay a parlé avec Hydro-Québec à Montréal."}'
```

Errors are returned as structured JSON:

```json
{
  "error": {
    "code": "invalid_request",
    "message": "provide exactly one supported input mode for this bundle"
  }
}
```
