# InferGo Serve

`infergo-serve` is the optional standalone HTTP entrypoint for InferGo.

Most Go teams will embed InferGo directly in an existing service. Use
`infergo-serve` when you want:

- a quick smoke-test surface
- a separate model process
- a simple HTTP boundary for non-Go callers

Install it:

```bash
go install github.com/pergamon-labs/infergo/cmd/infergo-serve@latest
```

If you cloned the repo, the same commands can be run with `go run ./cmd/infergo-serve`.

## Serve an exported family-1 bundle

Single-text example:

```bash
infergo-serve -task text -bundle ./artifacts/distilbert-sst2-alpha -addr 127.0.0.1:8080
```

Paired-text example:

```bash
infergo-serve -task text -bundle ./artifacts/mrpc-alpha -addr 127.0.0.1:8080
```

Health and metadata:

```bash
curl -s http://127.0.0.1:8080/healthz
curl -s http://127.0.0.1:8080/metadata
```

Single-text request:

```bash
curl -s -X POST http://127.0.0.1:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"This product is excellent and reliable."}'
```

Paired-text request:

```bash
curl -s -X POST http://127.0.0.1:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"The company said the deal closed.","text_pair":"The acquisition has been completed, the company said."}'
```

Exported bundles may also accept explicit tokenized input:

```bash
curl -s -X POST http://127.0.0.1:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{"input_ids":[101,2023,4031,2003,6581,1998,10539,1012,102],"attention_mask":[1,1,1,1,1,1,1,1,1]}'
```

The bundle path and curated-pack path are mutually exclusive:

- use `-bundle` for exported family-1 text bundles
- use `-pack` for curated checked-in packs

## Serve curated packs

Curated text pack:

```bash
infergo-serve -task text
```

Curated token pack:

```bash
infergo-serve -task token
```

Token example request:

```bash
curl -s -X POST http://127.0.0.1:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"Sophie Tremblay a parlé avec Hydro-Québec à Montréal."}'
```

## Configuration

Useful flags:

```bash
infergo-serve \
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

Errors are returned as structured JSON:

```json
{
  "error": {
    "code": "invalid_request",
    "message": "provide exactly one supported input mode for this bundle"
  }
}
```

To benchmark the HTTP handler path:

```bash
go test ./infer/httpserver -run '^$' -bench . -benchmem
```
