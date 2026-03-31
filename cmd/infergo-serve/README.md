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
    "message": "provide exactly one of text, tokens, or case_id"
  }
}
```
