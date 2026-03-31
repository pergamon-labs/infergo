# InferGo Serve

`infergo-serve` is the current first-class HTTP serving entrypoint for InferGo.

Run a text-classification pack:

```bash
go run ./cmd/infergo-serve -task text
```

Run a token-classification pack:

```bash
go run ./cmd/infergo-serve -task token
```

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
