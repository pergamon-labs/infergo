# infergo-entres-serve

`infergo-entres-serve` is the experimental serving command for InferGo's
family-2 internal dogfood bridge:

- numeric-feature TorchScript scoring
- aimed first at the current `entres` entity-resolution model shape

It is intentionally separate from [`cmd/infergo-serve`](../infergo-serve) so
the stable public serving surface remains focused on the primary family-1 alpha
contract.

## Status

- support level: experimental
- intended use: internal dogfooding
- backend requirement: `torchscript` backend with native libtorch support

## Bundle contract

The command expects a bundle directory with at least:

```text
my-entres-bridge/
  metadata.json
  model.torchscript.pt
```

See also:

- [`docs/alpha-family-2-entres-bridge.md`](../../docs/alpha-family-2-entres-bridge.md)

## Run

```bash
go run ./cmd/infergo-entres-serve \
  -bundle ./artifacts/entres-individual
```

Or with env-driven defaults:

```bash
INFERGO_ENTRES_SERVE_BUNDLE=./artifacts/entres-individual \
go run ./cmd/infergo-entres-serve
```

## Endpoints

- `GET /healthz`
- `GET /metadata`
- `POST /predict`

## Request contract

```json
{
  "vectors": [
    [0.0, 1.0, 0.5],
    [0.2, 0.3, 0.4]
  ],
  "message": [0.1, 0.2, 0.3]
}
```

## Response contract

```json
{
  "backend": "torchscript",
  "model_id": "pergamon/entres-individual",
  "scores": [0.42, 0.87]
}
```
