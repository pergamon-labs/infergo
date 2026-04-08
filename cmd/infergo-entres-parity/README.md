# infergo-entres-parity

`infergo-entres-parity` compares a local family-2 entity-resolution fixture
captured from the current Minerva runtime against an InferGo family-2 bridge
bundle.

This command is intentionally experimental and internal-first. It exists to
validate the `entres` dogfood bridge without broadening InferGo's public alpha
family claims.

## Usage

```bash
source ./scripts/setup_libtorch_local.sh

go run -tags torchscript_native ./cmd/infergo-entres-parity \
  -fixture ./dist/entres/parity/individual-fixture.json \
  -bundle ./dist/entres/individual \
  -tolerance 1e-6
```

The fixture file is expected to be generated locally from the current internal
runtime. It is not a public checked-in asset.

The current local parity flow has been validated against both:

- `pergamon/entres-individual`
- `pergamon/entres-organization`

See also:

- [`docs/alpha-family-2-entres-bridge.md`](../../docs/alpha-family-2-entres-bridge.md)
- [`cmd/infergo-entres-bundle/README.md`](../infergo-entres-bundle/README.md)
- [`cmd/infergo-entres-serve/README.md`](../infergo-entres-serve/README.md)
