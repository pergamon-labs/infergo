# Parity Check Example

This example demonstrates the first public-safe parity workflow:

```bash
go run ./cmd/infergo-parity -fixture ./testdata/parity/text-classification/fixture.json
```

The current fixture uses a synthetic BIOnet-backed text-classification artifact so we can validate the parity harness before moving to a Hugging Face / Transformers-backed reference path.
