# infergo-entres-bundle

`infergo-entres-bundle` scaffolds the experimental family-2 bridge bundle
around an existing TorchScript entity-resolution artifact.

It is intended for the internal `entres` dogfood path, not the main public
family-1 alpha exporter workflow.

## Run

```bash
go run ./cmd/infergo-entres-bundle \
  -model /path/to/ind_conv1d_compiled.pt \
  -profile-kind individual \
  -model-id pergamon/entres-individual \
  -out ./dist/entres/individual
```

By default the command creates a symlinked `model.torchscript.pt` inside the
bundle directory. Use `-copy-artifact` if you need a physical copy instead.

The generated metadata records:

- family/task/backend identifiers
- profile kind
- vector/message sizes
- input layout
- message strategy
- score interpretation
- source framework/format
