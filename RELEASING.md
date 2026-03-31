# Releasing InferGo

This repo is still in a narrow pre-alpha stage. Releases should optimize for
honesty and reproducibility over breadth.

## First public tag

Recommended first public tag:

```text
v0.1.0-prealpha.1
```

Why this tag:

- it signals that InferGo is public but still early
- it avoids implying broad compatibility
- it leaves room for more pre-alpha and alpha milestones before a stable `v1`

## Release checklist

Run these from the repo root:

1. Validate the normal test suite:

```bash
go test ./...
```

2. Validate the benchmark suite:

```bash
go test ./infer/packs -run '^$' -bench . -benchmem
```

3. Validate pack discovery:

```bash
go run ./cmd/infergo-packs
```

4. Validate at least one parity path:

```bash
go run ./cmd/infergo-parity \
  -reference ./testdata/reference/token-classification/infergo-basic-french-ner-reference.json \
  -infergo-bundle-dir ./testdata/native/token-classification/infergo-basic-french-ner-windowed-embedding-linear \
  -tolerance 1e-3
```

5. Confirm the public docs are aligned:

- `README.md`
- `COMPATIBILITY.md`
- `ARCHITECTURE.md`
- `CHANGELOG.md`

6. Confirm no proprietary or machine-specific artifacts were introduced.

7. Create the annotated tag:

```bash
git tag -a v0.1.0-prealpha.1 -m "InferGo v0.1.0-prealpha.1"
```

8. Publish the GitHub release using:

- [`CHANGELOG.md`](./CHANGELOG.md)
- [`docs/releases/v0.1.0-prealpha.1.md`](./docs/releases/v0.1.0-prealpha.1.md)

## Release guidance

The GitHub release title should stay explicit:

```text
InferGo v0.1.0-prealpha.1
```

The release description should emphasize:

- Go-native inference for backend services
- CPU-first native bundles
- curated parity-backed pack support
- raw-text support is intentionally narrow

The release description should not imply:

- blanket Hugging Face model support
- generic `.pt` loading
- full transformer execution
- training support
