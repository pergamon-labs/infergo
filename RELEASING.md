# Releasing InferGo

This repo is still in a narrow alpha stage. Releases should optimize for
honesty and reproducibility over breadth.

## Current release posture

InferGo should keep using explicit prerelease tags until the support boundary
widens meaningfully and repeated outside usage validates the workflow.

Current alpha line:

```text
v0.2.0-alpha.N
```

## Release checklist

Run these from the repo root:

1. Validate the normal test suite:

```bash
go test ./...
```

2. Capture a fresh local benchmark snapshot:

```bash
./scripts/benchmark_snapshot.sh -count 5 -out ./benchmarks/local/release-candidate.txt
```

3. Compare the snapshot with your previous local release snapshot:

```bash
./scripts/benchmark_compare.sh ./benchmarks/local/previous-release.txt ./benchmarks/local/release-candidate.txt
```

Do not commit those raw output files. Summarize only the notable local deltas
in the release notes when something materially changed.

4. Validate pack discovery:

```bash
go run ./cmd/infergo-packs
```

5. Validate at least one parity path:

```bash
go run ./cmd/infergo-parity \
  -reference ./testdata/reference/token-classification/infergo-basic-french-ner-reference.json \
  -infergo-bundle-dir ./testdata/native/token-classification/infergo-basic-french-ner-windowed-embedding-linear \
  -tolerance 1e-3
```

6. Confirm the public docs are aligned:

- `README.md`
- `COMPATIBILITY.md`
- `ARCHITECTURE.md`
- `BENCHMARKS.md`
- `CHANGELOG.md`
- bundle-version compatibility notes still match the enforced loader/exporter behavior

7. Confirm no proprietary or machine-specific artifacts were introduced.

8. Create the annotated tag:

```bash
git tag -a v0.2.0-alpha.N -m "InferGo v0.2.0-alpha.N"
```

9. Publish the GitHub release using:

- [`CHANGELOG.md`](./CHANGELOG.md)
- the matching file under [`docs/releases/`](./docs/releases/)

## Release guidance

The GitHub release title should stay explicit:

```text
InferGo v0.2.0-alpha.N
```

The release description should emphasize:

- Go-native inference for backend services
- family-1 BYOM export/import for the documented supported family
- library-first usage with optional HTTP serving
- benchmark deltas as illustrative local measurements, not guarantees

The release description should not imply:

- blanket Hugging Face model support
- generic `.pt` loading
- full transformer execution
- training support
