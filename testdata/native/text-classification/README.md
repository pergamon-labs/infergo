# InferGo-Native Text Classification Bundles

This directory holds checked-in InferGo-native text-classification bundles used
by the Go-only parity path.

Current bundles:

- `distilbert-sst2-token-id-bag/` is the narrow baseline that projects active
  token ids into a fixed bag-of-token feature vector, then runs a BIOnet linear
  classifier in pure Go.
- `distilbert-sst2-embedding-avg-pool/` is the next native step. It maps
  active token ids into a compact embedding table, average-pools across the
  sequence, and then runs a BIOnet linear head.

This is intentionally narrower than general transformer execution. Its purpose
is to prove three things:

- InferGo can define and load a native bundle format.
- Go can produce the candidate side of the parity report without libtorch.
- The parity harness can stay stable while the native artifact format evolves.

Regenerate the default `embedding-avg-pool` bundle from the repo root with:

```bash
go run ./internal/tools/nativebundlegen \
  -reference ./testdata/reference/text-classification/distilbert-sst2-reference.json \
  -output-dir ./testdata/native/text-classification/distilbert-sst2-embedding-avg-pool
```

Regenerate the `token-id-bag` baseline with:

```bash
go run ./internal/tools/nativebundlegen \
  -mode token-id-bag \
  -reference ./testdata/reference/text-classification/distilbert-sst2-reference.json \
  -output-dir ./testdata/native/text-classification/distilbert-sst2-token-id-bag
```
