# InferGo-Native Text Classification Bundles

This directory holds checked-in InferGo-native text-classification bundles used
by the Go-only parity path.

Current bundles:

- one or more checked-in native bundles exist per supported pack in
  `testdata/reference/text-classification/model-packs.json`
- the DistilBERT SST-2 pack currently keeps:
  - `distilbert-sst2-token-id-bag/`
  - `distilbert-sst2-embedding-avg-pool/`
  - `distilbert-sst2-embedding-masked-avg-pool/`
- the Twitter RoBERTa sentiment pack currently keeps:
  - `twitter-roberta-sentiment-embedding-masked-avg-pool/`

The native bundle generator also has an experimental `-use-layernorm` flag for
the masked-pooling path. That is useful for iteration, but it is not yet part
of the checked-in supported parity path.

This is intentionally narrower than general transformer execution. Its purpose
is to prove three things:

- InferGo can define and load a native bundle format.
- Go can produce the candidate side of the parity report without libtorch.
- The parity harness can stay stable while the native artifact format evolves.

Regenerate all supported text-classification packs from the repo root with:

```bash
uv run --with torch==2.10.0 --with transformers==5.3.0 \
  python ./scripts/build_text_classification_reference_pack.py
```

List the supported text-classification pack keys with:

```bash
uv run python ./scripts/build_text_classification_reference_pack.py --list
```

Regenerate one supported pack explicitly with:

```bash
uv run --with torch==2.10.0 --with transformers==5.3.0 \
  python ./scripts/build_text_classification_reference_pack.py \
  --pack-key distilbert-sst2
```

The lower-level generator is still available when you want to experiment
outside the supported pack workflow. For example, regenerate the earlier
`embedding-avg-pool` bundle explicitly with:

```bash
go run ./internal/tools/nativebundlegen \
  -mode embedding-avg-pool \
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
