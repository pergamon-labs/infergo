# Benchmarks

InferGo ships a small benchmark suite for the currently honest native developer
paths:

- raw-text text classification through `infergo-basic-sst2`
- raw-text token classification through `infergo-basic-french-ner`
- HTTP metadata and prediction paths through `infer/httpserver`

These benchmarks are meant to answer the first backend-team questions:

- how expensive is bundle load at process start
- what is steady-state CPU latency for prediction
- what do allocations look like with `-benchmem`

InferGo does not check machine-specific benchmark numbers into the repo. Run the
suite on your own hardware and compare deltas over time.

## Run the benchmark suite

From the repo root:

```bash
go test ./infer/packs ./infer/httpserver -run '^$' -bench . -benchmem
```

If you want to focus on startup cost only:

```bash
go test ./infer/packs -run '^$' -bench '^BenchmarkLoad' -benchmem
```

If you want to focus on steady-state prediction only:

```bash
go test ./infer/packs ./infer/httpserver -run '^$' -bench '^Benchmark(Predict|Metadata)' -benchmem
```

If you want a more stable local read, run several passes:

```bash
go test ./infer/packs ./infer/httpserver -run '^$' -bench . -benchmem -count=5
```

## What is covered today

- `BenchmarkLoadTextPackInfergoBasicSST2`
- `BenchmarkPredictTextInfergoBasicSST2`
- `BenchmarkLoadTokenPackInfergoBasicFrenchNER`
- `BenchmarkPredictTextInfergoBasicFrenchNER`
- `BenchmarkPredictTokensInfergoBasicFrenchNER`
- `BenchmarkMetadataTextPackInfergoBasicSST2`
- `BenchmarkPredictTextInfergoBasicSST2HTTP`
- `BenchmarkPredictTokenTextInfergoBasicFrenchNERHTTP`
- `BenchmarkPredictTokenTokensInfergoBasicFrenchNERHTTP`

These are intentionally narrow. They measure the public `infer/packs` surface
and the stable `infer/httpserver` REST surface that outside developers are most
likely to use first, not every internal runtime primitive.

## How to read the results

- `ns/op`: rough latency per operation
- `B/op`: bytes allocated per operation
- `allocs/op`: allocation count per operation

The load benchmarks are the closest current proxy for startup cost.

The predict benchmarks represent hot-path native inference after the pack is
already loaded.

The HTTP benchmarks add JSON decode/encode and route handling on top of that
same curated pack surface.
