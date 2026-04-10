# Contributing

This scaffold is intentionally small and extraction-oriented.

InferGo is currently in alpha. Please keep contributions aligned with the
narrow, explicitly documented support surface.

Current contribution priorities:

1. keep the public API small and honest
2. avoid broad compatibility claims without parity coverage
3. prefer public-safe fixtures and examples
4. keep backend-specific code behind backend boundaries
5. open an issue before broadening support surface beyond the documented alpha
   contract

## Community norms

By participating in this project, you agree to follow the guidance in
[CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md).

## Pull requests

Before opening a pull request:

- run `go test ./...`
- update `README.md` if the public workflow changed
- update `COMPATIBILITY.md` if support claims changed
- keep examples, bundles, and fixtures public-safe and reproducible

If your change is expected to affect load cost, inference latency, HTTP
overhead, or allocations in the public `infer/packs` or `infer/httpserver`
paths:

- capture a local before/after snapshot with `./scripts/benchmark_snapshot.sh`
- compare the runs with `./scripts/benchmark_compare.sh`
- summarize only the meaningful local delta in the PR or issue discussion
- do not commit raw machine-specific benchmark output

For the current benchmark workflow, see [`BENCHMARKS.md`](./BENCHMARKS.md).

If you want to propose support for a new model family, tokenizer, runtime, or
format bridge, open an issue first so the support boundary can be discussed
before implementation.

## Before adding support for a backend

- document the artifact/export path
- define the supported task/model shapes
- add a public example
- add parity or golden tests
- document known limitations

## Before promoting an internal runtime detail to public API

- verify that it is useful outside a single backend
- verify naming and docs are stable enough to support
- avoid exposing low-level packages just because they exist internally today
