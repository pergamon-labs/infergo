# InferGo

InferGo is a Go-native inference toolkit for backend services.

It is built for teams that want to run small inference workloads directly in Go
without turning Python, PyTorch, or TensorFlow into a required production
runtime.

Current prerelease: [`v0.1.0-prealpha.1`](https://github.com/pergamon-labs/infergo/releases/tag/v0.1.0-prealpha.1)

## Why InferGo

- Go-native inference for backend services
- CPU-first native runtime path with narrow, explicit support claims
- Stable public Go APIs in [`infer/`](./infer) and curated pack helpers in
  [`infer/packs/`](./infer/packs)
- A first-class REST surface through [`infer/httpserver/`](./infer/httpserver)
  and [`cmd/infergo-serve/`](./cmd/infergo-serve)
- Parity-driven validation against public reference implementations
- Honest non-goals: not a training framework, not a blanket `.pt` loader, and
  not a general transformer runtime

## Installation

InferGo currently targets Go `1.26.1` or newer.

Add the library to your project:

```bash
go get github.com/pergamon-labs/infergo
```

If you want to run the checked-in examples and repo tools, clone the repo:

```bash
git clone https://github.com/pergamon-labs/infergo.git
cd infergo
go test ./...
```

## Quickstart

The fastest path from clone to a real result is:

1. List the curated packs that ship with the repository:

```bash
go run ./cmd/infergo-packs
```

2. Run the first raw-text text-classification example:

```bash
go run ./examples/bionet-classifier \
  -text "This product is excellent and reliable."
```

3. Run the first raw-text token-classification HTTP example:

```bash
go run ./cmd/infergo-serve -task token
```

Then call it:

```bash
curl -s -X POST http://127.0.0.1:8081/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"Sophie Tremblay a parlé avec Hydro-Québec à Montréal."}'
```

4. Run a parity check against a checked-in public reference:

```bash
go run ./cmd/infergo-parity \
  -reference ./testdata/reference/token-classification/distilbert-ner-reference.json \
  -infergo-bundle-dir ./testdata/native/token-classification/distilbert-ner-windowed-embedding-linear \
  -tolerance 1e-3
```

## Use as a library

InferGo's stable public package surface starts in [`infer/`](./infer) and
[`infer/packs/`](./infer/packs). The stable REST surface starts in
[`infer/httpserver/`](./infer/httpserver).

For the easiest curated path, load a checked-in pack:

```go
package main

import (
	"fmt"
	"log"

	"github.com/pergamon-labs/infergo/infer/packs"
)

func main() {
	pack, err := packs.LoadTextPack("infergo-basic-sst2")
	if err != nil {
		log.Fatal(err)
	}
	defer pack.Close()

	result, err := pack.PredictText("This product is excellent and reliable.")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("label=%s logits=%v\n", result.Label, result.Logits)
}
```

See also:

- [`examples/bionet-classifier/`](./examples/bionet-classifier)
- [`infer/`](./infer)
- [`infer/packs/`](./infer/packs)

## Serve over HTTP

InferGo now ships a first-class HTTP serving entrypoint:

Text classification:

```bash
go run ./cmd/infergo-serve -task text
```

Then call it:

```bash
curl -s -X POST http://127.0.0.1:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"This product is excellent and reliable."}'
```

Token classification:

```bash
go run ./cmd/infergo-serve -task token
```

Then call it:

```bash
curl -s -X POST http://127.0.0.1:8081/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"Sophie Tremblay a parlé avec Hydro-Québec à Montréal."}'
```

See also:

- [`cmd/infergo-serve/`](./cmd/infergo-serve)
- [`infer/httpserver/`](./infer/httpserver)
- [`examples/http-server/`](./examples/http-server)
- [`examples/token-http-server/`](./examples/token-http-server)

## Supported today

InferGo is intentionally narrow in `v0.1.0-prealpha.1`.

| Area | Status |
| --- | --- |
| Native text-classification bundles | Supported |
| Native token-classification bundles | Supported |
| Raw-text text prediction | Supported for curated validated packs only |
| Raw-text token prediction | Supported for curated validated packs only |
| Pack discovery CLI | Supported via [`cmd/infergo-packs/`](./cmd/infergo-packs) |
| Parity CLI | Supported via [`cmd/infergo-parity/`](./cmd/infergo-parity) |
| REST serving CLI | Supported via [`cmd/infergo-serve/`](./cmd/infergo-serve) |
| Stable HTTP handler package | Supported via [`infer/httpserver/`](./infer/httpserver) |
| Optional TorchScript bridge | Experimental / backend-specific |
| gRPC serving surface | Not yet |

For the canonical support contract, see
[`COMPATIBILITY.md`](./COMPATIBILITY.md).

## Benchmarks

Example snapshot from one local run on `darwin/arm64` with an Apple M3 Max:

| Benchmark | Result |
| --- | --- |
| `LoadTextPack(infergo-basic-sst2)` | about `0.36 ms/op`, `166925 B/op`, `2265 allocs/op` |
| `PredictText(infergo-basic-sst2)` | about `1.6 µs/op`, `1616 B/op`, `57 allocs/op` |
| `LoadTokenPack(infergo-basic-french-ner)` | about `0.61 ms/op`, `260463 B/op`, `2927 allocs/op` |
| `PredictText(infergo-basic-french-ner)` | about `7.5 µs/op`, `11464 B/op`, `229 allocs/op` |
| `PredictTokens(infergo-basic-french-ner)` | about `5.7 µs/op`, `9656 B/op`, `191 allocs/op` |
| `HTTP metadata (infergo-basic-sst2)` | about `4.1 µs/op`, `6386 B/op`, `20 allocs/op` |
| `HTTP predict text (infergo-basic-sst2)` | about `10.7 µs/op`, `9528 B/op`, `90 allocs/op` |
| `HTTP predict text (infergo-basic-french-ner)` | about `28.2 µs/op`, `21432 B/op`, `273 allocs/op` |

These numbers are only a point-in-time example. To reproduce them on your own
hardware, see [`BENCHMARKS.md`](./BENCHMARKS.md).

## Project status

InferGo is public, but still pre-alpha.

Current posture:

- CPU-first native inference in Go
- parity-backed support for curated public-safe packs
- BIOnet as the first real native backend path
- optional TorchScript bridge kept intentionally narrow

Non-goals for this release line:

- training
- blanket `.pt` compatibility
- general transformer execution
- GPU-first deployment story
- turning the checked-in packs into a model zoo

## Docs and project files

- [`COMPATIBILITY.md`](./COMPATIBILITY.md)
- [`ARCHITECTURE.md`](./ARCHITECTURE.md)
- [`BENCHMARKS.md`](./BENCHMARKS.md)
- [`CHANGELOG.md`](./CHANGELOG.md)
- [`RELEASING.md`](./RELEASING.md)
- [`CONTRIBUTING.md`](./CONTRIBUTING.md)
- [`cmd/infergo-serve/README.md`](./cmd/infergo-serve/README.md)
- [`docs/releases/v0.1.0-prealpha.1.md`](./docs/releases/v0.1.0-prealpha.1.md)
