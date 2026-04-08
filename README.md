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

Install the main CLIs without cloning the repo:

```bash
go install github.com/pergamon-labs/infergo/cmd/infergo-export@latest
go install github.com/pergamon-labs/infergo/cmd/infergo-serve@latest
go install github.com/pergamon-labs/infergo/cmd/infergo-parity@latest
```

For BYOM export, `infergo-export` still needs `uv` plus Python-side
`transformers` dependencies at export time. That dependency does not carry into
runtime serving once the bundle is built.

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

For exported family-1 bundles, use the higher-level bundle helper:

```go
bundle, err := infer.LoadTextBundle("./dist/family1/distilbert-sst2-alpha")
if err != nil {
	log.Fatal(err)
}
defer bundle.Close()

result, err := bundle.PredictText("This product is excellent and reliable.")
if err != nil {
	log.Fatal(err)
}
```

See also:

- [`examples/exported-bundle-classifier/`](./examples/exported-bundle-classifier)
- [`docs/alpha-family-1-walkthrough.md`](./docs/alpha-family-1-walkthrough.md)

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

`infergo-serve` now supports:

- structured JSON errors
- graceful shutdown on `SIGINT` and `SIGTERM`
- configurable read, write, idle, and shutdown timeouts
- env-driven defaults through `INFERGO_SERVE_*`

Then call it:

```bash
curl -s -X POST http://127.0.0.1:8081/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"Sophie Tremblay a parlé avec Hydro-Québec à Montréal."}'
```

Exported family-1 text bundle over the non-curated path:

```bash
go run ./cmd/infergo-serve -task text -bundle ./dist/family1/mrpc-alpha
```

Single-text exported bundle request:

```bash
curl -s -X POST http://127.0.0.1:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"This product is excellent and reliable."}'
```

Paired-text exported bundle request:

```bash
curl -s -X POST http://127.0.0.1:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"The company said the deal closed.","text_pair":"The acquisition has been completed, the company said."}'
```

See also:

- [`cmd/infergo-serve/`](./cmd/infergo-serve)
- [`infer/httpserver/`](./infer/httpserver)
- [`examples/http-server/`](./examples/http-server)
- [`examples/token-http-server/`](./examples/token-http-server)
- [`examples/ner-service/`](./examples/ner-service)

## Bring your own model (alpha)

The first supported BYOM path is family 1:

- PyTorch-origin
- Hugging Face Transformers-style
- encoder text classification / paired-text classification

Install the exporter:

```bash
go install github.com/pergamon-labs/infergo/cmd/infergo-export@latest
```

Write a starting input template:

```bash
infergo-export template -kind single -out ./family1-inputs.json
```

Then export a supported model:

```bash
infergo-export export \
  --model distilbert/distilbert-base-uncased-finetuned-sst-2-english \
  --input ./family1-inputs.json \
  --out ./artifacts/distilbert-sst2-alpha \
  --reference-output ./artifacts/distilbert-sst2-reference.json
```

Then either:

- load it from Go with `infer.LoadTextBundle(...)`
- serve it with `infergo-serve -task text -bundle ...`
- validate it with `infergo-parity ...`

For the full supported path, see
[`docs/alpha-family-1-walkthrough.md`](./docs/alpha-family-1-walkthrough.md).

## Supported today

InferGo is intentionally narrow in `v0.1.0-prealpha.1`.

| Area | Status |
| --- | --- |
| Native text-classification bundles | Supported |
| Native token-classification bundles | Supported |
| Raw-text text prediction | Supported for curated validated packs only |
| Raw-text token prediction | Supported for curated validated packs only |
| Sample NER extraction service | Supported via [`examples/ner-service/`](./examples/ner-service) |
| Pack discovery CLI | Supported via [`cmd/infergo-packs/`](./cmd/infergo-packs) |
| Parity CLI | Supported via [`cmd/infergo-parity/`](./cmd/infergo-parity) |
| REST serving CLI | Supported via [`cmd/infergo-serve/`](./cmd/infergo-serve) |
| Stable HTTP handler package | Supported via [`infer/httpserver/`](./infer/httpserver) |
| Family-1 alpha exporter | Experimental via [`cmd/infergo-export/`](./cmd/infergo-export) |
| Exported family-1 bundle serving | Experimental via [`cmd/infergo-serve -bundle`](./cmd/infergo-serve), with tokenizer-backed raw text for supported exported bundles |
| Token-classification BYOM export/import | Not part of alpha; token classification remains curated-pack and sample-service only |
| Structured JSON error responses | Supported |
| Graceful shutdown and timeout config | Supported via `infer/httpserver.ServerConfig` and `cmd/infergo-serve` |
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
- [`cmd/infergo-export/README.md`](./cmd/infergo-export/README.md)
- [`cmd/infergo-serve/README.md`](./cmd/infergo-serve/README.md)
- [`docs/alpha-roadmap.md`](./docs/alpha-roadmap.md)
- [`docs/alpha-supported-model-family.md`](./docs/alpha-supported-model-family.md)
- [`docs/alpha-native-artifact-contract.md`](./docs/alpha-native-artifact-contract.md)
- [`docs/alpha-family-1-exporter-contract.md`](./docs/alpha-family-1-exporter-contract.md)
- [`docs/alpha-family-1-walkthrough.md`](./docs/alpha-family-1-walkthrough.md)
- [`docs/alpha-family-2-entres-bridge.md`](./docs/alpha-family-2-entres-bridge.md)
- [`docs/alpha-family-2-validation-checklist.md`](./docs/alpha-family-2-validation-checklist.md)
- [`docs/alpha-gaps-and-missing-primitives.md`](./docs/alpha-gaps-and-missing-primitives.md)
- [`docs/releases/v0.1.0-prealpha.1.md`](./docs/releases/v0.1.0-prealpha.1.md)
