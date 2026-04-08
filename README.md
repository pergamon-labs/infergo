# InferGo

InferGo is a Go-native inference toolkit for backend services.

It is built for teams that want to export a supported model once, load it in
Go, and run predictions without Python in production.

Current prerelease:
[`v0.1.0-prealpha.1`](https://github.com/pergamon-labs/infergo/releases/tag/v0.1.0-prealpha.1)

Start here:

- [docs/GETTING_STARTED.md](/Users/jatto/Documents/workspaces/pergamon-labs/infergo/docs/GETTING_STARTED.md)
- [docs/USE_CASES.md](/Users/jatto/Documents/workspaces/pergamon-labs/infergo/docs/USE_CASES.md)
- [docs/PHILOSOPHY.md](/Users/jatto/Documents/workspaces/pergamon-labs/infergo/docs/PHILOSOPHY.md)

## What InferGo is for

- embed small inference workloads directly in Go services
- optionally expose those same bundles through a standalone HTTP process
- keep support claims narrow, explicit, and parity-backed

InferGo is **not**:

- a blanket `.pt` loader
- a general transformer runtime
- a training framework
- a model zoo

## Installation

InferGo currently targets Go `1.26.1` or newer.

Add the library to your project:

```bash
go get github.com/pergamon-labs/infergo
```

Install the main CLIs:

```bash
go install github.com/pergamon-labs/infergo/cmd/infergo-export@latest
go install github.com/pergamon-labs/infergo/cmd/infergo-serve@latest
go install github.com/pergamon-labs/infergo/cmd/infergo-parity@latest
```

`infergo-export` still needs `uv` plus Python-side `transformers` dependencies
at export time. That dependency does not carry into runtime serving once the
bundle is built.

## Quickstart: bring your own model

This is the primary alpha path. It does **not** require cloning this repo.

1. Write a starter input set:

```bash
infergo-export template -kind single -out ./family1-inputs.json
```

2. Edit `./family1-inputs.json` so it contains a few representative,
public-safe examples from your task.

3. Export a supported model:

```bash
infergo-export export \
  -model distilbert/distilbert-base-uncased-finetuned-sst-2-english \
  -input ./family1-inputs.json \
  -out ./artifacts/distilbert-sst2-alpha \
  -reference-output ./artifacts/distilbert-sst2-reference.json
```

4. Use the exported bundle from Go:

```go
package main

import (
	"fmt"
	"log"

	"github.com/pergamon-labs/infergo/infer"
)

func main() {
	bundle, err := infer.LoadTextBundle("./artifacts/distilbert-sst2-alpha")
	if err != nil {
		log.Fatal(err)
	}
	defer bundle.Close()

	result, err := bundle.PredictText("This product is excellent and reliable.")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("label=%s logits=%v\n", result.Label, result.Logits)
}
```

5. Validate parity:

```bash
infergo-parity \
  -reference ./artifacts/distilbert-sst2-reference.json \
  -infergo-bundle-dir ./artifacts/distilbert-sst2-alpha \
  -tolerance 1e-4
```

For the full supported path, see
[docs/alpha-family-1-walkthrough.md](/Users/jatto/Documents/workspaces/pergamon-labs/infergo/docs/alpha-family-1-walkthrough.md).

## Use in your Go app

For most Go teams, this is the normal mode: load the bundle once and call it
from your existing service code.

Single-text classification:

```go
bundle, err := infer.LoadTextBundle("./artifacts/distilbert-sst2-alpha")
if err != nil {
	log.Fatal(err)
}
defer bundle.Close()

result, err := bundle.PredictText("This product is excellent and reliable.")
if err != nil {
	log.Fatal(err)
}
```

Paired-text classification:

```go
result, err := bundle.PredictTextPair(
	"The company said the deal closed.",
	"The acquisition has been completed, the company said.",
)
if err != nil {
	log.Fatal(err)
}
```

If you want repo-based examples too, see:

- [examples/exported-bundle-classifier/README.md](/Users/jatto/Documents/workspaces/pergamon-labs/infergo/examples/exported-bundle-classifier/README.md)
- [examples/bionet-classifier/README.md](/Users/jatto/Documents/workspaces/pergamon-labs/infergo/examples/bionet-classifier/README.md)

## Serve over HTTP, if you want a standalone model process

Most teams embedding InferGo in an existing Go service will not need this.

Use `infergo-serve` when you want:

- a quick smoke-test surface
- a separate model process
- a simple HTTP boundary for non-Go callers

Installed binary:

```bash
infergo-serve -task text -bundle ./artifacts/distilbert-sst2-alpha -addr 127.0.0.1:8080
```

Repo-local alternative:

```bash
go run ./cmd/infergo-serve -task text -bundle ./artifacts/distilbert-sst2-alpha -addr 127.0.0.1:8080
```

Then call it:

```bash
curl -s -X POST http://127.0.0.1:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"This product is excellent and reliable."}'
```

Paired-text exported bundles accept:

```bash
curl -s -X POST http://127.0.0.1:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"The company said the deal closed.","text_pair":"The acquisition has been completed, the company said."}'
```

## Curated repo examples

If you cloned the repo and want to evaluate the current checked-in fixtures:

- list packs:

```bash
go run ./cmd/infergo-packs
```

- run a curated text example:

```bash
go run ./examples/bionet-classifier -text "This product is excellent and reliable."
```

- run a curated token example:

```bash
go run ./cmd/infergo-serve -task token
```

- run the sample NER extraction service:

```bash
go run ./examples/ner-service -addr 127.0.0.1:8080
```

## Supported today

InferGo is intentionally narrow in `v0.1.0-prealpha.1`.

| Area | Status |
| --- | --- |
| Family-1 BYOM export/import for text classification | Experimental |
| Exported family-1 bundles loaded directly in Go | Experimental |
| Exported family-1 bundles served over HTTP | Experimental |
| Curated native text-classification packs | Supported |
| Curated native token-classification packs | Supported |
| Sample NER extraction service | Supported |
| Token-classification BYOM export/import | Not part of alpha |
| Family-2 `entres` bridge | Experimental and internal-first |
| gRPC serving surface | Not yet |

## Project status

InferGo is public, but still pre-alpha.

Current posture:

- CPU-first native inference in Go
- library-first usage
- HTTP as an optional deployment mode
- parity-backed support for documented paths only

## Documentation

For users:

- [docs/GETTING_STARTED.md](/Users/jatto/Documents/workspaces/pergamon-labs/infergo/docs/GETTING_STARTED.md)
- [docs/USE_CASES.md](/Users/jatto/Documents/workspaces/pergamon-labs/infergo/docs/USE_CASES.md)
- [docs/PHILOSOPHY.md](/Users/jatto/Documents/workspaces/pergamon-labs/infergo/docs/PHILOSOPHY.md)
- [docs/alpha-family-1-walkthrough.md](/Users/jatto/Documents/workspaces/pergamon-labs/infergo/docs/alpha-family-1-walkthrough.md)
- [cmd/infergo-export/README.md](/Users/jatto/Documents/workspaces/pergamon-labs/infergo/cmd/infergo-export/README.md)
- [cmd/infergo-serve/README.md](/Users/jatto/Documents/workspaces/pergamon-labs/infergo/cmd/infergo-serve/README.md)

Reference:

- [COMPATIBILITY.md](/Users/jatto/Documents/workspaces/pergamon-labs/infergo/COMPATIBILITY.md)
- [ARCHITECTURE.md](/Users/jatto/Documents/workspaces/pergamon-labs/infergo/ARCHITECTURE.md)
- [pkg.go.dev `infer`](https://pkg.go.dev/github.com/pergamon-labs/infergo/infer)
- [pkg.go.dev `infer/httpserver`](https://pkg.go.dev/github.com/pergamon-labs/infergo/infer/httpserver)

Project:

- [BENCHMARKS.md](/Users/jatto/Documents/workspaces/pergamon-labs/infergo/BENCHMARKS.md)
- [CONTRIBUTING.md](/Users/jatto/Documents/workspaces/pergamon-labs/infergo/CONTRIBUTING.md)
