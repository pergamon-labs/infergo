# InferGo Alpha Family-1 Walkthrough

If you are evaluating InferGo as a user, start here.

If you are looking for roadmap or maintainer planning notes, use
[`alpha-roadmap.md`](./alpha-roadmap.md) separately.

This is the **public-safe end-to-end path** for the first supported InferGo
alpha family.

It shows how to:

1. export a supported PyTorch/Transformers model into an InferGo-native bundle
2. load the exported bundle from Go
3. optionally serve the exported bundle over HTTP
4. run parity against the source model

This walkthrough is intentionally narrow and honest:

- it uses the current projection-based family-1 exporter
- it targets the current supported BERT-style tokenizer-json subset
- it proves the supported alpha path without claiming general transformer
  execution in Go

## Supported examples

Single-text classification:

- `distilbert/distilbert-base-uncased-finetuned-sst-2-english`

Paired-text classification:

- `textattack/bert-base-uncased-MRPC`

## Prerequisites

- Go `1.26.1` or newer
- [`uv`](https://docs.astral.sh/uv/)
- internet access for the first model download from Hugging Face
- a writable local workspace for your bundle and input files

Install the CLI entrypoints:

```bash
go install github.com/pergamon-labs/infergo/cmd/infergo-export@latest
go install github.com/pergamon-labs/infergo/cmd/infergo-serve@latest
go install github.com/pergamon-labs/infergo/cmd/infergo-parity@latest
```

## 1. Export a supported model

### Single-text example

Write a starter input set:

```bash
infergo-export template -kind single -out ./family1-inputs.json
```

Then edit `./family1-inputs.json` with a few representative public-safe
examples and export:

```bash
infergo-export export \
  -model distilbert/distilbert-base-uncased-finetuned-sst-2-english \
  -input ./family1-inputs.json \
  -out ./artifacts/distilbert-sst2-alpha \
  -reference-output ./artifacts/distilbert-sst2-source-reference.json
```

### Paired-text example

Write a starter paired-text input set:

```bash
infergo-export template -kind pair -out ./family1-pairs.json
```

Then edit `./family1-pairs.json` and export:

```bash
infergo-export export \
  -model textattack/bert-base-uncased-MRPC \
  -input ./family1-pairs.json \
  -out ./artifacts/mrpc-alpha \
  -reference-output ./artifacts/mrpc-source-reference.json
```

The exported bundle is an InferGo-native directory with:

```text
artifacts/mrpc-alpha/
  metadata.json
  labels.json
  model.gob
  embeddings.gob
  tokenizer/
    manifest.json
    tokenizer.json
    tokenizer_config.json
```

## 2. Load and run the exported bundle from Go

This is the primary runtime story for most Go teams. Use
`infer.LoadTextBundle(...)` inside your existing service and call the model in
process.

### Single-text example

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

### Paired-text example

```go
package main

import (
	"fmt"
	"log"

	"github.com/pergamon-labs/infergo/infer"
)

func main() {
	bundle, err := infer.LoadTextBundle("./artifacts/mrpc-alpha")
	if err != nil {
		log.Fatal(err)
	}
	defer bundle.Close()

	result, err := bundle.PredictTextPair(
		"The company said the deal closed.",
		"The acquisition has been completed, the company said.",
	)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("label=%s logits=%v\n", result.Label, result.Logits)
}
```

If you also cloned the repo, you can run the bundled example directly:

```bash
go run ./examples/exported-bundle-classifier \
  -bundle ./artifacts/distilbert-sst2-alpha \
  -text "This product is excellent and reliable."
```

Or for paired text:

```bash
go run ./examples/exported-bundle-classifier \
  -bundle ./artifacts/mrpc-alpha \
  -text "The company said the deal closed." \
  -text-pair "The acquisition has been completed, the company said."
```

## 3. Optionally serve the exported bundle over HTTP

Use `infergo-serve` only when you want a standalone model process or a simple
HTTP boundary. If you already have a Go service, most of the time you can stop
after step 2.

### Single-text bundle

```bash
infergo-serve -task text -bundle ./artifacts/distilbert-sst2-alpha -addr 127.0.0.1:8080
```

Then call it:

```bash
curl -s -X POST http://127.0.0.1:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"This product is excellent and reliable."}'
```

### Paired-text bundle

```bash
infergo-serve -task text -bundle ./artifacts/mrpc-alpha -addr 127.0.0.1:8080
```

Then call it:

```bash
curl -s -X POST http://127.0.0.1:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"The company said the deal closed.","text_pair":"The acquisition has been completed, the company said."}'
```

You can inspect the supported inputs through:

```bash
curl -s http://127.0.0.1:8080/metadata
```

For the MRPC example, the metadata should include:

- `supports_raw_text=true`
- `supports_pair_text=true`
- `supported_inputs=["text","text+text_pair","input_ids"]`

## 4. Run parity against the source model

### Single-text parity

```bash
infergo-parity \
  -reference ./artifacts/distilbert-sst2-source-reference.json \
  -infergo-bundle-dir ./artifacts/distilbert-sst2-alpha \
  -tolerance 1e-4
```

### Paired-text parity

```bash
infergo-parity \
  -reference ./artifacts/mrpc-source-reference.json \
  -infergo-bundle-dir ./artifacts/mrpc-alpha \
  -tolerance 1e-4
```

## What this proves today

This walkthrough proves that a backend engineer can:

- start from a supported PyTorch/Transformers model
- export it into an InferGo-native bundle
- load it from Go without Python at runtime
- serve it over HTTP
- verify parity against the source implementation

## What it does not prove

This walkthrough does **not** mean:

- InferGo supports arbitrary Hugging Face models
- InferGo is a general transformer runtime
- InferGo supports every tokenizer family
- InferGo loads raw `.pt` files directly

The supported family and contract still live here:
- [alpha-supported-model-family.md](./alpha-supported-model-family.md)
- [alpha-native-artifact-contract.md](./alpha-native-artifact-contract.md)
- [alpha-family-1-exporter-contract.md](./alpha-family-1-exporter-contract.md)
