# InferGo Alpha Family-1 Walkthrough

This is the **public-safe end-to-end path** for the first supported InferGo
alpha family.

It shows how to:

1. export a supported PyTorch/Transformers model into an InferGo-native bundle
2. load the exported bundle from Go
3. serve the exported bundle over HTTP
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

Clone the repo and verify the baseline:

```bash
git clone https://github.com/pergamon-labs/infergo.git
cd infergo
go test ./...
```

## 1. Export a supported model

### Single-text example

```bash
uv run --with torch==2.10.0 --with transformers==5.3.0 \
  python ./scripts/export_encoder_text_bundle.py \
  --model distilbert/distilbert-base-uncased-finetuned-sst-2-english \
  --input ./testdata/reference/text-classification/sst2-inputs.json \
  --out ./dist/family1/distilbert-sst2-alpha \
  --reference-output ./dist/family1/distilbert-sst2-source-reference.json
```

### Paired-text example

```bash
uv run --with torch==2.10.0 --with transformers==5.3.0 \
  python ./scripts/export_encoder_text_bundle.py \
  --model textattack/bert-base-uncased-MRPC \
  --input ./testdata/reference/text-classification/mrpc-pairs-inputs.json \
  --out ./dist/family1/mrpc-alpha \
  --reference-output ./dist/family1/mrpc-source-reference.json
```

The exported bundle is an InferGo-native directory with:

```text
dist/family1/mrpc-alpha/
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

For the higher-level family-1 library path, use `infer.LoadTextBundle(...)`.

### Single-text example

```go
package main

import (
	"fmt"
	"log"

	"github.com/pergamon-labs/infergo/infer"
)

func main() {
	bundle, err := infer.LoadTextBundle("./dist/family1/distilbert-sst2-alpha")
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
	bundle, err := infer.LoadTextBundle("./dist/family1/mrpc-alpha")
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

You can also run the repo example directly:

```bash
go run ./examples/exported-bundle-classifier \
  -bundle ./dist/family1/distilbert-sst2-alpha \
  -text "This product is excellent and reliable."
```

Or for paired text:

```bash
go run ./examples/exported-bundle-classifier \
  -bundle ./dist/family1/mrpc-alpha \
  -text "The company said the deal closed." \
  -text-pair "The acquisition has been completed, the company said."
```

## 3. Serve the exported bundle over HTTP

### Single-text bundle

```bash
go run ./cmd/infergo-serve \
  -task text \
  -bundle ./dist/family1/distilbert-sst2-alpha \
  -addr 127.0.0.1:8080
```

Then call it:

```bash
curl -s -X POST http://127.0.0.1:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"This product is excellent and reliable."}'
```

### Paired-text bundle

```bash
go run ./cmd/infergo-serve \
  -task text \
  -bundle ./dist/family1/mrpc-alpha \
  -addr 127.0.0.1:8080
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
go run ./cmd/infergo-parity \
  -reference ./dist/family1/distilbert-sst2-source-reference.json \
  -infergo-bundle-dir ./dist/family1/distilbert-sst2-alpha \
  -tolerance 1e-4
```

### Paired-text parity

```bash
go run ./cmd/infergo-parity \
  -reference ./dist/family1/mrpc-source-reference.json \
  -infergo-bundle-dir ./dist/family1/mrpc-alpha \
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
