# Getting Started

This is the fastest way to evaluate InferGo as it exists today.

## What you need

- Go `1.26.1` or newer
- `uv` for export-time Python tooling

Add InferGo to your project:

```bash
go get github.com/pergamon-labs/infergo
```

Install the CLI tools:

```bash
go install github.com/pergamon-labs/infergo/cmd/infergo-export@latest
go install github.com/pergamon-labs/infergo/cmd/infergo-serve@latest
go install github.com/pergamon-labs/infergo/cmd/infergo-parity@latest
```

You only need Python-side tooling at export time. You do not need Python in
production after the bundle is built.

Current raw-text bundle support is intentionally narrow: exported family-1
bundles can embed runtime tokenizer metadata today only for two validated
`hf-tokenizer-json` subsets:

- BERT-style WordPiece with `TemplateProcessing`
- RoBERTa-style ByteLevel BPE with `RobertaProcessing`

## Choose your path

### Path 1: Bring your own supported model

This is the main public alpha path.

1. Write a starter input file:

```bash
infergo-export template -kind single -out ./family1-inputs.json
```

2. Edit that file with a few representative public-safe examples.

3. Export a supported model:

```bash
infergo-export export \
  -model distilbert/distilbert-base-uncased-finetuned-sst-2-english \
  -input ./family1-inputs.json \
  -out ./artifacts/distilbert-sst2-alpha \
  -reference-output ./artifacts/distilbert-sst2-reference.json
```

4. Load it from Go:

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

For the full supported BYOM path, see
[`alpha-family-1-walkthrough.md`](./alpha-family-1-walkthrough.md).

If export fails, see
[`../cmd/infergo-export/README.md`](../cmd/infergo-export/README.md)
for the current supported boundary and common recovery steps.

### Path 2: Use a curated checked-in pack

This is useful when you want to try the current repo fixtures quickly.

If you cloned the repo:

```bash
go run ./cmd/infergo-packs
go run ./examples/bionet-classifier -text "This product is excellent and reliable."
```

### Path 3: Serve a bundle over HTTP

This is optional. Most Go teams will embed InferGo directly in an existing
service.

```bash
infergo-serve -task text -bundle ./artifacts/distilbert-sst2-alpha -addr 127.0.0.1:8080
```

Then call it:

```bash
curl -s -X POST http://127.0.0.1:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"This product is excellent and reliable."}'
```

## What to read next

- [`PHILOSOPHY.md`](./PHILOSOPHY.md)
- [`USE_CASES.md`](./USE_CASES.md)
- [`alpha-family-1-walkthrough.md`](./alpha-family-1-walkthrough.md)
- [`../COMPATIBILITY.md`](../COMPATIBILITY.md)
- [`../cmd/infergo-export/README.md`](../cmd/infergo-export/README.md)
- [`../cmd/infergo-serve/README.md`](../cmd/infergo-serve/README.md)
