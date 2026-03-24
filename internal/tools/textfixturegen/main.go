package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/pergamon-labs/infergo/backends/bionet"
	"github.com/pergamon-labs/infergo/backends/bionet/runtime/functional"
	"github.com/pergamon-labs/infergo/backends/bionet/runtime/module"
	"github.com/pergamon-labs/infergo/backends/bionet/runtime/tensor"
	"github.com/pergamon-labs/infergo/internal/parity"
)

func main() {
	outputDir := filepath.Join("testdata", "parity", "text-classification")
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		fail("create output directory", err)
	}

	artifactPath := filepath.Join(outputDir, "model.gob")
	fixturePath := filepath.Join(outputDir, "fixture.json")

	model := module.New(
		functional.ActivationSoftmax,
		functional.ActivationParams{},
		module.New(
			functional.ActivationLinear,
			functional.ActivationParams{
				Weights: tensor.New([]float64{
					1.6, -1.2, 0.8, 1.4,
					-1.4, 1.1, -0.6, -1.3,
				}, []int{2, 4}),
				Bias: tensor.New([]float64{0.2, -0.2}, []int{2}),
			},
		),
	)

	if err := module.SaveToFile(model, artifactPath); err != nil {
		fail("save model artifact", err)
	}

	loadedModel, err := bionet.LoadModel(artifactPath)
	if err != nil {
		fail("load generated artifact", err)
	}

	fixture := parity.TextClassificationFixture{
		Name:      "Synthetic Text Classification",
		Backend:   bionet.Backend{}.Name(),
		Artifact:  "model.gob",
		Labels:    []string{"positive", "negative"},
		Tolerance: 1e-9,
		Cases: []parity.TextClassificationCase{
			{
				ID:       "positive-review",
				Text:     "this product is excellent",
				Features: []float64{2, 0, 1, 1},
			},
			{
				ID:       "negative-review",
				Text:     "this was a terrible experience",
				Features: []float64{0, 2, 1, 0},
			},
			{
				ID:       "mixed-review",
				Text:     "the service was okay",
				Features: []float64{1, 1, 0, 0},
			},
			{
				ID:       "support-praise",
				Text:     "fast reliable helpful team",
				Features: []float64{1, 0, 2, 1},
			},
		},
	}

	for i := range fixture.Cases {
		probabilities, err := loadedModel.PredictVector(fixture.Cases[i].Features)
		if err != nil {
			fail(fmt.Sprintf("predict fixture case %q", fixture.Cases[i].ID), err)
		}

		fixture.Cases[i].ExpectedProbabilities = probabilities
		fixture.Cases[i].ExpectedLabel = fixture.Labels[argMax(probabilities)]
	}

	file, err := os.Create(fixturePath)
	if err != nil {
		fail("create fixture json", err)
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(fixture); err != nil {
		fail("encode fixture json", err)
	}
}

func argMax(values []float64) int {
	bestIdx := 0
	bestValue := values[0]
	for i := 1; i < len(values); i++ {
		if values[i] > bestValue {
			bestValue = values[i]
			bestIdx = i
		}
	}

	return bestIdx
}

func fail(action string, err error) {
	fmt.Fprintf(os.Stderr, "textfixturegen: %s: %v\n", action, err)
	os.Exit(1)
}
