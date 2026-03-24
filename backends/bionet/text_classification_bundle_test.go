package bionet_test

import (
	"testing"

	"github.com/pergamon-labs/infergo/backends/bionet"
	"github.com/pergamon-labs/infergo/internal/parity"
)

func TestLoadTextClassificationBundle(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		referencePath string
		bundleDirs    []string
	}{
		{
			referencePath: "../../testdata/reference/text-classification/distilbert-sst2-reference.json",
			bundleDirs: []string{
				"../../testdata/native/text-classification/distilbert-sst2-token-id-bag",
				"../../testdata/native/text-classification/distilbert-sst2-embedding-avg-pool",
				"../../testdata/native/text-classification/distilbert-sst2-embedding-masked-avg-pool",
			},
		},
		{
			referencePath: "../../testdata/reference/text-classification/twitter-roberta-sentiment-reference.json",
			bundleDirs: []string{
				"../../testdata/native/text-classification/twitter-roberta-sentiment-embedding-masked-avg-pool",
			},
		},
	}

	for _, tt := range testCases {
		reference, err := parity.LoadTransformersTextClassificationReference(tt.referencePath)
		if err != nil {
			t.Fatalf("LoadTransformersTextClassificationReference(%q) error = %v", tt.referencePath, err)
		}

		inputIDs := make([][]int64, len(reference.Cases))
		attentionMasks := make([][]int64, len(reference.Cases))
		for i, item := range reference.Cases {
			inputIDs[i] = intsToInt64(item.InputIDs)
			attentionMasks[i] = intsToInt64(item.AttentionMask)
		}

		for _, bundleDir := range tt.bundleDirs {
			bundle, err := bionet.LoadTextClassificationBundle(bundleDir)
			if err != nil {
				t.Fatalf("LoadTextClassificationBundle(%q) error = %v", bundleDir, err)
			}

			logitsBatch, err := bundle.PredictBatch(inputIDs, attentionMasks)
			if err != nil {
				t.Fatalf("PredictBatch(%q) error = %v", bundleDir, err)
			}

			if len(logitsBatch) != len(reference.Cases) {
				t.Fatalf("PredictBatch(%q) batch size = %d, want %d", bundleDir, len(logitsBatch), len(reference.Cases))
			}

			for i, logits := range logitsBatch {
				labelIdx := argMax(logits)
				if got, want := bundle.Labels()[labelIdx], reference.Cases[i].ExpectedLabel; got != want {
					t.Fatalf("bundle %q case %q label = %q, want %q", bundleDir, reference.Cases[i].ID, got, want)
				}
			}
		}
	}
}

func intsToInt64(values []int) []int64 {
	output := make([]int64, len(values))
	for i, value := range values {
		output[i] = int64(value)
	}

	return output
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
