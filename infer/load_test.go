package infer_test

import (
	"testing"

	"github.com/pergamon-labs/infergo/infer"
	"github.com/pergamon-labs/infergo/internal/parity"
)

func TestLoadTextClassifierAndPredict(t *testing.T) {
	t.Parallel()

	classifier, err := infer.LoadTextClassifier("../testdata/native/text-classification/distilbert-sst2-embedding-masked-avg-pool")
	if err != nil {
		t.Fatalf("LoadTextClassifier() error = %v", err)
	}
	defer classifier.Close()

	reference, err := parity.LoadTransformersTextClassificationReference("../testdata/reference/text-classification/distilbert-sst2-reference.json")
	if err != nil {
		t.Fatalf("LoadTransformersTextClassificationReference() error = %v", err)
	}

	item := reference.Cases[0]
	prediction, err := classifier.Predict(infer.TextInput{InputIDs: intsToInt64(item.InputIDs)})
	if err != nil {
		t.Fatalf("Predict() error = %v", err)
	}

	if prediction.Backend != "bionet" {
		t.Fatalf("prediction.Backend = %q, want bionet", prediction.Backend)
	}
	if prediction.ModelID != reference.ModelID {
		t.Fatalf("prediction.ModelID = %q, want %q", prediction.ModelID, reference.ModelID)
	}
	if prediction.Label != item.ExpectedLabel {
		t.Fatalf("prediction.Label = %q, want %q", prediction.Label, item.ExpectedLabel)
	}
}

func TestLoadTokenClassifierAndPredict(t *testing.T) {
	t.Parallel()

	classifier, err := infer.LoadTokenClassifier("../testdata/native/token-classification/distilcamembert-french-ner-windowed-embedding-linear")
	if err != nil {
		t.Fatalf("LoadTokenClassifier() error = %v", err)
	}
	defer classifier.Close()

	reference, err := parity.LoadTransformersTokenClassificationReference("../testdata/reference/token-classification/distilcamembert-french-ner-reference.json")
	if err != nil {
		t.Fatalf("LoadTransformersTokenClassificationReference() error = %v", err)
	}

	item := reference.Cases[0]
	prediction, err := classifier.Predict(infer.TokenInput{InputIDs: intsToInt64(item.InputIDs)})
	if err != nil {
		t.Fatalf("Predict() error = %v", err)
	}

	if prediction.Backend != "bionet" {
		t.Fatalf("prediction.Backend = %q, want bionet", prediction.Backend)
	}
	if got, want := len(prediction.TokenLabels), len(item.ExpectedLabels); got != want {
		t.Fatalf("len(prediction.TokenLabels) = %d, want %d", got, want)
	}
	if got, want := prediction.TokenLabels[0], item.ExpectedLabels[0]; got != want {
		t.Fatalf("prediction.TokenLabels[0] = %q, want %q", got, want)
	}
}

func intsToInt64(values []int) []int64 {
	output := make([]int64, len(values))
	for i, value := range values {
		output[i] = int64(value)
	}
	return output
}
