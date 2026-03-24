package parity

import (
	"os"
	"path/filepath"
	"testing"
)

type fakeTorchScriptPredictor struct {
	logits  [][]float64
	labels  []string
	modelID string
}

func (f fakeTorchScriptPredictor) PredictBatch(inputIDs, attentionMasks [][]int64) ([][]float64, error) {
	return f.logits, nil
}

func (f fakeTorchScriptPredictor) Labels() []string { return f.labels }
func (f fakeTorchScriptPredictor) ModelID() string  { return f.modelID }

func TestCompareTransformersTextClassification(t *testing.T) {
	t.Parallel()

	referencePath := "../../testdata/reference/text-classification/distilbert-sst2-reference.json"
	candidatePath := filepath.Join(t.TempDir(), "candidate.json")

	payload := `{
  "name": "candidate",
  "source": "torchscript",
  "model_id": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
  "task": "text-classification",
  "artifact": "dist/model.torchscript.pt",
  "generated_at": "2026-03-23T00:00:00Z",
  "labels": ["NEGATIVE", "POSITIVE"],
  "cases": [
    {
      "id": "positive-review",
      "text": "This product is excellent and reliable.",
      "input_ids": [101,2023,4031,2003,6581,1998,10539,1012,102],
      "attention_mask": [1,1,1,1,1,1,1,1,1],
      "observed_logits": [-4.337915420532227,4.705198764801025],
      "observed_probabilities": [0.00011818823986686766,0.9998817443847656],
      "observed_label": "POSITIVE"
    },
    {
      "id": "negative-review",
      "text": "This was a terrible experience.",
      "input_ids": [101,2023,2001,1037,6659,3325,1012,102],
      "attention_mask": [1,1,1,1,1,1,1,1],
      "observed_logits": [3.716874599456787,-3.1733884811401367],
      "observed_probabilities": [0.9989833235740662,0.0010166114661842585],
      "observed_label": "NEGATIVE"
    },
    {
      "id": "mixed-review",
      "text": "The service was okay overall.",
      "input_ids": [101,1996,2326,2001,3100,3452,1012,102],
      "attention_mask": [1,1,1,1,1,1,1,1],
      "observed_logits": [-4.009629726409912,4.371831893920898],
      "observed_probabilities": [0.0002290222910232842,0.9997709393501282],
      "observed_label": "POSITIVE"
    },
    {
      "id": "support-praise",
      "text": "Fast helpful support team.",
      "input_ids": [101,3435,14044,2490,2136,1012,102],
      "attention_mask": [1,1,1,1,1,1,1],
      "observed_logits": [-4.13802433013916,4.479250431060791],
      "observed_probabilities": [0.00018091990204993635,0.9998190999031067],
      "observed_label": "POSITIVE"
    }
  ]
}`

	if err := os.WriteFile(candidatePath, []byte(payload), 0o644); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	report, err := CompareTransformersTextClassification(referencePath, candidatePath, 1e-9)
	if err != nil {
		t.Fatalf("CompareTransformersTextClassification() error = %v", err)
	}

	if !report.Passed() {
		t.Fatalf("expected comparison to pass, got report:\n%s", report.String())
	}
}

func TestBuildTextClassificationCandidate(t *testing.T) {
	t.Parallel()

	reference, err := LoadTransformersTextClassificationReference("../../testdata/reference/text-classification/distilbert-sst2-reference.json")
	if err != nil {
		t.Fatalf("LoadTransformersTextClassificationReference() error = %v", err)
	}

	predictor := fakeTorchScriptPredictor{
		logits: [][]float64{
			{-4.337915420532227, 4.705198764801025},
			{3.716874599456787, -3.1733884811401367},
			{-4.009629726409912, 4.371831893920898},
			{-4.13802433013916, 4.479250431060791},
		},
		labels:  []string{"NEGATIVE", "POSITIVE"},
		modelID: "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
	}

	candidate, err := BuildTextClassificationCandidate(reference, predictor, "dist/torchscript/distilbert-sst2/model.torchscript.pt", "infergo-test")
	if err != nil {
		t.Fatalf("BuildTextClassificationCandidate() error = %v", err)
	}

	if len(candidate.Cases) != 4 {
		t.Fatalf("expected 4 candidate cases, got %d", len(candidate.Cases))
	}

	if candidate.Cases[0].ObservedLabel != "POSITIVE" {
		t.Fatalf("unexpected observed label %q", candidate.Cases[0].ObservedLabel)
	}
}

func TestRunBionetTextClassificationBundle(t *testing.T) {
	t.Parallel()

	referencePath := "../../testdata/reference/text-classification/distilbert-sst2-reference.json"
	bundleDir := "../../testdata/native/text-classification/distilbert-sst2-token-id-bag"
	candidatePath := filepath.Join(t.TempDir(), "candidate.json")

	candidate, err := RunBionetTextClassificationBundle(referencePath, bundleDir)
	if err != nil {
		t.Fatalf("RunBionetTextClassificationBundle() error = %v", err)
	}

	if err := SaveTextClassificationCandidate(candidate, candidatePath); err != nil {
		t.Fatalf("SaveTextClassificationCandidate() error = %v", err)
	}

	report, err := CompareTransformersTextClassification(referencePath, candidatePath, 1e-4)
	if err != nil {
		t.Fatalf("CompareTransformersTextClassification() error = %v", err)
	}

	if !report.Passed() {
		t.Fatalf("expected bionet native comparison to pass, got report:\n%s", report.String())
	}
}
