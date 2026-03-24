package parity

import (
	"encoding/json"
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

	reference := TransformersTextClassificationReference{
		Name:        "synthetic reference",
		Source:      "test",
		ModelID:     "example/model",
		Task:        "text-classification",
		GeneratedAt: "2026-03-24T00:00:00Z",
		Labels:      []string{"NEGATIVE", "POSITIVE"},
		Cases: []TransformersTextClassificationReferenceCase{
			{
				ID:                    "case-a",
				Text:                  "excellent support",
				InputIDs:              []int{101, 2001, 102},
				AttentionMask:         []int{1, 1, 1},
				ExpectedLogits:        []float64{-1.5, 1.5},
				ExpectedProbabilities: softmax([]float64{-1.5, 1.5}),
				ExpectedLabel:         "POSITIVE",
			},
			{
				ID:                    "case-b",
				Text:                  "terrible experience",
				InputIDs:              []int{101, 2002, 102},
				AttentionMask:         []int{1, 1, 1},
				ExpectedLogits:        []float64{1.8, -1.8},
				ExpectedProbabilities: softmax([]float64{1.8, -1.8}),
				ExpectedLabel:         "NEGATIVE",
			},
		},
	}
	candidate := TextClassificationCandidate{
		Name:        "candidate",
		Source:      "test",
		ModelID:     "example/model",
		Task:        "text-classification",
		Artifact:    "dist/model",
		GeneratedAt: "2026-03-24T00:00:00Z",
		Labels:      []string{"NEGATIVE", "POSITIVE"},
		Cases: []TextClassificationCandidateCase{
			{
				ID:                    "case-a",
				Text:                  "excellent support",
				InputIDs:              []int{101, 2001, 102},
				AttentionMask:         []int{1, 1, 1},
				ObservedLogits:        []float64{-1.5, 1.5},
				ObservedProbabilities: softmax([]float64{-1.5, 1.5}),
				ObservedLabel:         "POSITIVE",
			},
			{
				ID:                    "case-b",
				Text:                  "terrible experience",
				InputIDs:              []int{101, 2002, 102},
				AttentionMask:         []int{1, 1, 1},
				ObservedLogits:        []float64{1.8, -1.8},
				ObservedProbabilities: softmax([]float64{1.8, -1.8}),
				ObservedLabel:         "NEGATIVE",
			},
		},
	}

	referencePath := writeTempJSON(t, "reference.json", reference)
	candidatePath := writeTempJSON(t, "candidate.json", candidate)

	report, err := CompareTransformersTextClassification(referencePath, candidatePath, 1e-9)
	if err != nil {
		t.Fatalf("CompareTransformersTextClassification() error = %v", err)
	}

	if !report.Passed() {
		t.Fatalf("expected comparison to pass, got report:\n%s", report.String())
	}
}

func TestCompareTransformersTextClassificationRequiresFullCandidateCoverage(t *testing.T) {
	t.Parallel()

	reference := TransformersTextClassificationReference{
		Name:        "synthetic reference",
		Source:      "test",
		ModelID:     "example/model",
		Task:        "text-classification",
		GeneratedAt: "2026-03-24T00:00:00Z",
		Labels:      []string{"NEGATIVE", "POSITIVE"},
		Cases: []TransformersTextClassificationReferenceCase{
			{
				ID:                    "case-a",
				Text:                  "excellent support",
				InputIDs:              []int{101, 2001, 102},
				AttentionMask:         []int{1, 1, 1},
				ExpectedLogits:        []float64{-1.5, 1.5},
				ExpectedProbabilities: softmax([]float64{-1.5, 1.5}),
				ExpectedLabel:         "POSITIVE",
			},
			{
				ID:                    "case-b",
				Text:                  "terrible experience",
				InputIDs:              []int{101, 2002, 102},
				AttentionMask:         []int{1, 1, 1},
				ExpectedLogits:        []float64{1.8, -1.8},
				ExpectedProbabilities: softmax([]float64{1.8, -1.8}),
				ExpectedLabel:         "NEGATIVE",
			},
		},
	}
	candidate := TextClassificationCandidate{
		Name:        "candidate",
		Source:      "test",
		ModelID:     "example/model",
		Task:        "text-classification",
		Artifact:    "dist/model",
		GeneratedAt: "2026-03-24T00:00:00Z",
		Labels:      []string{"NEGATIVE", "POSITIVE"},
		Cases: []TextClassificationCandidateCase{
			{
				ID:                    "case-a",
				Text:                  "excellent support",
				InputIDs:              []int{101, 2001, 102},
				AttentionMask:         []int{1, 1, 1},
				ObservedLogits:        []float64{-1.5, 1.5},
				ObservedProbabilities: softmax([]float64{-1.5, 1.5}),
				ObservedLabel:         "POSITIVE",
			},
		},
	}

	referencePath := writeTempJSON(t, "reference.json", reference)
	candidatePath := writeTempJSON(t, "candidate.json", candidate)

	_, err := CompareTransformersTextClassification(referencePath, candidatePath, 1e-9)
	if err == nil {
		t.Fatal("expected comparison to fail when a reference case is missing from the candidate")
	}
}

func TestBuildTextClassificationCandidate(t *testing.T) {
	t.Parallel()

	reference := TransformersTextClassificationReference{
		Name:        "synthetic reference",
		Source:      "test",
		ModelID:     "example/model",
		Task:        "text-classification",
		GeneratedAt: "2026-03-24T00:00:00Z",
		Labels:      []string{"NEGATIVE", "POSITIVE"},
		Cases: []TransformersTextClassificationReferenceCase{
			{ID: "case-a", Text: "excellent support", InputIDs: []int{101, 2001, 102}, AttentionMask: []int{1, 1, 1}},
			{ID: "case-b", Text: "terrible experience", InputIDs: []int{101, 2002, 102}, AttentionMask: []int{1, 1, 1}},
		},
	}

	predictor := fakeTorchScriptPredictor{
		logits: [][]float64{
			{-1.5, 1.5},
			{1.8, -1.8},
		},
		labels:  []string{"NEGATIVE", "POSITIVE"},
		modelID: "example/model",
	}

	candidate, err := BuildTextClassificationCandidate(reference, predictor, "dist/model", "infergo-test")
	if err != nil {
		t.Fatalf("BuildTextClassificationCandidate() error = %v", err)
	}

	if len(candidate.Cases) != len(reference.Cases) {
		t.Fatalf("expected %d candidate cases, got %d", len(reference.Cases), len(candidate.Cases))
	}

	if candidate.Cases[0].ObservedLabel != "POSITIVE" {
		t.Fatalf("unexpected observed label %q", candidate.Cases[0].ObservedLabel)
	}
}

func TestRunBionetTextClassificationBundle(t *testing.T) {
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
		for _, bundleDir := range tt.bundleDirs {
			candidatePath := filepath.Join(t.TempDir(), filepath.Base(bundleDir)+".json")

			candidate, err := RunBionetTextClassificationBundle(tt.referencePath, bundleDir)
			if err != nil {
				t.Fatalf("RunBionetTextClassificationBundle(%q) error = %v", bundleDir, err)
			}

			if err := SaveTextClassificationCandidate(candidate, candidatePath); err != nil {
				t.Fatalf("SaveTextClassificationCandidate(%q) error = %v", bundleDir, err)
			}

			report, err := CompareTransformersTextClassification(tt.referencePath, candidatePath, 1e-4)
			if err != nil {
				t.Fatalf("CompareTransformersTextClassification(%q) error = %v", bundleDir, err)
			}

			if !report.Passed() {
				t.Fatalf("expected bionet native comparison for %q to pass, got report:\n%s", bundleDir, report.String())
			}
		}
	}
}

func writeTempJSON(t *testing.T, name string, value any) string {
	t.Helper()

	raw, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		t.Fatalf("json.MarshalIndent(%s) error = %v", name, err)
	}

	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, append(raw, '\n'), 0o644); err != nil {
		t.Fatalf("WriteFile(%s) error = %v", name, err)
	}

	return path
}
