package parity

import (
	"path/filepath"
	"testing"

	"github.com/pergamon-labs/infergo/internal/modelpacks"
)

const tokenClassificationParityTolerance = 1e-3

type fakeTokenClassificationPredictor struct {
	logits  [][][]float64
	labels  []string
	modelID string
}

func (f fakeTokenClassificationPredictor) PredictBatch(inputIDs, attentionMasks [][]int64) ([][][]float64, error) {
	return f.logits, nil
}

func (f fakeTokenClassificationPredictor) Labels() []string { return f.labels }
func (f fakeTokenClassificationPredictor) ModelID() string  { return f.modelID }

func TestCompareTransformersTokenClassification(t *testing.T) {
	t.Parallel()

	reference := TransformersTokenClassificationReference{
		Name:        "synthetic token reference",
		Source:      "test",
		ModelID:     "example/model",
		Task:        "token-classification",
		GeneratedAt: "2026-03-26T00:00:00Z",
		Labels:      []string{"O", "B-PER"},
		Cases: []TransformersTokenClassificationReferenceCase{
			{
				ID:                    "case-a",
				Text:                  "John arrived",
				Tokens:                []string{"john", "arrived"},
				InputIDs:              []int{101, 202},
				AttentionMask:         []int{1, 1},
				ScoringMask:           []int{1, 1},
				ExpectedLogits:        [][]float64{{-1.0, 1.0}, {2.0, -2.0}},
				ExpectedProbabilities: [][]float64{softmax([]float64{-1.0, 1.0}), softmax([]float64{2.0, -2.0})},
				ExpectedLabels:        []string{"B-PER", "O"},
			},
		},
	}
	candidate := TokenClassificationCandidate{
		Name:        "candidate",
		Source:      "test",
		ModelID:     "example/model",
		Task:        "token-classification",
		Artifact:    "dist/model",
		GeneratedAt: "2026-03-26T00:00:00Z",
		Labels:      []string{"O", "B-PER"},
		Cases: []TokenClassificationCandidateCase{
			{
				ID:                    "case-a",
				Text:                  "John arrived",
				Tokens:                []string{"john", "arrived"},
				InputIDs:              []int{101, 202},
				AttentionMask:         []int{1, 1},
				ScoringMask:           []int{1, 1},
				ObservedLogits:        [][]float64{{-1.0, 1.0}, {2.0, -2.0}},
				ObservedProbabilities: [][]float64{softmax([]float64{-1.0, 1.0}), softmax([]float64{2.0, -2.0})},
				ObservedLabels:        []string{"B-PER", "O"},
			},
		},
	}

	referencePath := writeTempJSON(t, "reference-token.json", reference)
	candidatePath := writeTempJSON(t, "candidate-token.json", candidate)

	report, err := CompareTransformersTokenClassification(referencePath, candidatePath, 1e-9)
	if err != nil {
		t.Fatalf("CompareTransformersTokenClassification() error = %v", err)
	}

	if !report.Passed() {
		t.Fatalf("expected comparison to pass, got report:\n%s", report.String())
	}
}

func TestCompareTransformersTokenClassificationIgnoresConstantLogitShift(t *testing.T) {
	t.Parallel()

	reference := TransformersTokenClassificationReference{
		Name:        "synthetic token reference",
		Source:      "test",
		ModelID:     "example/model",
		Task:        "token-classification",
		GeneratedAt: "2026-03-26T00:00:00Z",
		Labels:      []string{"O", "B-PER"},
		Cases: []TransformersTokenClassificationReferenceCase{
			{
				ID:                    "case-a",
				Text:                  "John arrived",
				Tokens:                []string{"john", "arrived"},
				InputIDs:              []int{101, 202},
				AttentionMask:         []int{1, 1},
				ScoringMask:           []int{1, 1},
				ExpectedLogits:        [][]float64{{-1.0, 1.0}, {2.0, -2.0}},
				ExpectedProbabilities: [][]float64{softmax([]float64{-1.0, 1.0}), softmax([]float64{2.0, -2.0})},
				ExpectedLabels:        []string{"B-PER", "O"},
			},
		},
	}
	candidate := TokenClassificationCandidate{
		Name:        "candidate",
		Source:      "test",
		ModelID:     "example/model",
		Task:        "token-classification",
		Artifact:    "dist/model",
		GeneratedAt: "2026-03-26T00:00:00Z",
		Labels:      []string{"O", "B-PER"},
		Cases: []TokenClassificationCandidateCase{
			{
				ID:                    "case-a",
				Text:                  "John arrived",
				Tokens:                []string{"john", "arrived"},
				InputIDs:              []int{101, 202},
				AttentionMask:         []int{1, 1},
				ScoringMask:           []int{1, 1},
				ObservedLogits:        [][]float64{{4.0, 6.0}, {-1.0, -5.0}},
				ObservedProbabilities: [][]float64{softmax([]float64{4.0, 6.0}), softmax([]float64{-1.0, -5.0})},
				ObservedLabels:        []string{"B-PER", "O"},
			},
		},
	}

	referencePath := writeTempJSON(t, "reference-token-shift.json", reference)
	candidatePath := writeTempJSON(t, "candidate-token-shift.json", candidate)

	report, err := CompareTransformersTokenClassification(referencePath, candidatePath, 1e-9)
	if err != nil {
		t.Fatalf("CompareTransformersTokenClassification() error = %v", err)
	}

	if !report.Passed() {
		t.Fatalf("expected comparison with constant logit shift to pass, got report:\n%s", report.String())
	}
}

func TestBuildTokenClassificationCandidate(t *testing.T) {
	t.Parallel()

	reference := TransformersTokenClassificationReference{
		Name:        "synthetic token reference",
		Source:      "test",
		ModelID:     "example/model",
		Task:        "token-classification",
		GeneratedAt: "2026-03-26T00:00:00Z",
		Labels:      []string{"O", "B-PER"},
		Cases: []TransformersTokenClassificationReferenceCase{
			{
				ID:            "case-a",
				Text:          "John arrived",
				Tokens:        []string{"john", "arrived"},
				InputIDs:      []int{101, 202},
				AttentionMask: []int{1, 1},
				ScoringMask:   []int{1, 1},
			},
		},
	}

	predictor := fakeTokenClassificationPredictor{
		logits: [][][]float64{
			{
				{-1.0, 1.0},
				{2.0, -2.0},
			},
		},
		labels:  []string{"O", "B-PER"},
		modelID: "example/model",
	}

	candidate, err := BuildTokenClassificationCandidate(reference, predictor, "dist/model", "infergo-test")
	if err != nil {
		t.Fatalf("BuildTokenClassificationCandidate() error = %v", err)
	}

	if got := candidate.Cases[0].ObservedLabels[0]; got != "B-PER" {
		t.Fatalf("unexpected first observed label %q", got)
	}
}

func TestRunBionetTokenClassificationBundle(t *testing.T) {
	t.Parallel()

	manifest, err := modelpacks.LoadTokenClassificationManifest(tokenClassificationPackManifestPath)
	if err != nil {
		t.Fatalf("LoadTokenClassificationManifest() error = %v", err)
	}

	for _, pack := range manifest.Packs {
		referencePath := "../../" + pack.ReferencePath
		bundleDir := "../../" + pack.NativeBundleDir
		candidatePath := filepath.Join(t.TempDir(), filepath.Base(bundleDir)+".json")

		candidate, err := RunBionetTokenClassificationBundle(referencePath, bundleDir)
		if err != nil {
			t.Fatalf("RunBionetTokenClassificationBundle(%q) error = %v", bundleDir, err)
		}

		if err := SaveTokenClassificationCandidate(candidate, candidatePath); err != nil {
			t.Fatalf("SaveTokenClassificationCandidate(%q) error = %v", bundleDir, err)
		}

		report, err := CompareTransformersTokenClassification(referencePath, candidatePath, tokenClassificationParityTolerance)
		if err != nil {
			t.Fatalf("CompareTransformersTokenClassification(%q) error = %v", bundleDir, err)
		}

		if !report.Passed() {
			t.Fatalf("expected token-classification comparison for %q to pass, got report:\n%s", bundleDir, report.String())
		}
	}
}
