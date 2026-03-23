package parity

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
)

// TorchScriptTextClassificationCandidate stores the outputs produced by a
// local TorchScript artifact over the public reference input set.
type TorchScriptTextClassificationCandidate struct {
	Name        string                                       `json:"name"`
	Source      string                                       `json:"source"`
	ModelID     string                                       `json:"model_id"`
	Task        string                                       `json:"task"`
	Artifact    string                                       `json:"artifact"`
	GeneratedAt string                                       `json:"generated_at"`
	Labels      []string                                     `json:"labels"`
	Cases       []TorchScriptTextClassificationCandidateCase `json:"cases"`
}

// TorchScriptTextClassificationCandidateCase stores the local-run outputs for
// one text example.
type TorchScriptTextClassificationCandidateCase struct {
	ID                    string    `json:"id"`
	Text                  string    `json:"text"`
	InputIDs              []int     `json:"input_ids"`
	AttentionMask         []int     `json:"attention_mask"`
	ObservedLogits        []float64 `json:"observed_logits"`
	ObservedProbabilities []float64 `json:"observed_probabilities"`
	ObservedLabel         string    `json:"observed_label"`
}

// TransformersTextClassificationComparisonReport stores the case-by-case diff
// between a Transformers reference file and a locally-run candidate file.
type TransformersTextClassificationComparisonReport struct {
	ReferencePath string
	CandidatePath string
	ModelID       string
	Tolerance     float64
	CaseResults   []TransformersTextClassificationComparisonCase
}

// TransformersTextClassificationComparisonCase captures one case comparison.
type TransformersTextClassificationComparisonCase struct {
	ID              string
	ExpectedLabel   string
	ObservedLabel   string
	MaxLogitAbsDiff float64
	MaxProbAbsDiff  float64
	LabelMatch      bool
	Passed          bool
}

// LoadTorchScriptTextClassificationCandidate loads a TorchScript run JSON file.
func LoadTorchScriptTextClassificationCandidate(path string) (TorchScriptTextClassificationCandidate, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return TorchScriptTextClassificationCandidate{}, fmt.Errorf("read torchscript candidate: %w", err)
	}

	var candidate TorchScriptTextClassificationCandidate
	if err := json.Unmarshal(raw, &candidate); err != nil {
		return TorchScriptTextClassificationCandidate{}, fmt.Errorf("decode torchscript candidate: %w", err)
	}

	if candidate.Name == "" {
		return TorchScriptTextClassificationCandidate{}, fmt.Errorf("decode torchscript candidate: missing name")
	}

	if candidate.ModelID == "" {
		return TorchScriptTextClassificationCandidate{}, fmt.Errorf("decode torchscript candidate: missing model id")
	}

	if len(candidate.Cases) == 0 {
		return TorchScriptTextClassificationCandidate{}, fmt.Errorf("decode torchscript candidate: no cases defined")
	}

	return candidate, nil
}

// CompareTransformersTextClassification compares a generated reference file with
// a locally-run TorchScript candidate file.
func CompareTransformersTextClassification(referencePath, candidatePath string, tolerance float64) (TransformersTextClassificationComparisonReport, error) {
	if tolerance <= 0 {
		return TransformersTextClassificationComparisonReport{}, fmt.Errorf("compare transformers text classification: tolerance must be greater than zero")
	}

	reference, err := LoadTransformersTextClassificationReference(referencePath)
	if err != nil {
		return TransformersTextClassificationComparisonReport{}, err
	}

	candidate, err := LoadTorchScriptTextClassificationCandidate(candidatePath)
	if err != nil {
		return TransformersTextClassificationComparisonReport{}, err
	}

	if reference.ModelID != candidate.ModelID {
		return TransformersTextClassificationComparisonReport{}, fmt.Errorf("compare transformers text classification: model id mismatch (%s != %s)", reference.ModelID, candidate.ModelID)
	}

	referenceByID := make(map[string]TransformersTextClassificationReferenceCase, len(reference.Cases))
	for _, item := range reference.Cases {
		referenceByID[item.ID] = item
	}

	report := TransformersTextClassificationComparisonReport{
		ReferencePath: referencePath,
		CandidatePath: candidatePath,
		ModelID:       reference.ModelID,
		Tolerance:     tolerance,
	}

	for _, candidateCase := range candidate.Cases {
		referenceCase, ok := referenceByID[candidateCase.ID]
		if !ok {
			return TransformersTextClassificationComparisonReport{}, fmt.Errorf("compare transformers text classification: candidate case %q missing in reference", candidateCase.ID)
		}

		if len(referenceCase.ExpectedLogits) != len(candidateCase.ObservedLogits) {
			return TransformersTextClassificationComparisonReport{}, fmt.Errorf("compare transformers text classification: logits length mismatch for %q", candidateCase.ID)
		}

		if len(referenceCase.ExpectedProbabilities) != len(candidateCase.ObservedProbabilities) {
			return TransformersTextClassificationComparisonReport{}, fmt.Errorf("compare transformers text classification: probability length mismatch for %q", candidateCase.ID)
		}

		maxLogitDiff := sliceMaxAbsDiff(referenceCase.ExpectedLogits, candidateCase.ObservedLogits)
		maxProbDiff := sliceMaxAbsDiff(referenceCase.ExpectedProbabilities, candidateCase.ObservedProbabilities)
		labelMatch := referenceCase.ExpectedLabel == candidateCase.ObservedLabel

		report.CaseResults = append(report.CaseResults, TransformersTextClassificationComparisonCase{
			ID:              candidateCase.ID,
			ExpectedLabel:   referenceCase.ExpectedLabel,
			ObservedLabel:   candidateCase.ObservedLabel,
			MaxLogitAbsDiff: maxLogitDiff,
			MaxProbAbsDiff:  maxProbDiff,
			LabelMatch:      labelMatch,
			Passed:          labelMatch && maxLogitDiff <= tolerance && maxProbDiff <= tolerance,
		})
	}

	return report, nil
}

// Passed returns true when every comparison case is within tolerance.
func (r TransformersTextClassificationComparisonReport) Passed() bool {
	for _, result := range r.CaseResults {
		if !result.Passed {
			return false
		}
	}

	return true
}

// String renders a compact human-readable comparison summary.
func (r TransformersTextClassificationComparisonReport) String() string {
	summary := fmt.Sprintf(
		"model=%s cases=%d tolerance=%.6g status=%s\n",
		r.ModelID,
		len(r.CaseResults),
		r.Tolerance,
		passFail(r.Passed()),
	)

	for _, result := range r.CaseResults {
		summary += fmt.Sprintf(
			"- %s: expected=%s observed=%s max_logit_abs_diff=%.6g max_prob_abs_diff=%.6g status=%s\n",
			result.ID,
			result.ExpectedLabel,
			result.ObservedLabel,
			result.MaxLogitAbsDiff,
			result.MaxProbAbsDiff,
			passFail(result.Passed),
		)
	}

	return summary
}

func sliceMaxAbsDiff(reference, observed []float64) float64 {
	maxDiff := 0.0
	for i := range reference {
		diff := math.Abs(reference[i] - observed[i])
		if diff > maxDiff {
			maxDiff = diff
		}
	}

	return maxDiff
}
