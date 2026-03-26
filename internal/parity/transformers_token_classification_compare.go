package parity

import (
	"encoding/json"
	"fmt"
	"os"
	"time"

	"github.com/pergamon-labs/infergo/backends/bionet"
)

// TokenClassificationPredictor captures the narrow batch prediction contract
// used by parity-backed token-classification comparisons.
type TokenClassificationPredictor interface {
	PredictBatch(inputIDs, attentionMasks [][]int64) ([][][]float64, error)
	Labels() []string
	ModelID() string
}

// TokenClassificationCandidate stores the outputs produced by a local artifact
// over the public token-classification reference input set.
type TokenClassificationCandidate struct {
	Name        string                             `json:"name"`
	Source      string                             `json:"source"`
	ModelID     string                             `json:"model_id"`
	Task        string                             `json:"task"`
	Artifact    string                             `json:"artifact"`
	GeneratedAt string                             `json:"generated_at"`
	Labels      []string                           `json:"labels"`
	Cases       []TokenClassificationCandidateCase `json:"cases"`
}

// TokenClassificationCandidateCase stores the local-run outputs for one text
// example.
type TokenClassificationCandidateCase struct {
	ID                    string      `json:"id"`
	Text                  string      `json:"text"`
	Tokens                []string    `json:"tokens"`
	InputIDs              []int       `json:"input_ids"`
	AttentionMask         []int       `json:"attention_mask"`
	ScoringMask           []int       `json:"scoring_mask"`
	ObservedLogits        [][]float64 `json:"observed_logits"`
	ObservedProbabilities [][]float64 `json:"observed_probabilities"`
	ObservedLabels        []string    `json:"observed_labels"`
}

// TransformersTokenClassificationComparisonReport stores the case-by-case diff
// between a Transformers reference file and a locally-run candidate file.
type TransformersTokenClassificationComparisonReport struct {
	ReferencePath string
	CandidatePath string
	ModelID       string
	Tolerance     float64
	CaseResults   []TransformersTokenClassificationComparisonCase
}

// TransformersTokenClassificationComparisonCase captures one case comparison.
type TransformersTokenClassificationComparisonCase struct {
	ID               string
	ScoredTokens     int
	MismatchedTokens int
	MaxLogitAbsDiff  float64
	MaxProbAbsDiff   float64
	Passed           bool
}

// LoadTokenClassificationCandidate loads a local candidate JSON file.
func LoadTokenClassificationCandidate(path string) (TokenClassificationCandidate, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return TokenClassificationCandidate{}, fmt.Errorf("read token classification candidate: %w", err)
	}

	var candidate TokenClassificationCandidate
	if err := json.Unmarshal(raw, &candidate); err != nil {
		return TokenClassificationCandidate{}, fmt.Errorf("decode token classification candidate: %w", err)
	}

	if candidate.Name == "" {
		return TokenClassificationCandidate{}, fmt.Errorf("decode token classification candidate: missing name")
	}

	if candidate.ModelID == "" {
		return TokenClassificationCandidate{}, fmt.Errorf("decode token classification candidate: missing model id")
	}

	if len(candidate.Cases) == 0 {
		return TokenClassificationCandidate{}, fmt.Errorf("decode token classification candidate: no cases defined")
	}

	return candidate, nil
}

// CompareTransformersTokenClassification compares a generated token
// classification reference file with a locally-run candidate file.
func CompareTransformersTokenClassification(referencePath, candidatePath string, tolerance float64) (TransformersTokenClassificationComparisonReport, error) {
	if tolerance <= 0 {
		return TransformersTokenClassificationComparisonReport{}, fmt.Errorf("compare transformers token classification: tolerance must be greater than zero")
	}

	reference, err := LoadTransformersTokenClassificationReference(referencePath)
	if err != nil {
		return TransformersTokenClassificationComparisonReport{}, err
	}

	candidate, err := LoadTokenClassificationCandidate(candidatePath)
	if err != nil {
		return TransformersTokenClassificationComparisonReport{}, err
	}

	if reference.ModelID != candidate.ModelID {
		return TransformersTokenClassificationComparisonReport{}, fmt.Errorf("compare transformers token classification: model id mismatch (%s != %s)", reference.ModelID, candidate.ModelID)
	}

	referenceByID := make(map[string]TransformersTokenClassificationReferenceCase, len(reference.Cases))
	for _, item := range reference.Cases {
		referenceByID[item.ID] = item
	}

	report := TransformersTokenClassificationComparisonReport{
		ReferencePath: referencePath,
		CandidatePath: candidatePath,
		ModelID:       reference.ModelID,
		Tolerance:     tolerance,
	}
	seenCandidateIDs := make(map[string]struct{}, len(candidate.Cases))

	for _, candidateCase := range candidate.Cases {
		referenceCase, ok := referenceByID[candidateCase.ID]
		if !ok {
			return TransformersTokenClassificationComparisonReport{}, fmt.Errorf("compare transformers token classification: candidate case %q missing in reference", candidateCase.ID)
		}
		if _, exists := seenCandidateIDs[candidateCase.ID]; exists {
			return TransformersTokenClassificationComparisonReport{}, fmt.Errorf("compare transformers token classification: duplicate candidate case %q", candidateCase.ID)
		}
		seenCandidateIDs[candidateCase.ID] = struct{}{}

		if err := validateTokenCaseLengths(referenceCase, candidateCase); err != nil {
			return TransformersTokenClassificationComparisonReport{}, fmt.Errorf("compare transformers token classification: %q: %w", candidateCase.ID, err)
		}

		maxLogitDiff := 0.0
		maxProbDiff := 0.0
		mismatchedTokens := 0
		scoredTokens := 0

		for pos, shouldScore := range referenceCase.ScoringMask {
			if shouldScore == 0 {
				continue
			}
			scoredTokens++

			logitDiff := sliceMaxAbsDiff(referenceCase.ExpectedLogits[pos], candidateCase.ObservedLogits[pos])
			probDiff := sliceMaxAbsDiff(referenceCase.ExpectedProbabilities[pos], candidateCase.ObservedProbabilities[pos])
			if logitDiff > maxLogitDiff {
				maxLogitDiff = logitDiff
			}
			if probDiff > maxProbDiff {
				maxProbDiff = probDiff
			}
			if referenceCase.ExpectedLabels[pos] != candidateCase.ObservedLabels[pos] {
				mismatchedTokens++
			}
		}

		report.CaseResults = append(report.CaseResults, TransformersTokenClassificationComparisonCase{
			ID:               candidateCase.ID,
			ScoredTokens:     scoredTokens,
			MismatchedTokens: mismatchedTokens,
			MaxLogitAbsDiff:  maxLogitDiff,
			MaxProbAbsDiff:   maxProbDiff,
			Passed:           mismatchedTokens == 0 && maxLogitDiff <= tolerance && maxProbDiff <= tolerance,
		})
	}

	for _, referenceCase := range reference.Cases {
		if _, ok := seenCandidateIDs[referenceCase.ID]; !ok {
			return TransformersTokenClassificationComparisonReport{}, fmt.Errorf("compare transformers token classification: reference case %q missing in candidate", referenceCase.ID)
		}
	}

	return report, nil
}

// RunBionetTokenClassificationBundle loads an InferGo-native BIOnet token
// classification bundle and produces the same candidate JSON shape used by
// file-based comparisons.
func RunBionetTokenClassificationBundle(referencePath, bundleDir string) (TokenClassificationCandidate, error) {
	reference, err := LoadTransformersTokenClassificationReference(referencePath)
	if err != nil {
		return TokenClassificationCandidate{}, err
	}

	model, err := bionet.LoadTokenClassificationBundle(bundleDir)
	if err != nil {
		return TokenClassificationCandidate{}, fmt.Errorf("run bionet token classification bundle: %w", err)
	}
	defer model.Close()

	return BuildTokenClassificationCandidate(reference, model, bundleDir, "infergo-native")
}

// BuildTokenClassificationCandidate builds a candidate payload using any
// compatible token-classification predictor.
func BuildTokenClassificationCandidate(reference TransformersTokenClassificationReference, predictor TokenClassificationPredictor, artifact, source string) (TokenClassificationCandidate, error) {
	inputIDs := make([][]int64, len(reference.Cases))
	attentionMasks := make([][]int64, len(reference.Cases))

	for i, item := range reference.Cases {
		inputIDs[i] = intsToInt64(item.InputIDs)
		attentionMasks[i] = intsToInt64(item.AttentionMask)
	}

	logitsBatch, err := predictor.PredictBatch(inputIDs, attentionMasks)
	if err != nil {
		return TokenClassificationCandidate{}, fmt.Errorf("build token classification candidate: %w", err)
	}

	if len(logitsBatch) != len(reference.Cases) {
		return TokenClassificationCandidate{}, fmt.Errorf("build token classification candidate: output batch size mismatch (%d != %d)", len(logitsBatch), len(reference.Cases))
	}

	candidate := TokenClassificationCandidate{
		Name:        fmt.Sprintf("%s native candidate", reference.Name),
		Source:      source,
		ModelID:     predictor.ModelID(),
		Task:        reference.Task,
		Artifact:    artifact,
		GeneratedAt: time.Now().UTC().Format(time.RFC3339),
		Labels:      predictor.Labels(),
		Cases:       make([]TokenClassificationCandidateCase, 0, len(reference.Cases)),
	}

	for i, item := range reference.Cases {
		caseLogits := logitsBatch[i]
		if len(caseLogits) != len(item.InputIDs) {
			return TokenClassificationCandidate{}, fmt.Errorf("build token classification candidate: case %q output length mismatch (%d != %d)", item.ID, len(caseLogits), len(item.InputIDs))
		}

		observedProbabilities := make([][]float64, len(caseLogits))
		observedLabels := make([]string, len(caseLogits))
		for pos, logits := range caseLogits {
			probabilities := softmax(logits)
			labelIdx := argMax(probabilities)
			observedProbabilities[pos] = probabilities
			observedLabels[pos] = labelAt(candidate.Labels, labelIdx)
		}

		candidate.Cases = append(candidate.Cases, TokenClassificationCandidateCase{
			ID:                    item.ID,
			Text:                  item.Text,
			Tokens:                append([]string(nil), item.Tokens...),
			InputIDs:              append([]int(nil), item.InputIDs...),
			AttentionMask:         append([]int(nil), item.AttentionMask...),
			ScoringMask:           append([]int(nil), item.ScoringMask...),
			ObservedLogits:        cloneFloat2D(caseLogits),
			ObservedProbabilities: cloneFloat2D(observedProbabilities),
			ObservedLabels:        append([]string(nil), observedLabels...),
		})
	}

	return candidate, nil
}

// SaveTokenClassificationCandidate writes a token-classification candidate
// payload to disk.
func SaveTokenClassificationCandidate(candidate TokenClassificationCandidate, path string) error {
	raw, err := json.MarshalIndent(candidate, "", "  ")
	if err != nil {
		return fmt.Errorf("save token classification candidate: %w", err)
	}

	if err := os.WriteFile(path, append(raw, '\n'), 0o644); err != nil {
		return fmt.Errorf("save token classification candidate: %w", err)
	}

	return nil
}

// Passed returns true when every comparison case is within tolerance.
func (r TransformersTokenClassificationComparisonReport) Passed() bool {
	for _, result := range r.CaseResults {
		if !result.Passed {
			return false
		}
	}
	return true
}

// String renders a compact human-readable comparison summary.
func (r TransformersTokenClassificationComparisonReport) String() string {
	summary := fmt.Sprintf(
		"model=%s cases=%d tolerance=%.6g status=%s\n",
		r.ModelID,
		len(r.CaseResults),
		r.Tolerance,
		passFail(r.Passed()),
	)

	for _, result := range r.CaseResults {
		summary += fmt.Sprintf(
			"- %s: scored_tokens=%d mismatched_tokens=%d max_logit_abs_diff=%.6g max_prob_abs_diff=%.6g status=%s\n",
			result.ID,
			result.ScoredTokens,
			result.MismatchedTokens,
			result.MaxLogitAbsDiff,
			result.MaxProbAbsDiff,
			passFail(result.Passed),
		)
	}

	return summary
}

func validateTokenCaseLengths(referenceCase TransformersTokenClassificationReferenceCase, candidateCase TokenClassificationCandidateCase) error {
	expectedLen := len(referenceCase.InputIDs)
	if len(referenceCase.AttentionMask) != expectedLen ||
		len(referenceCase.ScoringMask) != expectedLen ||
		len(referenceCase.Tokens) != expectedLen ||
		len(referenceCase.ExpectedLogits) != expectedLen ||
		len(referenceCase.ExpectedProbabilities) != expectedLen ||
		len(referenceCase.ExpectedLabels) != expectedLen {
		return fmt.Errorf("reference case lengths are inconsistent")
	}

	if len(candidateCase.InputIDs) != expectedLen ||
		len(candidateCase.AttentionMask) != expectedLen ||
		len(candidateCase.ScoringMask) != expectedLen ||
		len(candidateCase.Tokens) != expectedLen ||
		len(candidateCase.ObservedLogits) != expectedLen ||
		len(candidateCase.ObservedProbabilities) != expectedLen ||
		len(candidateCase.ObservedLabels) != expectedLen {
		return fmt.Errorf("candidate case lengths are inconsistent")
	}

	for pos := 0; pos < expectedLen; pos++ {
		if len(referenceCase.ExpectedLogits[pos]) != len(candidateCase.ObservedLogits[pos]) {
			return fmt.Errorf("token %d logits length mismatch", pos)
		}
		if len(referenceCase.ExpectedProbabilities[pos]) != len(candidateCase.ObservedProbabilities[pos]) {
			return fmt.Errorf("token %d probabilities length mismatch", pos)
		}
	}

	return nil
}

func cloneFloat2D(values [][]float64) [][]float64 {
	out := make([][]float64, len(values))
	for i := range values {
		out[i] = append([]float64(nil), values[i]...)
	}
	return out
}
