package parity

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"time"

	"github.com/pergamon-labs/infergo/backends/bionet"
	"github.com/pergamon-labs/infergo/backends/torchscript"
)

// TextClassificationPredictor captures the narrow batch prediction contract used
// by parity-backed text classification comparisons.
type TextClassificationPredictor interface {
	PredictBatch(inputIDs, attentionMasks [][]int64) ([][]float64, error)
	Labels() []string
	ModelID() string
}

// TextClassificationCandidate stores the outputs produced by a local artifact
// over the public reference input set.
type TextClassificationCandidate struct {
	Name        string                            `json:"name"`
	Source      string                            `json:"source"`
	ModelID     string                            `json:"model_id"`
	Task        string                            `json:"task"`
	Artifact    string                            `json:"artifact"`
	GeneratedAt string                            `json:"generated_at"`
	Labels      []string                          `json:"labels"`
	Cases       []TextClassificationCandidateCase `json:"cases"`
}

// TextClassificationCandidateCase stores the local-run outputs for one text
// example.
type TextClassificationCandidateCase struct {
	ID                    string    `json:"id"`
	Text                  string    `json:"text"`
	TextPair              string    `json:"text_pair,omitempty"`
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

// LoadTextClassificationCandidate loads a local candidate JSON file.
func LoadTextClassificationCandidate(path string) (TextClassificationCandidate, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return TextClassificationCandidate{}, fmt.Errorf("read text classification candidate: %w", err)
	}

	var candidate TextClassificationCandidate
	if err := json.Unmarshal(raw, &candidate); err != nil {
		return TextClassificationCandidate{}, fmt.Errorf("decode text classification candidate: %w", err)
	}

	if candidate.Name == "" {
		return TextClassificationCandidate{}, fmt.Errorf("decode text classification candidate: missing name")
	}

	if candidate.ModelID == "" {
		return TextClassificationCandidate{}, fmt.Errorf("decode text classification candidate: missing model id")
	}

	if len(candidate.Cases) == 0 {
		return TextClassificationCandidate{}, fmt.Errorf("decode text classification candidate: no cases defined")
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

	candidate, err := LoadTextClassificationCandidate(candidatePath)
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
	seenCandidateIDs := make(map[string]struct{}, len(candidate.Cases))

	for _, candidateCase := range candidate.Cases {
		referenceCase, ok := referenceByID[candidateCase.ID]
		if !ok {
			return TransformersTextClassificationComparisonReport{}, fmt.Errorf("compare transformers text classification: candidate case %q missing in reference", candidateCase.ID)
		}
		if _, exists := seenCandidateIDs[candidateCase.ID]; exists {
			return TransformersTextClassificationComparisonReport{}, fmt.Errorf("compare transformers text classification: duplicate candidate case %q", candidateCase.ID)
		}
		seenCandidateIDs[candidateCase.ID] = struct{}{}

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

	for _, referenceCase := range reference.Cases {
		if _, ok := seenCandidateIDs[referenceCase.ID]; !ok {
			return TransformersTextClassificationComparisonReport{}, fmt.Errorf("compare transformers text classification: reference case %q missing in candidate", referenceCase.ID)
		}
	}

	return report, nil
}

// RunTorchScriptTextClassificationBundle loads a TorchScript bundle through the
// native InferGo backend path and produces the same candidate JSON shape used by
// file-based comparisons.
func RunTorchScriptTextClassificationBundle(referencePath, bundleDir string) (TextClassificationCandidate, error) {
	reference, err := LoadTransformersTextClassificationReference(referencePath)
	if err != nil {
		return TextClassificationCandidate{}, err
	}

	model, err := torchscript.LoadBundle(bundleDir)
	if err != nil {
		return TextClassificationCandidate{}, fmt.Errorf("run torchscript bundle: %w", err)
	}
	defer model.Close()

	return BuildTextClassificationCandidate(reference, model, bundleDir, "infergo-torchscript")
}

// RunBionetTextClassificationBundle loads an InferGo-native BIOnet bundle and
// produces the same candidate JSON shape used by file-based comparisons.
func RunBionetTextClassificationBundle(referencePath, bundleDir string) (TextClassificationCandidate, error) {
	reference, err := LoadTransformersTextClassificationReference(referencePath)
	if err != nil {
		return TextClassificationCandidate{}, err
	}

	model, err := bionet.LoadTextClassificationBundle(bundleDir)
	if err != nil {
		return TextClassificationCandidate{}, fmt.Errorf("run bionet text classification bundle: %w", err)
	}
	defer model.Close()

	return BuildTextClassificationCandidate(reference, model, bundleDir, "infergo-native")
}

// BuildTextClassificationCandidate builds a candidate payload using any
// compatible text-classification predictor.
func BuildTextClassificationCandidate(reference TransformersTextClassificationReference, predictor TextClassificationPredictor, artifact, source string) (TextClassificationCandidate, error) {
	inputIDs := make([][]int64, len(reference.Cases))
	attentionMasks := make([][]int64, len(reference.Cases))

	for i, item := range reference.Cases {
		inputIDs[i] = intsToInt64(item.InputIDs)
		attentionMasks[i] = intsToInt64(item.AttentionMask)
	}

	logitsBatch, err := predictor.PredictBatch(inputIDs, attentionMasks)
	if err != nil {
		return TextClassificationCandidate{}, fmt.Errorf("build text classification candidate: %w", err)
	}

	if len(logitsBatch) != len(reference.Cases) {
		return TextClassificationCandidate{}, fmt.Errorf("build text classification candidate: output batch size mismatch (%d != %d)", len(logitsBatch), len(reference.Cases))
	}

	candidate := TextClassificationCandidate{
		Name:        fmt.Sprintf("%s native candidate", reference.Name),
		Source:      source,
		ModelID:     predictor.ModelID(),
		Task:        reference.Task,
		Artifact:    artifact,
		GeneratedAt: time.Now().UTC().Format(time.RFC3339),
		Labels:      predictor.Labels(),
		Cases:       make([]TextClassificationCandidateCase, 0, len(reference.Cases)),
	}

	for i, item := range reference.Cases {
		logits := logitsBatch[i]
		probabilities := softmax(logits)
		labelIdx := argMax(probabilities)

		candidate.Cases = append(candidate.Cases, TextClassificationCandidateCase{
			ID:                    item.ID,
			Text:                  item.Text,
			TextPair:              item.TextPair,
			InputIDs:              append([]int(nil), item.InputIDs...),
			AttentionMask:         append([]int(nil), item.AttentionMask...),
			ObservedLogits:        logits,
			ObservedProbabilities: probabilities,
			ObservedLabel:         labelAt(candidate.Labels, labelIdx),
		})
	}

	return candidate, nil
}

// SaveTextClassificationCandidate writes a candidate payload to disk.
func SaveTextClassificationCandidate(candidate TextClassificationCandidate, path string) error {
	raw, err := json.MarshalIndent(candidate, "", "  ")
	if err != nil {
		return fmt.Errorf("save text classification candidate: %w", err)
	}

	if err := os.WriteFile(path, append(raw, '\n'), 0o644); err != nil {
		return fmt.Errorf("save text classification candidate: %w", err)
	}

	return nil
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

func centeredSliceMaxAbsDiff(reference, observed []float64) float64 {
	if len(reference) == 0 || len(observed) == 0 {
		return 0
	}

	referenceMean := 0.0
	observedMean := 0.0
	for i := range reference {
		referenceMean += reference[i]
		observedMean += observed[i]
	}
	referenceMean /= float64(len(reference))
	observedMean /= float64(len(observed))

	maxDiff := 0.0
	for i := range reference {
		diff := math.Abs((reference[i] - referenceMean) - (observed[i] - observedMean))
		if diff > maxDiff {
			maxDiff = diff
		}
	}

	return maxDiff
}

func padIntSlice(values []int, targetLen int, padValue int) []int64 {
	output := make([]int64, targetLen)
	for i := 0; i < targetLen; i++ {
		if i < len(values) {
			output[i] = int64(values[i])
			continue
		}

		output[i] = int64(padValue)
	}

	return output
}

func intsToInt64(values []int) []int64 {
	output := make([]int64, len(values))
	for i, value := range values {
		output[i] = int64(value)
	}

	return output
}

func softmax(logits []float64) []float64 {
	if len(logits) == 0 {
		return nil
	}

	maxValue := logits[0]
	for _, value := range logits[1:] {
		if value > maxValue {
			maxValue = value
		}
	}

	expValues := make([]float64, len(logits))
	var sum float64
	for i, value := range logits {
		expValues[i] = math.Exp(value - maxValue)
		sum += expValues[i]
	}

	for i := range expValues {
		expValues[i] /= sum
	}

	return expValues
}
