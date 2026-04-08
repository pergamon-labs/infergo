package parity

import (
	"encoding/json"
	"fmt"
	"math"
	"os"

	"github.com/pergamon-labs/infergo/backends/torchscript"
)

// EntityResolutionFixture stores a fixed batch of engineered vectors and
// expected scores captured from the current internal ER runtime.
type EntityResolutionFixture struct {
	Name                 string                         `json:"name"`
	Source               string                         `json:"source"`
	ModelID              string                         `json:"model_id"`
	Task                 string                         `json:"task"`
	Family               string                         `json:"family"`
	Backend              string                         `json:"backend"`
	ProfileKind          string                         `json:"profile_kind"`
	GeneratedAt          string                         `json:"generated_at"`
	VectorSize           int                            `json:"vector_size"`
	MessageSize          int                            `json:"message_size"`
	InputLayout          string                         `json:"input_layout"`
	MessageStrategy      string                         `json:"message_strategy"`
	MessageProjection    string                         `json:"message_projection"`
	OutputInterpretation string                         `json:"output_interpretation"`
	Batches              []EntityResolutionFixtureBatch `json:"batches"`
}

// EntityResolutionFixtureBatch stores one scored batch. The shared message
// vector is preserved because ER scores depend on the whole batch.
type EntityResolutionFixtureBatch struct {
	ID      string                      `json:"id"`
	Message []float64                   `json:"message"`
	Cases   []EntityResolutionCaseInput `json:"cases"`
}

// EntityResolutionCaseInput stores one engineered vector plus the expected
// score captured from the current runtime.
type EntityResolutionCaseInput struct {
	ID            string    `json:"id"`
	LeftName      string    `json:"left_name"`
	RightName     string    `json:"right_name"`
	Vector        []float64 `json:"vector"`
	ExpectedScore float64   `json:"expected_score"`
}

// EntityResolutionPredictorMetadata is the narrow metadata contract needed by
// family-2 parity comparisons.
type EntityResolutionPredictorMetadata struct {
	ModelID              string
	Task                 string
	Family               string
	Backend              string
	ProfileKind          string
	VectorSize           int
	MessageSize          int
	InputLayout          string
	MessageStrategy      string
	MessageProjection    string
	OutputInterpretation string
}

// EntityResolutionPredictor captures the narrow family-2 scoring contract used
// by parity comparisons.
type EntityResolutionPredictor interface {
	Metadata() EntityResolutionPredictorMetadata
	PredictBatch(vectors [][]float64, message []float64) ([]float64, error)
	Close() error
}

// EntityResolutionComparisonReport stores the case-by-case diff between a
// captured family-2 fixture and the InferGo bridge output.
type EntityResolutionComparisonReport struct {
	ReferencePath string
	BundlePath    string
	ModelID       string
	ProfileKind   string
	Tolerance     float64
	BatchResults  []EntityResolutionComparisonBatch
}

// EntityResolutionComparisonBatch stores the comparison result for one scored
// batch.
type EntityResolutionComparisonBatch struct {
	ID              string
	MaxScoreAbsDiff float64
	CaseResults     []EntityResolutionComparisonCase
	Passed          bool
}

// EntityResolutionComparisonCase stores one case comparison result.
type EntityResolutionComparisonCase struct {
	ID           string
	LeftName     string
	RightName    string
	Expected     float64
	Observed     float64
	ScoreAbsDiff float64
	Passed       bool
}

type torchscriptEntityResolutionPredictor struct {
	bundle *torchscript.EntityResolutionBundle
}

func (p torchscriptEntityResolutionPredictor) Metadata() EntityResolutionPredictorMetadata {
	meta := p.bundle.Metadata()
	return EntityResolutionPredictorMetadata{
		ModelID:              meta.ModelID,
		Task:                 meta.Task,
		Family:               meta.Family,
		Backend:              meta.Backend,
		ProfileKind:          meta.ProfileKind,
		VectorSize:           meta.Inputs.VectorSize,
		MessageSize:          meta.Inputs.MessageSize,
		InputLayout:          meta.Inputs.InputLayout,
		MessageStrategy:      meta.Inputs.MessageStrategy,
		MessageProjection:    meta.Inputs.MessageProjection,
		OutputInterpretation: meta.Outputs.Interpretation,
	}
}

func (p torchscriptEntityResolutionPredictor) PredictBatch(vectors [][]float64, message []float64) ([]float64, error) {
	return p.bundle.PredictBatch(vectors, message)
}

func (p torchscriptEntityResolutionPredictor) Close() error {
	return p.bundle.Close()
}

// LoadEntityResolutionFixture loads and validates a family-2 parity fixture.
func LoadEntityResolutionFixture(path string) (EntityResolutionFixture, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return EntityResolutionFixture{}, fmt.Errorf("read entity-resolution fixture: %w", err)
	}

	var fixture EntityResolutionFixture
	if err := json.Unmarshal(raw, &fixture); err != nil {
		return EntityResolutionFixture{}, fmt.Errorf("decode entity-resolution fixture: %w", err)
	}

	if fixture.Name == "" {
		return EntityResolutionFixture{}, fmt.Errorf("decode entity-resolution fixture: missing name")
	}
	if fixture.ModelID == "" {
		return EntityResolutionFixture{}, fmt.Errorf("decode entity-resolution fixture: missing model id")
	}
	if fixture.Task != "entity-resolution-scoring" {
		return EntityResolutionFixture{}, fmt.Errorf("decode entity-resolution fixture: unsupported task %q", fixture.Task)
	}
	if fixture.Family != "numeric-feature-scoring" {
		return EntityResolutionFixture{}, fmt.Errorf("decode entity-resolution fixture: unsupported family %q", fixture.Family)
	}
	if fixture.Backend != "torchscript" {
		return EntityResolutionFixture{}, fmt.Errorf("decode entity-resolution fixture: unsupported backend %q", fixture.Backend)
	}
	if fixture.ProfileKind != "individual" && fixture.ProfileKind != "organization" {
		return EntityResolutionFixture{}, fmt.Errorf("decode entity-resolution fixture: unsupported profile kind %q", fixture.ProfileKind)
	}
	if fixture.VectorSize <= 0 {
		return EntityResolutionFixture{}, fmt.Errorf("decode entity-resolution fixture: invalid vector_size")
	}
	if fixture.MessageSize <= 0 {
		return EntityResolutionFixture{}, fmt.Errorf("decode entity-resolution fixture: invalid message_size")
	}
	if fixture.InputLayout != "stacked_sample_message_channels" {
		return EntityResolutionFixture{}, fmt.Errorf("decode entity-resolution fixture: unsupported input_layout %q", fixture.InputLayout)
	}
	if fixture.MessageStrategy != "caller_supplied_consensus_vector" {
		return EntityResolutionFixture{}, fmt.Errorf("decode entity-resolution fixture: unsupported message_strategy %q", fixture.MessageStrategy)
	}
	if fixture.MessageProjection != "legacy_first_value_broadcast" && fixture.MessageProjection != "full_vector" {
		return EntityResolutionFixture{}, fmt.Errorf("decode entity-resolution fixture: unsupported message_projection %q", fixture.MessageProjection)
	}
	if fixture.OutputInterpretation != "confidence" {
		return EntityResolutionFixture{}, fmt.Errorf("decode entity-resolution fixture: unsupported output_interpretation %q", fixture.OutputInterpretation)
	}
	if len(fixture.Batches) == 0 {
		return EntityResolutionFixture{}, fmt.Errorf("decode entity-resolution fixture: no batches defined")
	}

	for _, batch := range fixture.Batches {
		if batch.ID == "" {
			return EntityResolutionFixture{}, fmt.Errorf("decode entity-resolution fixture: batch missing id")
		}
		if len(batch.Message) != fixture.MessageSize {
			return EntityResolutionFixture{}, fmt.Errorf("decode entity-resolution fixture: batch %q message length mismatch (%d != %d)", batch.ID, len(batch.Message), fixture.MessageSize)
		}
		if len(batch.Cases) == 0 {
			return EntityResolutionFixture{}, fmt.Errorf("decode entity-resolution fixture: batch %q has no cases", batch.ID)
		}
		for _, item := range batch.Cases {
			if item.ID == "" {
				return EntityResolutionFixture{}, fmt.Errorf("decode entity-resolution fixture: batch %q case missing id", batch.ID)
			}
			if len(item.Vector) != fixture.VectorSize {
				return EntityResolutionFixture{}, fmt.Errorf("decode entity-resolution fixture: batch %q case %q vector length mismatch (%d != %d)", batch.ID, item.ID, len(item.Vector), fixture.VectorSize)
			}
			if math.IsNaN(item.ExpectedScore) || math.IsInf(item.ExpectedScore, 0) {
				return EntityResolutionFixture{}, fmt.Errorf("decode entity-resolution fixture: batch %q case %q expected score must be finite", batch.ID, item.ID)
			}
		}
	}

	return fixture, nil
}

// RunTorchScriptEntityResolutionFixture compares a family-2 local fixture
// against a TorchScript bridge bundle.
func RunTorchScriptEntityResolutionFixture(fixturePath, bundleDir string, tolerance float64) (EntityResolutionComparisonReport, error) {
	bundle, err := torchscript.LoadEntityResolutionBundle(bundleDir)
	if err != nil {
		return EntityResolutionComparisonReport{}, fmt.Errorf("run entity-resolution fixture: %w", err)
	}
	defer bundle.Close()

	return CompareEntityResolutionFixture(fixturePath, bundleDir, tolerance, torchscriptEntityResolutionPredictor{
		bundle: bundle,
	})
}

// CompareEntityResolutionFixture compares a captured family-2 fixture against
// any compatible entity-resolution predictor.
func CompareEntityResolutionFixture(fixturePath, bundlePath string, tolerance float64, predictor EntityResolutionPredictor) (EntityResolutionComparisonReport, error) {
	if tolerance <= 0 {
		return EntityResolutionComparisonReport{}, fmt.Errorf("compare entity-resolution fixture: tolerance must be greater than zero")
	}

	fixture, err := LoadEntityResolutionFixture(fixturePath)
	if err != nil {
		return EntityResolutionComparisonReport{}, err
	}

	meta := predictor.Metadata()
	if meta.ModelID != fixture.ModelID {
		return EntityResolutionComparisonReport{}, fmt.Errorf("compare entity-resolution fixture: model id mismatch (%s != %s)", meta.ModelID, fixture.ModelID)
	}
	if meta.Task != fixture.Task {
		return EntityResolutionComparisonReport{}, fmt.Errorf("compare entity-resolution fixture: task mismatch (%s != %s)", meta.Task, fixture.Task)
	}
	if meta.Family != fixture.Family {
		return EntityResolutionComparisonReport{}, fmt.Errorf("compare entity-resolution fixture: family mismatch (%s != %s)", meta.Family, fixture.Family)
	}
	if meta.Backend != fixture.Backend {
		return EntityResolutionComparisonReport{}, fmt.Errorf("compare entity-resolution fixture: backend mismatch (%s != %s)", meta.Backend, fixture.Backend)
	}
	if meta.ProfileKind != fixture.ProfileKind {
		return EntityResolutionComparisonReport{}, fmt.Errorf("compare entity-resolution fixture: profile kind mismatch (%s != %s)", meta.ProfileKind, fixture.ProfileKind)
	}
	if meta.VectorSize != fixture.VectorSize {
		return EntityResolutionComparisonReport{}, fmt.Errorf("compare entity-resolution fixture: vector size mismatch (%d != %d)", meta.VectorSize, fixture.VectorSize)
	}
	if meta.MessageSize != fixture.MessageSize {
		return EntityResolutionComparisonReport{}, fmt.Errorf("compare entity-resolution fixture: message size mismatch (%d != %d)", meta.MessageSize, fixture.MessageSize)
	}
	if meta.InputLayout != fixture.InputLayout {
		return EntityResolutionComparisonReport{}, fmt.Errorf("compare entity-resolution fixture: input layout mismatch (%s != %s)", meta.InputLayout, fixture.InputLayout)
	}
	if meta.MessageStrategy != fixture.MessageStrategy {
		return EntityResolutionComparisonReport{}, fmt.Errorf("compare entity-resolution fixture: message strategy mismatch (%s != %s)", meta.MessageStrategy, fixture.MessageStrategy)
	}
	if meta.MessageProjection != fixture.MessageProjection {
		return EntityResolutionComparisonReport{}, fmt.Errorf("compare entity-resolution fixture: message projection mismatch (%s != %s)", meta.MessageProjection, fixture.MessageProjection)
	}
	if meta.OutputInterpretation != fixture.OutputInterpretation {
		return EntityResolutionComparisonReport{}, fmt.Errorf("compare entity-resolution fixture: output interpretation mismatch (%s != %s)", meta.OutputInterpretation, fixture.OutputInterpretation)
	}

	report := EntityResolutionComparisonReport{
		ReferencePath: fixturePath,
		BundlePath:    bundlePath,
		ModelID:       fixture.ModelID,
		ProfileKind:   fixture.ProfileKind,
		Tolerance:     tolerance,
		BatchResults:  make([]EntityResolutionComparisonBatch, 0, len(fixture.Batches)),
	}

	for _, batch := range fixture.Batches {
		vectors := make([][]float64, len(batch.Cases))
		for i, item := range batch.Cases {
			vectors[i] = item.Vector
		}

		observed, err := predictor.PredictBatch(vectors, batch.Message)
		if err != nil {
			return EntityResolutionComparisonReport{}, fmt.Errorf("compare entity-resolution fixture: batch %q: %w", batch.ID, err)
		}
		if len(observed) != len(batch.Cases) {
			return EntityResolutionComparisonReport{}, fmt.Errorf("compare entity-resolution fixture: batch %q score count mismatch (%d != %d)", batch.ID, len(observed), len(batch.Cases))
		}

		batchResult := EntityResolutionComparisonBatch{
			ID:          batch.ID,
			CaseResults: make([]EntityResolutionComparisonCase, 0, len(batch.Cases)),
			Passed:      true,
		}

		for i, item := range batch.Cases {
			diff := math.Abs(item.ExpectedScore - observed[i])
			if diff > batchResult.MaxScoreAbsDiff {
				batchResult.MaxScoreAbsDiff = diff
			}
			caseResult := EntityResolutionComparisonCase{
				ID:           item.ID,
				LeftName:     item.LeftName,
				RightName:    item.RightName,
				Expected:     item.ExpectedScore,
				Observed:     observed[i],
				ScoreAbsDiff: diff,
				Passed:       diff <= tolerance,
			}
			if !caseResult.Passed {
				batchResult.Passed = false
			}
			batchResult.CaseResults = append(batchResult.CaseResults, caseResult)
		}

		report.BatchResults = append(report.BatchResults, batchResult)
	}

	return report, nil
}

// Passed returns true when every compared ER case is within tolerance.
func (r EntityResolutionComparisonReport) Passed() bool {
	for _, batch := range r.BatchResults {
		if !batch.Passed {
			return false
		}
	}
	return true
}

// String renders a compact human-readable summary for family-2 parity.
func (r EntityResolutionComparisonReport) String() string {
	summary := fmt.Sprintf(
		"model=%s profile_kind=%s batches=%d tolerance=%.6g status=%s\n",
		r.ModelID,
		r.ProfileKind,
		len(r.BatchResults),
		r.Tolerance,
		passFail(r.Passed()),
	)

	for _, batch := range r.BatchResults {
		summary += fmt.Sprintf(
			"- %s: cases=%d max_score_abs_diff=%.6g status=%s\n",
			batch.ID,
			len(batch.CaseResults),
			batch.MaxScoreAbsDiff,
			passFail(batch.Passed),
		)
		for _, item := range batch.CaseResults {
			summary += fmt.Sprintf(
				"  - %s: left=%q right=%q expected=%.6g observed=%.6g score_abs_diff=%.6g status=%s\n",
				item.ID,
				item.LeftName,
				item.RightName,
				item.Expected,
				item.Observed,
				item.ScoreAbsDiff,
				passFail(item.Passed),
			)
		}
	}

	return summary
}
