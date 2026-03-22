package parity

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/minervaai/infergo/backends/bionet"
)

// TextClassificationFixture describes a public-safe parity fixture for a simple
// feature-vector-based text classification flow.
type TextClassificationFixture struct {
	Name      string                   `json:"name"`
	Backend   string                   `json:"backend"`
	Artifact  string                   `json:"artifact"`
	Labels    []string                 `json:"labels"`
	Tolerance float64                  `json:"tolerance"`
	Cases     []TextClassificationCase `json:"cases"`
}

// TextClassificationCase is a single labeled text example in the parity fixture.
type TextClassificationCase struct {
	ID                    string    `json:"id"`
	Text                  string    `json:"text"`
	Features              []float64 `json:"features"`
	ExpectedProbabilities []float64 `json:"expected_probabilities"`
	ExpectedLabel         string    `json:"expected_label"`
}

// TextClassificationCaseResult stores the output comparison for one fixture case.
type TextClassificationCaseResult struct {
	ID             string
	Text           string
	ExpectedLabel  string
	PredictedLabel string
	MaxAbsDiff     float64
	Passed         bool
}

// TextClassificationReport captures the end-to-end result of a parity run.
type TextClassificationReport struct {
	FixturePath string
	FixtureName string
	Backend     string
	Tolerance   float64
	CaseResults []TextClassificationCaseResult
}

// LoadTextClassificationFixture reads a parity fixture from disk.
func LoadTextClassificationFixture(path string) (TextClassificationFixture, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return TextClassificationFixture{}, fmt.Errorf("read parity fixture: %w", err)
	}

	var fixture TextClassificationFixture
	if err := json.Unmarshal(raw, &fixture); err != nil {
		return TextClassificationFixture{}, fmt.Errorf("decode parity fixture: %w", err)
	}

	if fixture.Name == "" {
		return TextClassificationFixture{}, fmt.Errorf("decode parity fixture: missing name")
	}

	if fixture.Backend != (bionet.Backend{}).Name() {
		return TextClassificationFixture{}, fmt.Errorf("decode parity fixture: unsupported backend %q", fixture.Backend)
	}

	if fixture.Artifact == "" {
		return TextClassificationFixture{}, fmt.Errorf("decode parity fixture: missing artifact path")
	}

	if fixture.Tolerance <= 0 {
		return TextClassificationFixture{}, fmt.Errorf("decode parity fixture: tolerance must be greater than zero")
	}

	if len(fixture.Cases) == 0 {
		return TextClassificationFixture{}, fmt.Errorf("decode parity fixture: no cases defined")
	}

	return fixture, nil
}

// RunTextClassificationFixture loads the configured artifact, runs all fixture
// cases, and compares outputs against the recorded probabilities.
func RunTextClassificationFixture(path string) (TextClassificationReport, error) {
	fixture, err := LoadTextClassificationFixture(path)
	if err != nil {
		return TextClassificationReport{}, err
	}

	artifactPath := filepath.Join(filepath.Dir(path), fixture.Artifact)
	model, err := bionet.LoadModel(artifactPath)
	if err != nil {
		return TextClassificationReport{}, fmt.Errorf("load parity artifact: %w", err)
	}

	report := TextClassificationReport{
		FixturePath: path,
		FixtureName: fixture.Name,
		Backend:     fixture.Backend,
		Tolerance:   fixture.Tolerance,
	}

	for _, tc := range fixture.Cases {
		probabilities, err := model.PredictVector(tc.Features)
		if err != nil {
			return TextClassificationReport{}, fmt.Errorf("run parity case %q: %w", tc.ID, err)
		}

		if len(probabilities) != len(tc.ExpectedProbabilities) {
			return TextClassificationReport{}, fmt.Errorf(
				"run parity case %q: probability length mismatch (%d != %d)",
				tc.ID,
				len(probabilities),
				len(tc.ExpectedProbabilities),
			)
		}

		predictedLabel := labelAt(fixture.Labels, argMax(probabilities))
		maxAbsDiff := maxAbsDiff(probabilities, tc.ExpectedProbabilities)
		report.CaseResults = append(report.CaseResults, TextClassificationCaseResult{
			ID:             tc.ID,
			Text:           tc.Text,
			ExpectedLabel:  tc.ExpectedLabel,
			PredictedLabel: predictedLabel,
			MaxAbsDiff:     maxAbsDiff,
			Passed:         maxAbsDiff <= fixture.Tolerance && predictedLabel == tc.ExpectedLabel,
		})
	}

	return report, nil
}

// Passed returns true when every fixture case stayed within tolerance and
// preserved the expected label.
func (r TextClassificationReport) Passed() bool {
	for _, result := range r.CaseResults {
		if !result.Passed {
			return false
		}
	}

	return true
}

// String renders a compact human-readable summary for CLI output.
func (r TextClassificationReport) String() string {
	summary := fmt.Sprintf(
		"fixture=%s backend=%s cases=%d tolerance=%.6g status=%s\n",
		r.FixtureName,
		r.Backend,
		len(r.CaseResults),
		r.Tolerance,
		passFail(r.Passed()),
	)

	for _, result := range r.CaseResults {
		summary += fmt.Sprintf(
			"- %s: label=%s predicted=%s max_abs_diff=%.6g status=%s\n",
			result.ID,
			result.ExpectedLabel,
			result.PredictedLabel,
			result.MaxAbsDiff,
			passFail(result.Passed),
		)
	}

	return summary
}

func passFail(ok bool) string {
	if ok {
		return "PASS"
	}

	return "FAIL"
}

func maxAbsDiff(actual, expected []float64) float64 {
	maxDiff := 0.0
	for i := range actual {
		diff := math.Abs(actual[i] - expected[i])
		if diff > maxDiff {
			maxDiff = diff
		}
	}

	return maxDiff
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

func labelAt(labels []string, idx int) string {
	if idx < 0 || idx >= len(labels) {
		return fmt.Sprintf("class_%d", idx)
	}

	return labels[idx]
}
