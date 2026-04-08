package parity

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type fakeEntityResolutionPredictor struct {
	meta   EntityResolutionPredictorMetadata
	scores []float64
}

func (f fakeEntityResolutionPredictor) Metadata() EntityResolutionPredictorMetadata {
	return f.meta
}

func (f fakeEntityResolutionPredictor) PredictBatch(vectors [][]float64, message []float64) ([]float64, error) {
	return append([]float64(nil), f.scores...), nil
}

func (f fakeEntityResolutionPredictor) Close() error {
	return nil
}

func TestLoadEntityResolutionFixture(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	path := filepath.Join(dir, "fixture.json")
	payload := `{
  "name": "Family 2 parity fixture",
  "source": "minerva-screening-go",
  "model_id": "pergamon/entres-individual",
  "task": "entity-resolution-scoring",
  "family": "numeric-feature-scoring",
  "backend": "torchscript",
  "profile_kind": "individual",
  "generated_at": "2026-04-08T06:00:00Z",
  "vector_size": 3,
  "message_size": 3,
  "input_layout": "stacked_sample_message_channels",
  "message_strategy": "caller_supplied_consensus_vector",
  "message_projection": "legacy_first_value_broadcast",
  "output_interpretation": "confidence",
  "batches": [
    {
      "id": "batch-1",
      "message": [0.5, 0.25, 0.125],
      "cases": [
        {
          "id": "edge-1",
          "left_name": "Jane Doe",
          "right_name": "Janet Doe",
          "vector": [1, 0, 0],
          "expected_score": 0.91
        }
      ]
    }
  ]
}`
	if err := os.WriteFile(path, []byte(payload), 0o644); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	fixture, err := LoadEntityResolutionFixture(path)
	if err != nil {
		t.Fatalf("LoadEntityResolutionFixture() error = %v", err)
	}
	if fixture.ProfileKind != "individual" {
		t.Fatalf("ProfileKind = %q, want individual", fixture.ProfileKind)
	}
	if len(fixture.Batches) != 1 || len(fixture.Batches[0].Cases) != 1 {
		t.Fatalf("unexpected fixture batch shape: %+v", fixture.Batches)
	}
}

func TestCompareEntityResolutionFixture(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	path := filepath.Join(dir, "fixture.json")
	payload := `{
  "name": "Family 2 parity fixture",
  "source": "minerva-screening-go",
  "model_id": "pergamon/entres-individual",
  "task": "entity-resolution-scoring",
  "family": "numeric-feature-scoring",
  "backend": "torchscript",
  "profile_kind": "individual",
  "generated_at": "2026-04-08T06:00:00Z",
  "vector_size": 2,
  "message_size": 2,
  "input_layout": "stacked_sample_message_channels",
  "message_strategy": "caller_supplied_consensus_vector",
  "message_projection": "legacy_first_value_broadcast",
  "output_interpretation": "confidence",
  "batches": [
    {
      "id": "batch-1",
      "message": [0.5, 0.5],
      "cases": [
        {
          "id": "edge-1",
          "left_name": "Jane Doe",
          "right_name": "Janet Doe",
          "vector": [1, 0],
          "expected_score": 0.91
        },
        {
          "id": "edge-2",
          "left_name": "Jane Doe",
          "right_name": "Alice Smith",
          "vector": [0, 1],
          "expected_score": 0.12
        }
      ]
    }
  ]
}`
	if err := os.WriteFile(path, []byte(payload), 0o644); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	report, err := CompareEntityResolutionFixture(path, "bundle-dir", 1e-4, fakeEntityResolutionPredictor{
		meta: EntityResolutionPredictorMetadata{
			ModelID:              "pergamon/entres-individual",
			Task:                 "entity-resolution-scoring",
			Family:               "numeric-feature-scoring",
			Backend:              "torchscript",
			ProfileKind:          "individual",
			VectorSize:           2,
			MessageSize:          2,
			InputLayout:          "stacked_sample_message_channels",
			MessageStrategy:      "caller_supplied_consensus_vector",
			MessageProjection:    "legacy_first_value_broadcast",
			OutputInterpretation: "confidence",
		},
		scores: []float64{0.91001, 0.12001},
	})
	if err != nil {
		t.Fatalf("CompareEntityResolutionFixture() error = %v", err)
	}
	if !report.Passed() {
		t.Fatalf("report should pass: %s", report.String())
	}
}

func TestCompareEntityResolutionFixtureRejectsMetadataMismatch(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	path := filepath.Join(dir, "fixture.json")
	payload := `{
  "name": "Family 2 parity fixture",
  "source": "minerva-screening-go",
  "model_id": "pergamon/entres-individual",
  "task": "entity-resolution-scoring",
  "family": "numeric-feature-scoring",
  "backend": "torchscript",
  "profile_kind": "individual",
  "generated_at": "2026-04-08T06:00:00Z",
  "vector_size": 2,
  "message_size": 2,
  "input_layout": "stacked_sample_message_channels",
  "message_strategy": "caller_supplied_consensus_vector",
  "message_projection": "legacy_first_value_broadcast",
  "output_interpretation": "confidence",
  "batches": [
    {
      "id": "batch-1",
      "message": [0.5, 0.5],
      "cases": [
        {
          "id": "edge-1",
          "left_name": "Jane Doe",
          "right_name": "Janet Doe",
          "vector": [1, 0],
          "expected_score": 0.91
        }
      ]
    }
  ]
}`
	if err := os.WriteFile(path, []byte(payload), 0o644); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	_, err := CompareEntityResolutionFixture(path, "bundle-dir", 1e-4, fakeEntityResolutionPredictor{
		meta: EntityResolutionPredictorMetadata{
			ModelID:              "pergamon/entres-organization",
			Task:                 "entity-resolution-scoring",
			Family:               "numeric-feature-scoring",
			Backend:              "torchscript",
			ProfileKind:          "individual",
			VectorSize:           2,
			MessageSize:          2,
			InputLayout:          "stacked_sample_message_channels",
			MessageStrategy:      "caller_supplied_consensus_vector",
			MessageProjection:    "legacy_first_value_broadcast",
			OutputInterpretation: "confidence",
		},
		scores: []float64{0.91},
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "model id mismatch") {
		t.Fatalf("unexpected error: %v", err)
	}
}
