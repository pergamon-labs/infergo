package entres

import (
	"fmt"
	"slices"

	"github.com/pergamon-labs/infergo/backends/torchscript"
)

// Input describes one family-2 numeric-feature scoring request.
type Input struct {
	Vectors [][]float64 `json:"vectors"`
	Message []float64   `json:"message"`
}

// Prediction is the stable experimental output shape for family-2 scoring.
type Prediction struct {
	Backend string    `json:"backend"`
	ModelID string    `json:"model_id"`
	Scores  []float64 `json:"scores"`
}

// Metadata describes the validated bundle contract exposed by the experimental
// family-2 scorer API.
type Metadata struct {
	Family            string `json:"family"`
	Task              string `json:"task"`
	Backend           string `json:"backend"`
	ModelID           string `json:"model_id"`
	ProfileKind       string `json:"profile_kind,omitempty"`
	VectorSize        int    `json:"vector_size"`
	MessageSize       int    `json:"message_size"`
	MessageProjection string `json:"message_projection,omitempty"`
}

// Scorer is the experimental family-2 scorer contract.
type Scorer interface {
	BackendName() string
	ModelID() string
	Metadata() Metadata
	Predict(Input) (Prediction, error)
	Close() error
}

// Model wraps the family-2 bridge bundle behind an experimental API.
type Model struct {
	bundle *torchscript.EntityResolutionBundle
}

// Load opens a family-2 bundle from disk.
func Load(bundleDir string) (*Model, error) {
	bundle, err := torchscript.LoadEntityResolutionBundle(bundleDir)
	if err != nil {
		return nil, err
	}
	return &Model{bundle: bundle}, nil
}

// BackendName returns the stable backend identifier for the loaded model.
func (m *Model) BackendName() string {
	return torchscript.Backend{}.Name()
}

// ModelID returns the source model identifier.
func (m *Model) ModelID() string {
	if m == nil || m.bundle == nil {
		return ""
	}
	return m.bundle.ModelID()
}

// Metadata returns the bundle metadata exposed through the experimental API.
func (m *Model) Metadata() Metadata {
	if m == nil || m.bundle == nil {
		return Metadata{}
	}

	meta := m.bundle.Metadata()
	return Metadata{
		Family:            meta.Family,
		Task:              meta.Task,
		Backend:           meta.Backend,
		ModelID:           meta.ModelID,
		ProfileKind:       meta.ProfileKind,
		VectorSize:        meta.Inputs.VectorSize,
		MessageSize:       meta.Inputs.MessageSize,
		MessageProjection: meta.Inputs.MessageProjection,
	}
}

// Predict scores one batch of numeric-feature vectors.
func (m *Model) Predict(input Input) (Prediction, error) {
	if m == nil || m.bundle == nil {
		return Prediction{}, fmt.Errorf("predict: entres scorer is not initialized")
	}

	scores, err := m.bundle.PredictBatch(input.Vectors, input.Message)
	if err != nil {
		return Prediction{}, err
	}

	return Prediction{
		Backend: m.BackendName(),
		ModelID: m.ModelID(),
		Scores:  slices.Clone(scores),
	}, nil
}

// Close releases resources associated with the scorer.
func (m *Model) Close() error {
	if m == nil || m.bundle == nil {
		return nil
	}
	return m.bundle.Close()
}
