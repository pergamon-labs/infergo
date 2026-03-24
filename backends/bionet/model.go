package bionet

import (
	"fmt"
	"slices"

	"github.com/pergamon-labs/infergo/backends/bionet/runtime/module"
	"github.com/pergamon-labs/infergo/backends/bionet/runtime/tensor"
	"github.com/pergamon-labs/infergo/infer"
)

var _ infer.Model = (*Model)(nil)

// Model wraps a BIOnet runtime module behind a backend-specific model handle.
type Model struct {
	runtime *module.Module
}

// Load satisfies the generic loader contract for BIOnet artifacts.
func (Backend) Load(path string) (infer.Model, error) {
	return LoadModel(path)
}

// LoadModel loads a BIOnet gob artifact from disk.
func LoadModel(path string) (*Model, error) {
	runtimeModel, err := module.LoadFromFile(path)
	if err != nil {
		return nil, fmt.Errorf("load bionet model: %w", err)
	}

	return &Model{runtime: runtimeModel}, nil
}

// BackendName returns the stable backend identifier for this model.
func (*Model) BackendName() string {
	return Backend{}.Name()
}

// PredictVector runs a single feature vector through the BIOnet runtime model.
func (m *Model) PredictVector(features []float64) ([]float64, error) {
	if m == nil || m.runtime == nil {
		return nil, fmt.Errorf("predict vector: model is not initialized")
	}

	if len(features) == 0 {
		return nil, fmt.Errorf("predict vector: features must not be empty")
	}

	output, err := m.runtime.Forward(tensor.New(slices.Clone(features), []int{len(features)}))
	if err != nil {
		return nil, fmt.Errorf("predict vector: %w", err)
	}

	return slices.Clone(output.Values()), nil
}
