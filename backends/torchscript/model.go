package torchscript

import (
	"fmt"
	"path/filepath"

	"github.com/minervaai/infergo/backends/torchscript/binding"
	"github.com/minervaai/infergo/infer"
)

var _ infer.Backend = Backend{}
var _ infer.Loader = Backend{}
var _ infer.Model = (*Model)(nil)

// Predictor exposes the batch prediction surface used by parity and future
// backend consumers.
type Predictor interface {
	PredictBatch(inputIDs, attentionMasks [][]int64) ([][]float64, error)
	Labels() []string
	ModelID() string
	SequenceLength() int
	PadTokenID() int
	Close() error
}

// Model wraps a TorchScript bundle behind the torchscript backend.
type Model struct {
	metadata BundleMetadata
	module   *binding.Module
}

// Load satisfies the generic loader contract for TorchScript bundles.
func (Backend) Load(path string) (infer.Model, error) {
	return LoadBundle(path)
}

// LoadBundle loads a TorchScript bundle from disk.
func LoadBundle(bundleDir string) (*Model, error) {
	metadata, err := LoadBundleMetadata(bundleDir)
	if err != nil {
		return nil, err
	}

	modulePath := filepath.Join(bundleDir, metadata.Artifact)
	module, err := binding.LoadModule(modulePath)
	if err != nil {
		return nil, fmt.Errorf("load torchscript module: %w", err)
	}

	return &Model{
		metadata: metadata,
		module:   module,
	}, nil
}

// BackendName returns the stable backend identifier.
func (*Model) BackendName() string {
	return Backend{}.Name()
}

// PredictBatch runs a batch of token id and attention mask sequences.
func (m *Model) PredictBatch(inputIDs, attentionMasks [][]int64) ([][]float64, error) {
	if m == nil || m.module == nil {
		return nil, fmt.Errorf("predict batch: model is not initialized")
	}

	return m.module.ForwardTextClassification(inputIDs, attentionMasks)
}

// Labels returns the configured class labels.
func (m *Model) Labels() []string {
	return append([]string(nil), m.metadata.Labels...)
}

// ModelID returns the source model identifier for this bundle.
func (m *Model) ModelID() string {
	return m.metadata.ModelID
}

// SequenceLength returns the padded sequence length expected by the bundle.
func (m *Model) SequenceLength() int {
	return m.metadata.SequenceLength
}

// PadTokenID returns the pad token id used when exporting the bundle.
func (m *Model) PadTokenID() int {
	return m.metadata.PadTokenID
}

// Close releases native resources held by the model.
func (m *Model) Close() error {
	if m == nil || m.module == nil {
		return nil
	}

	err := m.module.Close()
	m.module = nil
	return err
}
