//go:build torchscript_native && cgo

package binding

import "github.com/pergamon-labs/infergo/backends/torchscript/bindingnative"

// Module wraps the native libtorch-backed module implementation.
type Module struct {
	native *bindingnative.Module
}

// LoadModule loads a TorchScript artifact into the native binding implementation.
func LoadModule(path string) (*Module, error) {
	nativeModule, err := bindingnative.LoadModule(path)
	if err != nil {
		return nil, err
	}

	return &Module{native: nativeModule}, nil
}

// ForwardTextClassification runs a batch through the native binding implementation.
func (m *Module) ForwardTextClassification(inputIDs, attentionMasks [][]int64) ([][]float64, error) {
	return m.native.ForwardTextClassification(inputIDs, attentionMasks)
}

// Close releases native resources.
func (m *Module) Close() error {
	if m == nil || m.native == nil {
		return nil
	}

	return m.native.Close()
}
