//go:build !torchscript_native || !cgo

package binding

import "fmt"

// Module is a stub wrapper used when the native TorchScript backend is not enabled.
type Module struct{}

// LoadModule returns a descriptive error when libtorch support is not compiled in.
func LoadModule(path string) (*Module, error) {
	return nil, fmt.Errorf("torchscript native backend requires build tag torchscript_native, CGO_ENABLED=1, libtorch headers/libs available, and CGO_CXXFLAGS / CGO_LDFLAGS configured for that libtorch install")
}

// ForwardTextClassification is unavailable in the stub build.
func (*Module) ForwardTextClassification(inputIDs, attentionMasks [][]int64) ([][]float64, error) {
	return nil, fmt.Errorf("torchscript native backend requires build tag torchscript_native, CGO_ENABLED=1, libtorch headers/libs available, and CGO_CXXFLAGS / CGO_LDFLAGS configured for that libtorch install")
}

// ForwardFeatureScoring is unavailable in the stub build.
func (*Module) ForwardFeatureScoring(vectors [][]float64, message []float64) ([]float64, error) {
	return nil, fmt.Errorf("torchscript native backend requires build tag torchscript_native, CGO_ENABLED=1, libtorch headers/libs available, and CGO_CXXFLAGS / CGO_LDFLAGS configured for that libtorch install")
}

// Close is a no-op for the stub build.
func (*Module) Close() error {
	return nil
}
