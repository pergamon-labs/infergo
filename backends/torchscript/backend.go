package torchscript

import "github.com/pergamon-labs/infergo/infer"

// Backend is the scaffolded TorchScript backend identifier.
type Backend struct{}

var _ infer.Backend = Backend{}

// Name returns the stable backend name.
func (Backend) Name() string {
	return "torchscript"
}
