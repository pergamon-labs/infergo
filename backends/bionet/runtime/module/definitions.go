package module

import "github.com/pergamon-labs/infergo/backends/bionet/runtime/tensor"

type (
	ModuleInterface interface {
		Forward(inputs tensor.Tensor) (tensor.Tensor, error)
	}
)
