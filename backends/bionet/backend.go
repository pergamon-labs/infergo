package bionet

import "github.com/pergamon-labs/infergo/infer"

// Backend is the scaffolded BIOnet backend identifier.
type Backend struct{}

var _ infer.Backend = Backend{}
var _ infer.Loader = Backend{}

// Name returns the stable backend name.
func (Backend) Name() string {
	return "bionet"
}
