package bionet

import "github.com/minervaai/infergo/infer"

// Backend is the scaffolded BIOnet backend identifier.
type Backend struct{}

var _ infer.Backend = Backend{}
var _ infer.Loader = Backend{}

// Name returns the stable backend name.
func (Backend) Name() string {
	return "bionet"
}
