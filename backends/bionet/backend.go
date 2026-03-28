package bionet

// Backend is the scaffolded BIOnet backend identifier.
type Backend struct{}

// Name returns the stable backend name.
func (Backend) Name() string {
	return "bionet"
}
