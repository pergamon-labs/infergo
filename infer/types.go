package infer

// Backend describes a named inference backend implementation.
type Backend interface {
	Name() string
}

// Model describes a loaded model that can report its backend identity.
type Model interface {
	BackendName() string
}

// Loader loads a model artifact into a backend-specific model implementation.
type Loader interface {
	Load(path string) (Model, error)
}

// Request is a minimal scaffold for future inference request shaping.
type Request struct {
	Inputs any
}

// Result is a minimal scaffold for future inference result shaping.
type Result struct {
	Outputs any
}
