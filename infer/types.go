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

// TextInput describes one text-classification inference request.
//
// When AttentionMask is omitted, InferGo treats every input token id as active.
type TextInput struct {
	InputIDs      []int64 `json:"input_ids"`
	AttentionMask []int64 `json:"attention_mask,omitempty"`
}

// TextPrediction stores the stable text-classification result shape returned by
// the public InferGo API.
type TextPrediction struct {
	Backend string    `json:"backend"`
	ModelID string    `json:"model_id"`
	Labels  []string  `json:"labels"`
	Logits  []float64 `json:"logits"`
	Label   string    `json:"label"`
}

// TextClassifier is the minimal stable public API for checked-in native
// text-classification bundles.
type TextClassifier interface {
	Model
	ModelID() string
	Labels() []string
	Predict(input TextInput) (TextPrediction, error)
	PredictBatch(inputs []TextInput) ([]TextPrediction, error)
	Close() error
}

// TokenInput describes one token-classification inference request.
//
// When AttentionMask is omitted, InferGo treats every input token id as active.
type TokenInput struct {
	InputIDs      []int64 `json:"input_ids"`
	AttentionMask []int64 `json:"attention_mask,omitempty"`
}

// TokenPrediction stores the stable token-classification result shape returned
// by the public InferGo API.
type TokenPrediction struct {
	Backend     string      `json:"backend"`
	ModelID     string      `json:"model_id"`
	Labels      []string    `json:"labels"`
	TokenLabels []string    `json:"token_labels"`
	TokenLogits [][]float64 `json:"token_logits"`
}

// TokenClassifier is the minimal stable public API for checked-in native
// token-classification bundles.
type TokenClassifier interface {
	Model
	ModelID() string
	Labels() []string
	Predict(input TokenInput) (TokenPrediction, error)
	PredictBatch(inputs []TokenInput) ([]TokenPrediction, error)
	Close() error
}
