package httpserver

import (
	"encoding/json"
	"fmt"
	"net/http"
	"slices"
	"time"

	"github.com/pergamon-labs/infergo/infer"
	"github.com/pergamon-labs/infergo/infer/packs"
)

// PredictRequest is the stable request contract for the current InferGo HTTP
// serving surface.
type PredictRequest struct {
	CaseID        string   `json:"case_id,omitempty"`
	Text          string   `json:"text,omitempty"`
	Tokens        []string `json:"tokens,omitempty"`
	InputIDs      []int64  `json:"input_ids,omitempty"`
	AttentionMask []int64  `json:"attention_mask,omitempty"`
}

// ErrorDetail describes one structured API error.
type ErrorDetail struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

// ErrorResponse is returned for JSON API errors.
type ErrorResponse struct {
	Error ErrorDetail `json:"error"`
}

// HealthResponse is returned from /healthz.
type HealthResponse struct {
	Status string `json:"status"`
}

// MetadataResponse describes the loaded pack and supported HTTP surface.
type MetadataResponse struct {
	Task                   string   `json:"task"`
	PackKey                string   `json:"pack_key,omitempty"`
	ModelID                string   `json:"model_id"`
	SupportsRawText        bool     `json:"supports_raw_text"`
	SupportsPairText       bool     `json:"supports_pair_text,omitempty"`
	SupportsTokenizedInput bool     `json:"supports_tokenized_input,omitempty"`
	SupportedInputs        []string `json:"supported_inputs,omitempty"`
	Endpoints              []string `json:"endpoints"`
}

// TextPredictResponse is the stable JSON response shape for text
// classification over the current curated pack surface.
type TextPredictResponse struct {
	Backend        string    `json:"backend"`
	ModelID        string    `json:"model_id"`
	Labels         []string  `json:"labels"`
	ObservedLabel  string    `json:"observed_label"`
	ObservedLogits []float64 `json:"observed_logits"`
	ReferenceCase  string    `json:"reference_case,omitempty"`
	Tokens         []string  `json:"tokens,omitempty"`
}

// TokenPredictResponse is the stable JSON response shape for token
// classification over the current curated pack surface.
type TokenPredictResponse struct {
	Backend       string      `json:"backend"`
	ModelID       string      `json:"model_id"`
	Labels        []string    `json:"labels"`
	Tokens        []string    `json:"tokens,omitempty"`
	TokenLabels   []string    `json:"token_labels"`
	TokenLogits   [][]float64 `json:"token_logits"`
	ReferenceCase string      `json:"reference_case,omitempty"`
}

// NewTextPackMux builds an HTTP mux for a loaded curated text pack.
func NewTextPackMux(pack *packs.TextPack, options ...Option) *http.ServeMux {
	cfg := applyOptions(options)
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", withLogging("healthz", cfg, func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			writeMethodNotAllowed(w, http.MethodGet)
			return
		}
		writeJSON(w, http.StatusOK, HealthResponse{Status: "ok"})
	}))
	mux.HandleFunc("/metadata", withLogging("metadata", cfg, func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			writeMethodNotAllowed(w, http.MethodGet)
			return
		}
		writeJSON(w, http.StatusOK, MetadataResponse{
			Task:                   "text-classification",
			PackKey:                pack.Key(),
			ModelID:                pack.ModelID(),
			SupportsRawText:        pack.SupportsRawText(),
			SupportsTokenizedInput: false,
			SupportedInputs:        supportedTextPackInputs(pack),
			Endpoints:              []string{"/healthz", "/metadata", "/predict"},
		})
	}))
	mux.HandleFunc("/predict", withLogging("predict", cfg, func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeMethodNotAllowed(w, http.MethodPost)
			return
		}

		var req PredictRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeError(w, http.StatusBadRequest, "invalid_json", "invalid json body")
			return
		}
		if err := validatePredictRequest(req); err != nil {
			writeError(w, http.StatusBadRequest, "invalid_request", err.Error())
			return
		}

		switch {
		case req.Text != "":
			result, err := pack.PredictText(req.Text)
			if err != nil {
				writeError(w, http.StatusBadRequest, "invalid_request", err.Error())
				return
			}
			writeJSON(w, http.StatusOK, TextPredictResponse{
				Backend:        result.Backend,
				ModelID:        result.ModelID,
				Labels:         slices.Clone(result.Labels),
				ObservedLabel:  result.Label,
				ObservedLogits: slices.Clone(result.Logits),
			})
		case len(req.Tokens) > 0:
			result, err := pack.PredictTokens(req.Tokens)
			if err != nil {
				writeError(w, http.StatusBadRequest, "invalid_request", err.Error())
				return
			}
			writeJSON(w, http.StatusOK, TextPredictResponse{
				Backend:        result.Backend,
				ModelID:        result.ModelID,
				Labels:         slices.Clone(result.Labels),
				ObservedLabel:  result.Label,
				ObservedLogits: slices.Clone(result.Logits),
				Tokens:         slices.Clone(req.Tokens),
			})
		case req.CaseID != "":
			result, err := pack.PredictReferenceCase(req.CaseID)
			if err != nil {
				writeError(w, http.StatusBadRequest, "invalid_request", err.Error())
				return
			}
			writeJSON(w, http.StatusOK, TextPredictResponse{
				Backend:        result.Backend,
				ModelID:        result.ModelID,
				Labels:         slices.Clone(result.Labels),
				ObservedLabel:  result.Label,
				ObservedLogits: slices.Clone(result.Logits),
				ReferenceCase:  req.CaseID,
			})
		default:
			writeError(w, http.StatusBadRequest, "invalid_request", "provide exactly one of text, tokens, or case_id")
		}
	}))
	mux.HandleFunc("/", withLogging("not_found", cfg, func(w http.ResponseWriter, r *http.Request) {
		writeError(w, http.StatusNotFound, "not_found", "route not found")
	}))
	return mux
}

// TextClassifierMetadata describes the generic serving capabilities for a
// loaded text-classification bundle that is not routed through curated packs.
type TextClassifierMetadata struct {
	ModelID                string
	SupportsRawText        bool
	SupportsPairText       bool
	SupportsTokenizedInput bool
}

// NewTextClassifierMux builds an HTTP mux for a generic loaded text
// classifier. It is the first non-curated serving surface for exported
// family-1 bundles and currently accepts tokenized inputs only.
func NewTextClassifierMux(classifier infer.TextClassifier, metadata TextClassifierMetadata, options ...Option) *http.ServeMux {
	cfg := applyOptions(options)
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", withLogging("healthz", cfg, func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			writeMethodNotAllowed(w, http.MethodGet)
			return
		}
		writeJSON(w, http.StatusOK, HealthResponse{Status: "ok"})
	}))
	mux.HandleFunc("/metadata", withLogging("metadata", cfg, func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			writeMethodNotAllowed(w, http.MethodGet)
			return
		}
		writeJSON(w, http.StatusOK, MetadataResponse{
			Task:                   "text-classification",
			ModelID:                metadata.ModelID,
			SupportsRawText:        metadata.SupportsRawText,
			SupportsPairText:       metadata.SupportsPairText,
			SupportsTokenizedInput: metadata.SupportsTokenizedInput,
			SupportedInputs:        supportedTextClassifierInputs(metadata),
			Endpoints:              []string{"/healthz", "/metadata", "/predict"},
		})
	}))
	mux.HandleFunc("/predict", withLogging("predict", cfg, func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeMethodNotAllowed(w, http.MethodPost)
			return
		}

		var req PredictRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeError(w, http.StatusBadRequest, "invalid_json", "invalid json body")
			return
		}
		if err := validateGenericTextPredictRequest(req, metadata); err != nil {
			writeError(w, http.StatusBadRequest, "invalid_request", err.Error())
			return
		}

		result, err := classifier.Predict(infer.TextInput{
			InputIDs:      slices.Clone(req.InputIDs),
			AttentionMask: slices.Clone(req.AttentionMask),
		})
		if err != nil {
			writeError(w, http.StatusBadRequest, "invalid_request", err.Error())
			return
		}

		writeJSON(w, http.StatusOK, TextPredictResponse{
			Backend:        result.Backend,
			ModelID:        result.ModelID,
			Labels:         slices.Clone(result.Labels),
			ObservedLabel:  result.Label,
			ObservedLogits: slices.Clone(result.Logits),
		})
	}))
	mux.HandleFunc("/", withLogging("not_found", cfg, func(w http.ResponseWriter, r *http.Request) {
		writeError(w, http.StatusNotFound, "not_found", "route not found")
	}))
	return mux
}

// NewTokenPackMux builds an HTTP mux for a loaded curated token pack.
func NewTokenPackMux(pack *packs.TokenPack, options ...Option) *http.ServeMux {
	cfg := applyOptions(options)
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", withLogging("healthz", cfg, func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			writeMethodNotAllowed(w, http.MethodGet)
			return
		}
		writeJSON(w, http.StatusOK, HealthResponse{Status: "ok"})
	}))
	mux.HandleFunc("/metadata", withLogging("metadata", cfg, func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			writeMethodNotAllowed(w, http.MethodGet)
			return
		}
		writeJSON(w, http.StatusOK, MetadataResponse{
			Task:                   "token-classification",
			PackKey:                pack.Key(),
			ModelID:                pack.ModelID(),
			SupportsRawText:        pack.SupportsRawText(),
			SupportsTokenizedInput: false,
			SupportedInputs:        supportedTokenPackInputs(pack),
			Endpoints:              []string{"/healthz", "/metadata", "/predict"},
		})
	}))
	mux.HandleFunc("/predict", withLogging("predict", cfg, func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeMethodNotAllowed(w, http.MethodPost)
			return
		}

		var req PredictRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeError(w, http.StatusBadRequest, "invalid_json", "invalid json body")
			return
		}
		if err := validatePredictRequest(req); err != nil {
			writeError(w, http.StatusBadRequest, "invalid_request", err.Error())
			return
		}

		switch {
		case req.CaseID != "":
			result, err := pack.PredictReferenceCase(req.CaseID)
			if err != nil {
				writeError(w, http.StatusBadRequest, "invalid_request", err.Error())
				return
			}
			writeJSON(w, http.StatusOK, TokenPredictResponse{
				Backend:       result.Backend,
				ModelID:       result.ModelID,
				Labels:        slices.Clone(result.Labels),
				TokenLabels:   slices.Clone(result.TokenLabels),
				TokenLogits:   clone2D(result.TokenLogits),
				ReferenceCase: req.CaseID,
			})
		case req.Text != "":
			result, err := pack.PredictText(req.Text)
			if err != nil {
				writeError(w, http.StatusBadRequest, "invalid_request", err.Error())
				return
			}
			writeJSON(w, http.StatusOK, TokenPredictResponse{
				Backend:     result.Backend,
				ModelID:     result.ModelID,
				Labels:      slices.Clone(result.Labels),
				TokenLabels: slices.Clone(result.TokenLabels),
				TokenLogits: clone2D(result.TokenLogits),
			})
		case len(req.Tokens) > 0:
			result, err := pack.PredictTokens(req.Tokens)
			if err != nil {
				writeError(w, http.StatusBadRequest, "invalid_request", err.Error())
				return
			}
			writeJSON(w, http.StatusOK, TokenPredictResponse{
				Backend:     result.Backend,
				ModelID:     result.ModelID,
				Labels:      slices.Clone(result.Labels),
				Tokens:      slices.Clone(req.Tokens),
				TokenLabels: slices.Clone(result.TokenLabels),
				TokenLogits: clone2D(result.TokenLogits),
			})
		default:
			writeError(w, http.StatusBadRequest, "invalid_request", "provide exactly one of text, tokens, or case_id")
		}
	}))
	mux.HandleFunc("/", withLogging("not_found", cfg, func(w http.ResponseWriter, r *http.Request) {
		writeError(w, http.StatusNotFound, "not_found", "route not found")
	}))
	return mux
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}

func writeError(w http.ResponseWriter, status int, code, message string) {
	writeJSON(w, status, ErrorResponse{
		Error: ErrorDetail{
			Code:    code,
			Message: message,
		},
	})
}

func writeMethodNotAllowed(w http.ResponseWriter, expected string) {
	writeError(w, http.StatusMethodNotAllowed, "method_not_allowed", "method not allowed; expected "+expected)
}

func clone2D(values [][]float64) [][]float64 {
	out := make([][]float64, len(values))
	for i := range values {
		out[i] = slices.Clone(values[i])
	}
	return out
}

func validatePredictRequest(req PredictRequest) error {
	var count int
	if req.CaseID != "" {
		count++
	}
	if req.Text != "" {
		count++
	}
	if len(req.Tokens) > 0 {
		count++
	}
	if count != 1 {
		return fmt.Errorf("provide exactly one of text, tokens, or case_id")
	}
	return nil
}

func validateGenericTextPredictRequest(req PredictRequest, metadata TextClassifierMetadata) error {
	var count int
	if len(req.InputIDs) > 0 {
		count++
	}
	if req.Text != "" || req.CaseID != "" || len(req.Tokens) > 0 {
		count++
	}
	if count != 1 {
		return fmt.Errorf("provide exactly one supported input mode for this bundle")
	}
	if len(req.InputIDs) == 0 {
		return fmt.Errorf("this bundle currently supports tokenized input only; provide input_ids and optional attention_mask")
	}
	if !metadata.SupportsTokenizedInput {
		return fmt.Errorf("this bundle does not support tokenized input")
	}
	if len(req.AttentionMask) > 0 && len(req.AttentionMask) != len(req.InputIDs) {
		return fmt.Errorf("input_ids and attention_mask length mismatch (%d != %d)", len(req.InputIDs), len(req.AttentionMask))
	}
	return nil
}

func supportedTextPackInputs(pack *packs.TextPack) []string {
	inputs := []string{"tokens", "case_id"}
	if pack != nil && pack.SupportsRawText() {
		inputs = append([]string{"text"}, inputs...)
	}
	return inputs
}

func supportedTokenPackInputs(pack *packs.TokenPack) []string {
	inputs := []string{"tokens", "case_id"}
	if pack != nil && pack.SupportsRawText() {
		inputs = append([]string{"text"}, inputs...)
	}
	return inputs
}

func supportedTextClassifierInputs(metadata TextClassifierMetadata) []string {
	inputs := make([]string, 0, 2)
	if metadata.SupportsRawText {
		inputs = append(inputs, "text")
	}
	if metadata.SupportsTokenizedInput {
		inputs = append(inputs, "input_ids")
	}
	return inputs
}

type statusRecorder struct {
	http.ResponseWriter
	status int
}

func (r *statusRecorder) WriteHeader(status int) {
	r.status = status
	r.ResponseWriter.WriteHeader(status)
}

func (r *statusRecorder) Write(data []byte) (int, error) {
	if r.status == 0 {
		r.status = http.StatusOK
	}
	return r.ResponseWriter.Write(data)
}

func withLogging(route string, cfg Config, next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		rec := &statusRecorder{ResponseWriter: w}
		started := time.Now()
		next(rec, r)
		if cfg.LogRequests && cfg.Logger != nil {
			cfg.Logger.Printf("infergo http route=%s method=%s status=%d duration=%s", route, r.Method, rec.status, time.Since(started).Round(time.Microsecond))
		}
	}
}
