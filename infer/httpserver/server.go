package httpserver

import (
	"encoding/json"
	"net/http"
	"slices"

	"github.com/pergamon-labs/infergo/infer/packs"
)

// PredictRequest is the stable request contract for the current InferGo HTTP
// serving surface.
type PredictRequest struct {
	CaseID string   `json:"case_id,omitempty"`
	Text   string   `json:"text,omitempty"`
	Tokens []string `json:"tokens,omitempty"`
}

// ErrorResponse is returned for JSON API errors.
type ErrorResponse struct {
	Error string `json:"error"`
}

// HealthResponse is returned from /healthz.
type HealthResponse struct {
	Status string `json:"status"`
}

// MetadataResponse describes the loaded pack and supported HTTP surface.
type MetadataResponse struct {
	Task            string   `json:"task"`
	PackKey         string   `json:"pack_key"`
	ModelID         string   `json:"model_id"`
	SupportsRawText bool     `json:"supports_raw_text"`
	Endpoints       []string `json:"endpoints"`
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
func NewTextPackMux(pack *packs.TextPack) *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			writeMethodNotAllowed(w, http.MethodGet)
			return
		}
		writeJSON(w, http.StatusOK, HealthResponse{Status: "ok"})
	})
	mux.HandleFunc("/metadata", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			writeMethodNotAllowed(w, http.MethodGet)
			return
		}
		writeJSON(w, http.StatusOK, MetadataResponse{
			Task:            "text-classification",
			PackKey:         pack.Key(),
			ModelID:         pack.ModelID(),
			SupportsRawText: pack.SupportsRawText(),
			Endpoints:       []string{"/healthz", "/metadata", "/predict"},
		})
	})
	mux.HandleFunc("/predict", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeMethodNotAllowed(w, http.MethodPost)
			return
		}

		var req PredictRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeError(w, http.StatusBadRequest, "invalid json body")
			return
		}

		switch {
		case req.Text != "":
			result, err := pack.PredictText(req.Text)
			if err != nil {
				writeError(w, http.StatusBadRequest, err.Error())
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
				writeError(w, http.StatusBadRequest, err.Error())
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
				writeError(w, http.StatusBadRequest, err.Error())
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
			writeError(w, http.StatusBadRequest, "provide text, tokens, or a valid case_id")
		}
	})
	return mux
}

// NewTokenPackMux builds an HTTP mux for a loaded curated token pack.
func NewTokenPackMux(pack *packs.TokenPack) *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			writeMethodNotAllowed(w, http.MethodGet)
			return
		}
		writeJSON(w, http.StatusOK, HealthResponse{Status: "ok"})
	})
	mux.HandleFunc("/metadata", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			writeMethodNotAllowed(w, http.MethodGet)
			return
		}
		writeJSON(w, http.StatusOK, MetadataResponse{
			Task:            "token-classification",
			PackKey:         pack.Key(),
			ModelID:         pack.ModelID(),
			SupportsRawText: pack.SupportsRawText(),
			Endpoints:       []string{"/healthz", "/metadata", "/predict"},
		})
	})
	mux.HandleFunc("/predict", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeMethodNotAllowed(w, http.MethodPost)
			return
		}

		var req PredictRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeError(w, http.StatusBadRequest, "invalid json body")
			return
		}

		switch {
		case req.CaseID != "":
			result, err := pack.PredictReferenceCase(req.CaseID)
			if err != nil {
				writeError(w, http.StatusBadRequest, err.Error())
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
				writeError(w, http.StatusBadRequest, err.Error())
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
				writeError(w, http.StatusBadRequest, err.Error())
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
			writeError(w, http.StatusBadRequest, "provide text, tokens, or a valid case_id")
		}
	})
	return mux
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}

func writeError(w http.ResponseWriter, status int, message string) {
	writeJSON(w, status, ErrorResponse{Error: message})
}

func writeMethodNotAllowed(w http.ResponseWriter, expected string) {
	writeError(w, http.StatusMethodNotAllowed, "method not allowed; expected "+expected)
}

func clone2D(values [][]float64) [][]float64 {
	out := make([][]float64, len(values))
	for i := range values {
		out[i] = slices.Clone(values[i])
	}
	return out
}
