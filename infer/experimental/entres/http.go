package entres

import (
	"encoding/json"
	"net/http"

	"github.com/pergamon-labs/infergo/infer/httpserver"
)

// HTTPPredictRequest is the family-2 experimental HTTP request contract.
type HTTPPredictRequest struct {
	Vectors [][]float64 `json:"vectors"`
	Message []float64   `json:"message"`
}

// HTTPPredictResponse is the family-2 experimental HTTP response contract.
type HTTPPredictResponse struct {
	Backend string    `json:"backend"`
	ModelID string    `json:"model_id"`
	Scores  []float64 `json:"scores"`
}

// HTTPMetadataResponse describes the loaded family-2 scorer.
type HTTPMetadataResponse struct {
	Task        string   `json:"task"`
	Family      string   `json:"family"`
	Backend     string   `json:"backend"`
	ModelID     string   `json:"model_id"`
	ProfileKind string   `json:"profile_kind,omitempty"`
	VectorSize  int      `json:"vector_size"`
	MessageSize int      `json:"message_size"`
	Endpoints   []string `json:"endpoints"`
}

type errorDetail struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

type errorResponse struct {
	Error errorDetail `json:"error"`
}

// NewMux returns an experimental HTTP mux for the family-2 scorer.
func NewMux(model Scorer) *http.ServeMux {
	mux := http.NewServeMux()

	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			writeMethodNotAllowed(w, http.MethodGet)
			return
		}
		writeJSON(w, http.StatusOK, httpserver.HealthResponse{Status: "ok"})
	})

	mux.HandleFunc("/metadata", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			writeMethodNotAllowed(w, http.MethodGet)
			return
		}
		meta := model.Metadata()
		writeJSON(w, http.StatusOK, HTTPMetadataResponse{
			Task:        meta.Task,
			Family:      meta.Family,
			Backend:     meta.Backend,
			ModelID:     meta.ModelID,
			ProfileKind: meta.ProfileKind,
			VectorSize:  meta.VectorSize,
			MessageSize: meta.MessageSize,
			Endpoints:   []string{"/healthz", "/metadata", "/predict"},
		})
	})

	mux.HandleFunc("/predict", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeMethodNotAllowed(w, http.MethodPost)
			return
		}

		var req HTTPPredictRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeError(w, http.StatusBadRequest, "invalid_json", "invalid json body")
			return
		}
		if len(req.Vectors) == 0 {
			writeError(w, http.StatusBadRequest, "invalid_request", "vectors must not be empty")
			return
		}
		if len(req.Message) == 0 {
			writeError(w, http.StatusBadRequest, "invalid_request", "message must not be empty")
			return
		}

		result, err := model.Predict(Input{
			Vectors: req.Vectors,
			Message: req.Message,
		})
		if err != nil {
			writeError(w, http.StatusBadRequest, "invalid_request", err.Error())
			return
		}

		writeJSON(w, http.StatusOK, HTTPPredictResponse(result))
	})

	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		writeError(w, http.StatusNotFound, "not_found", "route not found")
	})

	return mux
}

func writeMethodNotAllowed(w http.ResponseWriter, expected string) {
	writeError(w, http.StatusMethodNotAllowed, "method_not_allowed", "expected "+expected)
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}

func writeError(w http.ResponseWriter, status int, code, message string) {
	writeJSON(w, status, errorResponse{
		Error: errorDetail{
			Code:    code,
			Message: message,
		},
	})
}
