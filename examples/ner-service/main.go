package main

import (
	"encoding/json"
	"flag"
	"log"
	"net/http"
	"strings"
	"unicode"

	"github.com/pergamon-labs/infergo/infer/httpserver"
	"github.com/pergamon-labs/infergo/infer/packs"
)

type extractRequest struct {
	Text   string   `json:"text,omitempty"`
	Tokens []string `json:"tokens,omitempty"`
}

type metadataResponse struct {
	Task                   string   `json:"task"`
	PackKey                string   `json:"pack_key"`
	ModelID                string   `json:"model_id"`
	SupportsRawText        bool     `json:"supports_raw_text"`
	SupportsTokenizedInput bool     `json:"supports_tokenized_input"`
	Endpoints              []string `json:"endpoints"`
}

type namedEntity struct {
	Label      string   `json:"label"`
	Text       string   `json:"text"`
	Tokens     []string `json:"tokens"`
	StartToken int      `json:"start_token"`
	EndToken   int      `json:"end_token"`
}

type extractResponse struct {
	Backend     string        `json:"backend"`
	ModelID     string        `json:"model_id"`
	PackKey     string        `json:"pack_key"`
	Text        string        `json:"text,omitempty"`
	Tokens      []string      `json:"tokens"`
	TokenLabels []string      `json:"token_labels"`
	Entities    []namedEntity `json:"entities"`
}

type service struct {
	pack *packs.TokenPack
}

func main() {
	addr := flag.String("addr", ":8082", "http listen address")
	packKey := flag.String("pack", "infergo-basic-french-ner", "supported checked-in token pack key")
	flag.Parse()

	pack, err := packs.LoadTokenPack(*packKey)
	if err != nil {
		log.Fatalf("load token pack: %v", err)
	}
	defer pack.Close()

	svc := &service{pack: pack}
	server := &http.Server{
		Addr:    *addr,
		Handler: svc.newMux(),
	}

	log.Printf("InferGo NER example server listening on %s", *addr)
	log.Printf("This sample returns extracted entities from a checked-in token pack.")
	if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatal(err)
	}
}

func (s *service) newMux() *http.ServeMux {
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
		writeJSON(w, http.StatusOK, metadataResponse{
			Task:                   "named-entity-extraction",
			PackKey:                s.pack.Key(),
			ModelID:                s.pack.ModelID(),
			SupportsRawText:        s.pack.SupportsRawText(),
			SupportsTokenizedInput: true,
			Endpoints:              []string{"/healthz", "/metadata", "/extract"},
		})
	})
	mux.HandleFunc("/extract", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeMethodNotAllowed(w, http.MethodPost)
			return
		}

		var req extractRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeError(w, http.StatusBadRequest, "invalid_json", "invalid json body")
			return
		}

		var text string
		var tokens []string
		switch {
		case req.Text != "" && len(req.Tokens) == 0:
			if !s.pack.SupportsRawText() {
				writeError(w, http.StatusBadRequest, "invalid_request", "this pack does not support raw-text tokenization")
				return
			}
			text = req.Text
			var err error
			tokens, err = s.pack.TokenizeText(req.Text)
			if err != nil {
				writeError(w, http.StatusBadRequest, "invalid_request", err.Error())
				return
			}
		case req.Text == "" && len(req.Tokens) > 0:
			tokens = append([]string(nil), req.Tokens...)
		default:
			writeError(w, http.StatusBadRequest, "invalid_request", "provide exactly one of text or tokens")
			return
		}

		prediction, err := s.pack.PredictTokens(tokens)
		if err != nil {
			writeError(w, http.StatusBadRequest, "invalid_request", err.Error())
			return
		}
		if len(tokens) != len(prediction.TokenLabels) {
			writeError(w, http.StatusInternalServerError, "internal_error", "token and label lengths do not match")
			return
		}

		writeJSON(w, http.StatusOK, extractResponse{
			Backend:     prediction.Backend,
			ModelID:     prediction.ModelID,
			PackKey:     s.pack.Key(),
			Text:        text,
			Tokens:      append([]string(nil), tokens...),
			TokenLabels: append([]string(nil), prediction.TokenLabels...),
			Entities:    groupEntities(tokens, prediction.TokenLabels),
		})
	})
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		writeError(w, http.StatusNotFound, "not_found", "route not found")
	})
	return mux
}

func groupEntities(tokens, labels []string) []namedEntity {
	raw := make([]namedEntity, 0, len(tokens)/2)

	flush := func(current *namedEntity) {
		if current == nil || current.Label == "" || len(current.Tokens) == 0 {
			return
		}
		current.Text = joinEntityTokens(current.Tokens)
		raw = append(raw, *current)
	}

	var current *namedEntity
	for idx, label := range labels {
		prefix, entityLabel, ok := parseNERLabel(label)
		if !ok {
			flush(current)
			current = nil
			continue
		}

		shouldStartNew := current == nil || current.Label != entityLabel || prefix == "B"
		if shouldStartNew {
			flush(current)
			current = &namedEntity{
				Label:      entityLabel,
				StartToken: idx,
			}
		}

		current.EndToken = idx
		current.Tokens = append(current.Tokens, tokens[idx])
	}

	flush(current)

	entities := make([]namedEntity, 0, len(raw))
	for _, item := range raw {
		if len(entities) == 0 {
			entities = append(entities, item)
			continue
		}

		last := &entities[len(entities)-1]
		gapStart := last.EndToken + 1
		if last.Label == item.Label && gapStart <= item.StartToken && punctOnlyGap(tokens[gapStart:item.StartToken], labels[gapStart:item.StartToken]) {
			last.Tokens = append(last.Tokens, tokens[gapStart:item.StartToken]...)
			last.Tokens = append(last.Tokens, item.Tokens...)
			last.EndToken = item.EndToken
			last.Text = joinEntityTokens(last.Tokens)
			continue
		}

		entities = append(entities, item)
	}

	return entities
}

func parseNERLabel(label string) (prefix string, entityLabel string, ok bool) {
	if label == "" || label == "O" {
		return "", "", false
	}
	parts := strings.SplitN(label, "-", 2)
	if len(parts) == 2 {
		return parts[0], parts[1], true
	}
	return "I", label, true
}

func joinEntityTokens(tokens []string) string {
	if len(tokens) == 0 {
		return ""
	}

	var b strings.Builder
	for idx, token := range tokens {
		if idx > 0 && !glueToPrevious(token) && !glueToNext(tokens[idx-1]) {
			b.WriteByte(' ')
		}
		b.WriteString(token)
	}
	return b.String()
}

func glueToPrevious(token string) bool {
	for _, r := range token {
		if !unicode.IsPunct(r) {
			return false
		}
	}
	return token != ""
}

func glueToNext(token string) bool {
	switch token {
	case "'", "-", "/", ".":
		return true
	default:
		return false
	}
}

func punctOnlyGap(tokens, labels []string) bool {
	for idx, token := range tokens {
		if _, _, ok := parseNERLabel(labels[idx]); ok {
			return false
		}
		if !glueToPrevious(token) {
			return false
		}
	}
	return true
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}

func writeError(w http.ResponseWriter, status int, code, message string) {
	writeJSON(w, status, httpserver.ErrorResponse{
		Error: httpserver.ErrorDetail{
			Code:    code,
			Message: message,
		},
	})
}

func writeMethodNotAllowed(w http.ResponseWriter, expected string) {
	writeError(w, http.StatusMethodNotAllowed, "method_not_allowed", "method not allowed; expected "+expected)
}
