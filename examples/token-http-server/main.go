package main

import (
	"encoding/json"
	"flag"
	"log"
	"net/http"
	"strings"

	"github.com/pergamon-labs/infergo/infer/packs"
)

type predictRequest struct {
	CaseID string   `json:"case_id,omitempty"`
	Text   string   `json:"text,omitempty"`
	Tokens []string `json:"tokens,omitempty"`
}

type predictResponse struct {
	Backend       string      `json:"backend"`
	ModelID       string      `json:"model_id"`
	Labels        []string    `json:"labels"`
	Tokens        []string    `json:"tokens,omitempty"`
	TokenLabels   []string    `json:"token_labels"`
	TokenLogits   [][]float64 `json:"token_logits"`
	ScoringMask   []int       `json:"scoring_mask,omitempty"`
	ReferenceCase string      `json:"reference_case,omitempty"`
}

func main() {
	addr := flag.String("addr", ":8081", "http listen address")
	packKey := flag.String("pack", "infergo-basic-french-ner", "supported checked-in token pack key")
	flag.Parse()

	pack, err := packs.LoadTokenPack(*packKey)
	if err != nil {
		log.Fatalf("load token pack: %v", err)
	}
	defer pack.Close()

	http.HandleFunc("/predict", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req predictRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "invalid json body", http.StatusBadRequest)
			return
		}

		var tokens []string
		var prediction predictResponse
		if req.CaseID != "" {
			result, err := pack.PredictReferenceCase(req.CaseID)
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			prediction = predictResponse{
				Backend:     result.Backend,
				ModelID:     result.ModelID,
				Labels:      result.Labels,
				TokenLabels: result.TokenLabels,
				TokenLogits: result.TokenLogits,
			}
		} else if req.Text != "" {
			result, err := pack.PredictText(req.Text)
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			prediction = predictResponse{
				Backend:     result.Backend,
				ModelID:     result.ModelID,
				Labels:      result.Labels,
				TokenLabels: result.TokenLabels,
				TokenLogits: result.TokenLogits,
			}
		} else if len(req.Tokens) > 0 {
			result, err := pack.PredictTokens(req.Tokens)
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			prediction = predictResponse{
				Backend:     result.Backend,
				ModelID:     result.ModelID,
				Labels:      result.Labels,
				TokenLabels: result.TokenLabels,
				TokenLogits: result.TokenLogits,
			}
			tokens = append([]string(nil), req.Tokens...)
		} else {
			http.Error(w, "provide text, tokens, or a valid case_id", http.StatusBadRequest)
			return
		}

		resp := prediction
		resp.Tokens = tokens
		if req.CaseID != "" {
			resp.ReferenceCase = req.CaseID
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(resp); err != nil {
			http.Error(w, "encode response failed", http.StatusInternalServerError)
			return
		}
	})

	log.Printf("InferGo token example server listening on %s", *addr)
	curlAddr := *addr
	if strings.HasPrefix(curlAddr, ":") {
		curlAddr = "127.0.0.1" + curlAddr
	}
	log.Printf("Try raw text: curl -s -X POST http://%s/predict -H 'Content-Type: application/json' -d '{\"text\":\"Sophie Tremblay a parlé avec Hydro-Québec à Montréal.\"}' | jq", curlAddr)
	log.Printf("Try pieces: curl -s -X POST http://%s/predict -H 'Content-Type: application/json' -d '{\"tokens\":[\"jean\",\"dupont\",\"a\",\"rencontré\",\"airbus\",\"à\",\"paris\"]}' | jq", curlAddr)
	log.Printf("Try a checked-in case: curl -s -X POST http://%s/predict -H 'Content-Type: application/json' -d '{\"case_id\":\"frca-003\"}' | jq", curlAddr)
	if err := http.ListenAndServe(*addr, nil); err != nil {
		log.Fatal(err)
	}
}
