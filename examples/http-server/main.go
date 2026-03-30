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
	Backend        string    `json:"backend"`
	ModelID        string    `json:"model_id"`
	Labels         []string  `json:"labels"`
	ObservedLabel  string    `json:"observed_label"`
	ObservedLogits []float64 `json:"observed_logits"`
}

func main() {
	addr := flag.String("addr", ":8080", "http listen address")
	packKey := flag.String("pack", "infergo-basic-sst2", "supported checked-in text pack key")
	flag.Parse()

	pack, err := packs.LoadTextPack(*packKey)
	if err != nil {
		log.Fatalf("load text pack: %v", err)
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

		var prediction inferResponse
		switch {
		case req.Text != "":
			result, err := pack.PredictText(req.Text)
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			prediction = inferResponse{
				Backend:        result.Backend,
				ModelID:        result.ModelID,
				Labels:         result.Labels,
				ObservedLabel:  result.Label,
				ObservedLogits: result.Logits,
			}
		case len(req.Tokens) > 0:
			result, err := pack.PredictTokens(req.Tokens)
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			prediction = inferResponse{
				Backend:        result.Backend,
				ModelID:        result.ModelID,
				Labels:         result.Labels,
				ObservedLabel:  result.Label,
				ObservedLogits: result.Logits,
			}
		case req.CaseID != "":
			result, err := pack.PredictReferenceCase(req.CaseID)
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			prediction = inferResponse{
				Backend:        result.Backend,
				ModelID:        result.ModelID,
				Labels:         result.Labels,
				ObservedLabel:  result.Label,
				ObservedLogits: result.Logits,
			}
		default:
			http.Error(w, "provide tokens, text, or a valid case_id", http.StatusBadRequest)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(prediction); err != nil {
			http.Error(w, "encode response failed", http.StatusInternalServerError)
			return
		}
	})

	log.Printf("InferGo example server listening on %s", *addr)
	curlAddr := *addr
	if strings.HasPrefix(curlAddr, ":") {
		curlAddr = "127.0.0.1" + curlAddr
	}
	log.Printf("Try raw text: curl -s -X POST http://%s/predict -H 'Content-Type: application/json' -d '{\"text\":\"This product is excellent and reliable.\"}' | jq", curlAddr)
	log.Printf("Try tokens: curl -s -X POST http://%s/predict -H 'Content-Type: application/json' -d '{\"tokens\":[\"this\",\"product\",\"is\",\"excellent\",\"and\",\"reliable\",\".\"]}' | jq", curlAddr)
	log.Printf("Try a checked-in case: curl -s -X POST http://%s/predict -H 'Content-Type: application/json' -d '{\"case_id\":\"positive-review\"}' | jq", curlAddr)
	if err := http.ListenAndServe(*addr, nil); err != nil {
		log.Fatal(err)
	}
}

type inferResponse = predictResponse
