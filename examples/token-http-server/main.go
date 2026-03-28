package main

import (
	"encoding/json"
	"flag"
	"log"
	"net/http"
	"strings"

	"github.com/pergamon-labs/infergo/infer"
	"github.com/pergamon-labs/infergo/internal/parity"
)

type predictRequest struct {
	CaseID        string  `json:"case_id,omitempty"`
	InputIDs      []int64 `json:"input_ids,omitempty"`
	AttentionMask []int64 `json:"attention_mask,omitempty"`
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
	bundleDir := flag.String("bundle", "./testdata/native/token-classification/distilcamembert-french-ner-windowed-embedding-linear", "path to a checked-in InferGo-native token bundle")
	referencePath := flag.String("reference", "./testdata/reference/token-classification/distilcamembert-french-ner-reference.json", "path to a token-classification reference JSON file with demo cases")
	flag.Parse()

	classifier, err := infer.LoadTokenClassifier(*bundleDir)
	if err != nil {
		log.Fatalf("load classifier: %v", err)
	}
	defer classifier.Close()

	reference, err := parity.LoadTransformersTokenClassificationReference(*referencePath)
	if err != nil {
		log.Fatalf("load reference: %v", err)
	}
	referenceCases := make(map[string]parity.TransformersTokenClassificationReferenceCase, len(reference.Cases))
	for _, item := range reference.Cases {
		referenceCases[item.ID] = item
	}

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

		inputIDs := req.InputIDs
		attentionMask := req.AttentionMask
		var tokens []string
		var scoringMask []int
		if req.CaseID != "" {
			item, ok := referenceCases[req.CaseID]
			if !ok {
				http.Error(w, "unknown case_id", http.StatusBadRequest)
				return
			}
			inputIDs = intsToInt64(item.InputIDs)
			attentionMask = intsToInt64(item.AttentionMask)
			tokens = append([]string(nil), item.Tokens...)
			scoringMask = append([]int(nil), item.ScoringMask...)
		}

		if len(inputIDs) == 0 {
			http.Error(w, "provide input_ids or a valid case_id", http.StatusBadRequest)
			return
		}

		prediction, err := classifier.Predict(infer.TokenInput{
			InputIDs:      inputIDs,
			AttentionMask: attentionMask,
		})
		if err != nil {
			http.Error(w, "prediction failed", http.StatusInternalServerError)
			return
		}

		resp := predictResponse{
			Backend:     prediction.Backend,
			ModelID:     prediction.ModelID,
			Labels:      prediction.Labels,
			Tokens:      tokens,
			TokenLabels: prediction.TokenLabels,
			TokenLogits: prediction.TokenLogits,
			ScoringMask: scoringMask,
		}
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
	log.Printf("Try: curl -s -X POST http://%s/predict -H 'Content-Type: application/json' -d '{\"case_id\":\"frca-003\"}' | jq", curlAddr)
	if err := http.ListenAndServe(*addr, nil); err != nil {
		log.Fatal(err)
	}
}

func intsToInt64(values []int) []int64 {
	output := make([]int64, len(values))
	for i, value := range values {
		output[i] = int64(value)
	}
	return output
}
