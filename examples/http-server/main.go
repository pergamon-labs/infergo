package main

import (
	"encoding/json"
	"flag"
	"log"
	"net/http"
	"strings"

	"github.com/pergamon-labs/infergo/backends/bionet"
	"github.com/pergamon-labs/infergo/internal/parity"
)

type predictRequest struct {
	CaseID        string  `json:"case_id,omitempty"`
	InputIDs      []int64 `json:"input_ids,omitempty"`
	AttentionMask []int64 `json:"attention_mask,omitempty"`
}

type predictResponse struct {
	ModelID        string    `json:"model_id"`
	Labels         []string  `json:"labels"`
	ObservedLabel  string    `json:"observed_label"`
	ObservedLogits []float64 `json:"observed_logits"`
}

func main() {
	addr := flag.String("addr", ":8080", "http listen address")
	bundleDir := flag.String("bundle", "./testdata/native/text-classification/distilbert-sst2-embedding-masked-avg-pool", "path to a checked-in InferGo-native text bundle")
	referencePath := flag.String("reference", "./testdata/reference/text-classification/distilbert-sst2-reference.json", "path to a reference JSON file with demo cases")
	flag.Parse()

	bundle, err := bionet.LoadTextClassificationBundle(*bundleDir)
	if err != nil {
		log.Fatalf("load bundle: %v", err)
	}
	defer bundle.Close()

	reference, err := parity.LoadTransformersTextClassificationReference(*referencePath)
	if err != nil {
		log.Fatalf("load reference: %v", err)
	}
	referenceCases := make(map[string]parity.TransformersTextClassificationReferenceCase, len(reference.Cases))
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
		if req.CaseID != "" {
			item, ok := referenceCases[req.CaseID]
			if !ok {
				http.Error(w, "unknown case_id", http.StatusBadRequest)
				return
			}
			inputIDs = intsToInt64(item.InputIDs)
			attentionMask = intsToInt64(item.AttentionMask)
		}

		if len(inputIDs) == 0 || len(attentionMask) == 0 || len(inputIDs) != len(attentionMask) {
			http.Error(w, "provide matching input_ids and attention_mask or a valid case_id", http.StatusBadRequest)
			return
		}

		logitsBatch, err := bundle.PredictBatch([][]int64{inputIDs}, [][]int64{attentionMask})
		if err != nil {
			http.Error(w, "prediction failed", http.StatusInternalServerError)
			return
		}

		logits := logitsBatch[0]
		labelIdx := argMax(logits)
		resp := predictResponse{
			ModelID:        bundle.ModelID(),
			Labels:         bundle.Labels(),
			ObservedLabel:  bundle.Labels()[labelIdx],
			ObservedLogits: logits,
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(resp); err != nil {
			http.Error(w, "encode response failed", http.StatusInternalServerError)
			return
		}
	})

	log.Printf("InferGo example server listening on %s", *addr)
	curlAddr := *addr
	if strings.HasPrefix(curlAddr, ":") {
		curlAddr = "127.0.0.1" + curlAddr
	}
	log.Printf("Try: curl -s -X POST http://%s/predict -H 'Content-Type: application/json' -d '{\"case_id\":\"positive-review\"}' | jq", curlAddr)
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

func argMax(values []float64) int {
	bestIdx := 0
	bestValue := values[0]
	for i := 1; i < len(values); i++ {
		if values[i] > bestValue {
			bestValue = values[i]
			bestIdx = i
		}
	}
	return bestIdx
}
