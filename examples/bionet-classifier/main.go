package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/pergamon-labs/infergo/backends/bionet"
	"github.com/pergamon-labs/infergo/internal/parity"
)

func main() {
	bundleDir := flag.String("bundle", "./testdata/native/text-classification/distilbert-sst2-embedding-masked-avg-pool", "path to a checked-in InferGo-native text bundle")
	referencePath := flag.String("reference", "./testdata/reference/text-classification/distilbert-sst2-reference.json", "path to a reference JSON file with demo cases")
	caseID := flag.String("case-id", "positive-review", "reference case id to run")
	flag.Parse()

	reference, err := parity.LoadTransformersTextClassificationReference(*referencePath)
	if err != nil {
		log.Fatalf("load reference: %v", err)
	}

	item, err := findTextCase(reference, *caseID)
	if err != nil {
		log.Fatal(err)
	}

	bundle, err := bionet.LoadTextClassificationBundle(*bundleDir)
	if err != nil {
		log.Fatalf("load bundle: %v", err)
	}
	defer bundle.Close()

	logitsBatch, err := bundle.PredictBatch([][]int64{intsToInt64(item.InputIDs)}, [][]int64{intsToInt64(item.AttentionMask)})
	if err != nil {
		log.Fatalf("predict: %v", err)
	}

	logits := logitsBatch[0]
	labelIdx := argMax(logits)

	output := map[string]any{
		"bundle":          *bundleDir,
		"reference_case":  item.ID,
		"text":            item.Text,
		"tokens":          item.Tokens,
		"observed_logits": logits,
		"observed_label":  bundle.Labels()[labelIdx],
	}

	encoder := json.NewEncoder(os.Stdout)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(output); err != nil {
		log.Fatalf("encode output: %v", err)
	}
}

func findTextCase(reference parity.TransformersTextClassificationReference, caseID string) (parity.TransformersTextClassificationReferenceCase, error) {
	for _, item := range reference.Cases {
		if item.ID == caseID {
			return item, nil
		}
	}
	return parity.TransformersTextClassificationReferenceCase{}, fmt.Errorf("reference case %q not found", caseID)
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
