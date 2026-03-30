package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/pergamon-labs/infergo/infer/packs"
	"github.com/pergamon-labs/infergo/internal/parity"
)

func main() {
	packKey := flag.String("pack", "distilbert-sst2", "supported checked-in text pack key")
	referencePath := flag.String("reference", "./testdata/reference/text-classification/distilbert-sst2-reference.json", "path to a reference JSON file with demo cases")
	caseID := flag.String("case-id", "positive-review", "reference case id to run when -text is empty")
	text := flag.String("text", "", "raw text to score when the chosen pack supports it")
	flag.Parse()

	reference, err := parity.LoadTransformersTextClassificationReference(*referencePath)
	if err != nil {
		log.Fatalf("load reference: %v", err)
	}

	item, err := findTextCase(reference, *caseID)
	if err != nil {
		log.Fatal(err)
	}

	pack, err := packs.LoadTextPack(*packKey)
	if err != nil {
		log.Fatalf("load text pack: %v", err)
	}
	defer pack.Close()

	var prediction any
	if *text != "" {
		result, err := pack.PredictText(*text)
		if err != nil {
			log.Fatalf("predict text: %v", err)
		}
		prediction = map[string]any{
			"pack":            *packKey,
			"text":            *text,
			"backend":         result.Backend,
			"model_id":        result.ModelID,
			"labels":          result.Labels,
			"observed_logits": result.Logits,
			"observed_label":  result.Label,
		}
	} else {
		result, err := pack.PredictReferenceCase(*caseID)
		if err != nil {
			log.Fatalf("predict case: %v", err)
		}

		prediction = map[string]any{
			"pack":            *packKey,
			"reference_case":  item.ID,
			"text":            item.Text,
			"tokens":          item.Tokens,
			"backend":         result.Backend,
			"model_id":        result.ModelID,
			"labels":          result.Labels,
			"observed_logits": result.Logits,
			"observed_label":  result.Label,
		}
	}

	encoder := json.NewEncoder(os.Stdout)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(prediction); err != nil {
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
