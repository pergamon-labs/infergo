package main

import (
	"encoding/json"
	"flag"
	"log"
	"os"

	"github.com/pergamon-labs/infergo/infer"
)

func main() {
	bundleDir := flag.String("bundle", "./dist/family1/distilbert-sst2-alpha", "path to an exported family-1 text bundle")
	text := flag.String("text", "This product is excellent and reliable.", "raw text to classify")
	textPair := flag.String("text-pair", "", "optional second text for paired-text bundles")
	flag.Parse()

	bundle, err := infer.LoadTextBundle(*bundleDir)
	if err != nil {
		log.Fatalf("load text bundle: %v", err)
	}
	defer bundle.Close()

	var prediction infer.TextPrediction
	switch {
	case *textPair != "":
		prediction, err = bundle.PredictTextPair(*text, *textPair)
	default:
		prediction, err = bundle.PredictText(*text)
	}
	if err != nil {
		log.Fatalf("predict: %v", err)
	}

	payload := map[string]any{
		"bundle":                   *bundleDir,
		"text":                     *text,
		"text_pair":                *textPair,
		"model_id":                 bundle.ModelID(),
		"supports_raw_text":        bundle.SupportsRawText(),
		"supports_pair_text":       bundle.SupportsPairText(),
		"supports_tokenized_input": bundle.SupportsTokenizedInput(),
		"backend":                  prediction.Backend,
		"labels":                   prediction.Labels,
		"observed_label":           prediction.Label,
		"observed_logits":          prediction.Logits,
	}

	encoder := json.NewEncoder(os.Stdout)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(payload); err != nil {
		log.Fatalf("encode output: %v", err)
	}
}
