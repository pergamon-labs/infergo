package infer_test

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/pergamon-labs/infergo/infer"
)

func ExampleLoadTextBundle() {
	bundle, err := infer.LoadTextBundle("../testdata/native/text-classification/infergo-basic-sst2-embedding-masked-avg-pool")
	if err != nil {
		log.Fatal(err)
	}
	defer bundle.Close()

	fmt.Println(bundle.ModelID())
	fmt.Println(bundle.SupportsTokenizedInput())
	// Output:
	// infergo/basic-sst2-sentiment
	// true
}

func ExampleTextBundle_PredictText() {
	bundleDir, err := writeExampleAlphaTextBundle()
	if err != nil {
		log.Fatal(err)
	}
	defer os.RemoveAll(bundleDir)

	bundle, err := infer.LoadTextBundle(bundleDir)
	if err != nil {
		log.Fatal(err)
	}
	defer bundle.Close()

	result, err := bundle.PredictText("This product is excellent and reliable.")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(result.ModelID)
	fmt.Println(len(result.Logits))
	// Output:
	// example/distilbert-sst2-alpha
	// 2
}

func ExampleLoadTokenClassifier() {
	classifier, err := infer.LoadTokenClassifier("../testdata/native/token-classification/infergo-basic-french-ner-windowed-embedding-linear")
	if err != nil {
		log.Fatal(err)
	}
	defer classifier.Close()

	result, err := classifier.Predict(infer.TokenInput{
		InputIDs: []int64{0, 1, 2, 3, 4},
	})
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(result.Backend)
	fmt.Println(len(result.TokenLabels))
	// Output:
	// bionet
	// 5
}

func writeExampleAlphaTextBundle() (string, error) {
	sourceDir := filepath.Clean("../testdata/native/text-classification/distilbert-sst2-embedding-masked-avg-pool")
	bundleDir, err := os.MkdirTemp("", "infergo-example-alpha-bundle-")
	if err != nil {
		return "", err
	}

	if err := copyExampleFile(filepath.Join(sourceDir, "model.gob"), filepath.Join(bundleDir, "model.gob")); err != nil {
		return "", err
	}
	if err := copyExampleFile(filepath.Join(sourceDir, "embeddings.gob"), filepath.Join(bundleDir, "embeddings.gob")); err != nil {
		return "", err
	}

	if err := writeExampleJSON(filepath.Join(bundleDir, "labels.json"), map[string]any{
		"labels": []string{"NEGATIVE", "POSITIVE"},
	}); err != nil {
		return "", err
	}

	tokenizerDir := filepath.Join(bundleDir, "tokenizer")
	if err := os.MkdirAll(tokenizerDir, 0o755); err != nil {
		return "", err
	}
	if err := writeExampleJSON(filepath.Join(tokenizerDir, "manifest.json"), map[string]any{
		"kind":                "hf-tokenizer-json",
		"raw_text_supported":  true,
		"pair_text_supported": false,
		"files": map[string]string{
			"tokenizer_json":   "tokenizer.json",
			"tokenizer_config": "tokenizer_config.json",
		},
	}); err != nil {
		return "", err
	}
	if err := writeExampleJSON(filepath.Join(tokenizerDir, "tokenizer_config.json"), map[string]any{
		"do_lower_case": true,
	}); err != nil {
		return "", err
	}
	if err := writeExampleJSON(filepath.Join(tokenizerDir, "tokenizer.json"), map[string]any{
		"version": "1.0",
		"normalizer": map[string]any{
			"type":                 "BertNormalizer",
			"clean_text":           true,
			"handle_chinese_chars": true,
			"strip_accents":        nil,
			"lowercase":            true,
		},
		"pre_tokenizer": map[string]any{
			"type": "BertPreTokenizer",
		},
		"post_processor": map[string]any{
			"type": "TemplateProcessing",
			"single": []map[string]any{
				{"SpecialToken": map[string]any{"id": "[CLS]", "type_id": 0}},
				{"Sequence": map[string]any{"id": "A", "type_id": 0}},
				{"SpecialToken": map[string]any{"id": "[SEP]", "type_id": 0}},
			},
			"pair": []map[string]any{
				{"SpecialToken": map[string]any{"id": "[CLS]", "type_id": 0}},
				{"Sequence": map[string]any{"id": "A", "type_id": 0}},
				{"SpecialToken": map[string]any{"id": "[SEP]", "type_id": 0}},
				{"Sequence": map[string]any{"id": "B", "type_id": 1}},
				{"SpecialToken": map[string]any{"id": "[SEP]", "type_id": 1}},
			},
			"special_tokens": map[string]any{
				"[CLS]": map[string]any{"id": "[CLS]", "ids": []int{101}, "tokens": []string{"[CLS]"}},
				"[SEP]": map[string]any{"id": "[SEP]", "ids": []int{102}, "tokens": []string{"[SEP]"}},
			},
		},
		"decoder": map[string]any{
			"type":    "WordPiece",
			"prefix":  "##",
			"cleanup": true,
		},
		"model": map[string]any{
			"type":                      "WordPiece",
			"unk_token":                 "[UNK]",
			"continuing_subword_prefix": "##",
			"max_input_chars_per_word":  100,
			"vocab": map[string]int{
				"[PAD]":     0,
				"[UNK]":     100,
				"[CLS]":     101,
				"[SEP]":     102,
				"this":      2023,
				"product":   4031,
				"is":        2003,
				"excellent": 6581,
				"and":       1998,
				"reliable":  10539,
				".":         1012,
			},
		},
	}); err != nil {
		return "", err
	}

	if err := writeExampleJSON(filepath.Join(bundleDir, "metadata.json"), map[string]any{
		"bundle_format":    "infergo-native",
		"bundle_version":   "1.0",
		"family":           "encoder-text-classification",
		"task":             "text-classification",
		"backend":          "bionet",
		"backend_artifact": "model.gob",
		"model_id":         "example/distilbert-sst2-alpha",
		"source": map[string]any{
			"framework": "pytorch",
			"ecosystem": "transformers",
		},
		"inputs": map[string]any{
			"raw_text_supported":        true,
			"pair_text_supported":       false,
			"tokenized_input_supported": true,
			"max_sequence_length":       128,
		},
		"tokenizer": map[string]any{
			"manifest": "tokenizer/manifest.json",
		},
		"outputs": map[string]any{
			"kind":            "label_logits",
			"labels_artifact": "labels.json",
			"positive_label":  "POSITIVE",
			"negative_label":  "NEGATIVE",
			"threshold":       0.5,
		},
		"backend_config": map[string]any{
			"feature_mode": "embedding-masked-avg-pool",
			"feature_token_ids": []int{
				101, 102, 1012, 1998, 2003, 2023, 4031, 6581, 10539,
			},
			"embedding_artifact": "embeddings.gob",
		},
		"created_at": "2026-04-08T00:00:00Z",
		"created_by": map[string]any{
			"tool":    "infergo-export",
			"version": "0.1.0-alpha",
		},
	}); err != nil {
		return "", err
	}

	return bundleDir, nil
}

func writeExampleJSON(path string, value any) error {
	raw, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return err
	}
	raw = append(raw, '\n')
	return os.WriteFile(path, raw, 0o644)
}

func copyExampleFile(src, dst string) error {
	raw, err := os.ReadFile(src)
	if err != nil {
		return err
	}
	return os.WriteFile(dst, raw, 0o644)
}
