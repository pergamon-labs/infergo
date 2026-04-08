package infer_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/pergamon-labs/infergo/infer"
)

func TestLoadTextBundlePredictText(t *testing.T) {
	t.Parallel()

	bundleDir := writeAlphaTextBundleFixture(t, alphaTextBundleFixtureOptions{
		rawTextSupported: true,
	})

	bundle, err := infer.LoadTextBundle(bundleDir)
	if err != nil {
		t.Fatalf("LoadTextBundle() error = %v", err)
	}
	defer bundle.Close()

	if !bundle.SupportsRawText() {
		t.Fatalf("SupportsRawText() = false, want true")
	}
	if bundle.SupportsPairText() {
		t.Fatalf("SupportsPairText() = true, want false")
	}

	prediction, err := bundle.PredictText("This product is excellent and reliable.")
	if err != nil {
		t.Fatalf("PredictText() error = %v", err)
	}
	if prediction.Label == "" {
		t.Fatalf("prediction.Label = empty, want non-empty")
	}
}

func TestLoadTextBundlePredictTextPair(t *testing.T) {
	t.Parallel()

	bundleDir := writeAlphaTextBundleFixture(t, alphaTextBundleFixtureOptions{
		rawTextSupported:  true,
		pairTextSupported: true,
	})

	bundle, err := infer.LoadTextBundle(bundleDir)
	if err != nil {
		t.Fatalf("LoadTextBundle() error = %v", err)
	}
	defer bundle.Close()

	if !bundle.SupportsPairText() {
		t.Fatalf("SupportsPairText() = false, want true")
	}

	prediction, err := bundle.PredictTextPair(
		"This product is excellent and reliable.",
		"This product is reliable.",
	)
	if err != nil {
		t.Fatalf("PredictTextPair() error = %v", err)
	}
	if prediction.Label == "" {
		t.Fatalf("prediction.Label = empty, want non-empty")
	}
}

func TestLoadTextBundleRejectsPairWhenUnsupported(t *testing.T) {
	t.Parallel()

	bundleDir := writeAlphaTextBundleFixture(t, alphaTextBundleFixtureOptions{
		rawTextSupported: true,
	})

	bundle, err := infer.LoadTextBundle(bundleDir)
	if err != nil {
		t.Fatalf("LoadTextBundle() error = %v", err)
	}
	defer bundle.Close()

	if _, err := bundle.PredictTextPair("a", "b"); err == nil {
		t.Fatalf("PredictTextPair() error = nil, want unsupported pair-text failure")
	}
}

type alphaTextBundleFixtureOptions struct {
	rawTextSupported  bool
	pairTextSupported bool
}

func writeAlphaTextBundleFixture(t *testing.T, opts alphaTextBundleFixtureOptions) string {
	t.Helper()

	sourceDir := filepath.Clean("../testdata/native/text-classification/distilbert-sst2-embedding-masked-avg-pool")
	bundleDir := t.TempDir()
	copyFileForTest(t, filepath.Join(sourceDir, "model.gob"), filepath.Join(bundleDir, "model.gob"))
	copyFileForTest(t, filepath.Join(sourceDir, "embeddings.gob"), filepath.Join(bundleDir, "embeddings.gob"))

	writeJSONForTest(t, filepath.Join(bundleDir, "labels.json"), map[string]any{
		"labels": []string{"NEGATIVE", "POSITIVE"},
	})

	if opts.rawTextSupported {
		tokenizerDir := filepath.Join(bundleDir, "tokenizer")
		if err := os.MkdirAll(tokenizerDir, 0o755); err != nil {
			t.Fatalf("MkdirAll(tokenizer) error = %v", err)
		}

		writeJSONForTest(t, filepath.Join(tokenizerDir, "manifest.json"), map[string]any{
			"kind":                "hf-tokenizer-json",
			"raw_text_supported":  true,
			"pair_text_supported": opts.pairTextSupported,
			"files": map[string]string{
				"tokenizer_json":   "tokenizer.json",
				"tokenizer_config": "tokenizer_config.json",
			},
		})
		writeJSONForTest(t, filepath.Join(tokenizerDir, "tokenizer_config.json"), map[string]any{
			"do_lower_case": true,
		})
		writeJSONForTest(t, filepath.Join(tokenizerDir, "tokenizer.json"), map[string]any{
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
			"post_processor": buildTokenizerPostProcessor(),
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
		})
	}

	writeJSONForTest(t, filepath.Join(bundleDir, "metadata.json"), map[string]any{
		"bundle_format":    "infergo-native",
		"bundle_version":   "1.0",
		"family":           "encoder-text-classification",
		"task":             "text-classification",
		"backend":          "bionet",
		"backend_artifact": "model.gob",
		"model_id":         "example/distilbert-sst2-alpha",
		"source": map[string]any{
			"framework":      "pytorch",
			"ecosystem":      "transformers",
			"weights_format": "safetensors",
		},
		"inputs": map[string]any{
			"raw_text_supported":        opts.rawTextSupported,
			"pair_text_supported":       opts.pairTextSupported,
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
	})

	return bundleDir
}

func buildTokenizerPostProcessor() map[string]any {
	return map[string]any{
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
	}
}

func writeJSONForTest(t *testing.T, path string, value any) {
	t.Helper()

	raw, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		t.Fatalf("json.MarshalIndent(%s) error = %v", path, err)
	}
	raw = append(raw, '\n')
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatalf("WriteFile(%s) error = %v", path, err)
	}
}

func copyFileForTest(t *testing.T, src, dst string) {
	t.Helper()

	raw, err := os.ReadFile(src)
	if err != nil {
		t.Fatalf("ReadFile(%s) error = %v", src, err)
	}
	if err := os.WriteFile(dst, raw, 0o644); err != nil {
		t.Fatalf("WriteFile(%s) error = %v", dst, err)
	}
}
