package bionet_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/pergamon-labs/infergo/backends/bionet"
	"github.com/pergamon-labs/infergo/internal/parity"
)

func TestLoadTextClassificationBundleAlphaContractPredict(t *testing.T) {
	t.Parallel()

	bundleDir := writeAlphaTextClassificationBundleFixture(t, alphaTextBundleFixtureOptions{
		rawTextSupported: true,
	})

	bundle, err := bionet.LoadTextClassificationBundle(bundleDir)
	if err != nil {
		t.Fatalf("LoadTextClassificationBundle() error = %v", err)
	}

	reference, err := parity.LoadTransformersTextClassificationReference("../../testdata/reference/text-classification/distilbert-sst2-reference.json")
	if err != nil {
		t.Fatalf("LoadTransformersTextClassificationReference() error = %v", err)
	}

	item := reference.Cases[0]
	logitsBatch, err := bundle.PredictBatch([][]int64{intsToInt64(item.InputIDs)}, [][]int64{intsToInt64(item.AttentionMask)})
	if err != nil {
		t.Fatalf("PredictBatch() error = %v", err)
	}

	if got, want := bundle.ModelID(), reference.ModelID; got != want {
		t.Fatalf("bundle.ModelID() = %q, want %q", got, want)
	}
	if got, want := bundle.Labels()[argMax(logitsBatch[0])], item.ExpectedLabel; got != want {
		t.Fatalf("predicted label = %q, want %q", got, want)
	}
}

func TestLoadTextClassificationBundleAlphaContractMissingLabels(t *testing.T) {
	t.Parallel()

	bundleDir := writeAlphaTextClassificationBundleFixture(t, alphaTextBundleFixtureOptions{
		skipLabelsArtifact: true,
	})

	_, err := bionet.LoadTextClassificationBundle(bundleDir)
	if err == nil || !strings.Contains(err.Error(), "labels artifact") {
		t.Fatalf("LoadTextClassificationBundle() error = %v, want labels artifact failure", err)
	}
}

func TestLoadTextClassificationBundleAlphaContractMissingTokenizerManifestForRawText(t *testing.T) {
	t.Parallel()

	bundleDir := writeAlphaTextClassificationBundleFixture(t, alphaTextBundleFixtureOptions{
		rawTextSupported:      true,
		skipTokenizerManifest: true,
	})

	_, err := bionet.LoadTextClassificationBundle(bundleDir)
	if err == nil || !strings.Contains(err.Error(), "missing tokenizer manifest") {
		t.Fatalf("LoadTextClassificationBundle() error = %v, want tokenizer manifest failure", err)
	}
}

func TestLoadTextClassificationBundleAlphaContractRejectsUnsupportedTokenizerKind(t *testing.T) {
	t.Parallel()

	bundleDir := writeAlphaTextClassificationBundleFixture(t, alphaTextBundleFixtureOptions{
		rawTextSupported: true,
		tokenizerKind:    "wordpiece",
	})

	_, err := bionet.LoadTextClassificationBundle(bundleDir)
	if err == nil || !strings.Contains(err.Error(), `alpha supports only "hf-tokenizer-json" manifests`) {
		t.Fatalf("LoadTextClassificationBundle() error = %v, want unsupported tokenizer kind failure", err)
	}
}

func TestLoadTextClassificationBundleAlphaContractRejectsUnsupportedMajor(t *testing.T) {
	t.Parallel()

	bundleDir := writeAlphaTextClassificationBundleFixture(t, alphaTextBundleFixtureOptions{
		bundleVersion: "2.0",
	})

	_, err := bionet.LoadTextClassificationBundle(bundleDir)
	if err == nil || !strings.Contains(err.Error(), "unsupported bundle version major 2") {
		t.Fatalf("LoadTextClassificationBundle() error = %v, want unsupported major version failure", err)
	}
}

func TestLoadTextClassificationBundleAlphaContractRejectsUnsupportedMinor(t *testing.T) {
	t.Parallel()

	bundleDir := writeAlphaTextClassificationBundleFixture(t, alphaTextBundleFixtureOptions{
		bundleVersion: "1.1",
	})

	_, err := bionet.LoadTextClassificationBundle(bundleDir)
	if err == nil || !strings.Contains(err.Error(), "current alpha supports only 1.0 bundles") {
		t.Fatalf("LoadTextClassificationBundle() error = %v, want unsupported minor version failure", err)
	}
}

func TestLoadTextClassificationBundleAlphaContractRejectsLabelDimensionMismatch(t *testing.T) {
	t.Parallel()

	bundleDir := writeAlphaTextClassificationBundleFixture(t, alphaTextBundleFixtureOptions{
		labels: []string{"NEGATIVE", "POSITIVE", "MAYBE"},
	})

	_, err := bionet.LoadTextClassificationBundle(bundleDir)
	if err == nil || !strings.Contains(err.Error(), "labels count") {
		t.Fatalf("LoadTextClassificationBundle() error = %v, want labels count failure", err)
	}
}

func TestLoadTextClassificationRawTextEncoderAlphaContract(t *testing.T) {
	t.Parallel()

	bundleDir := writeAlphaTextClassificationBundleFixture(t, alphaTextBundleFixtureOptions{
		rawTextSupported: true,
	})

	encoder, err := bionet.LoadTextClassificationRawTextEncoder(bundleDir)
	if err != nil {
		t.Fatalf("LoadTextClassificationRawTextEncoder() error = %v", err)
	}

	reference, err := parity.LoadTransformersTextClassificationReference("../../testdata/reference/text-classification/distilbert-sst2-reference.json")
	if err != nil {
		t.Fatalf("LoadTransformersTextClassificationReference() error = %v", err)
	}

	item := reference.Cases[0]
	inputIDs, attentionMask, err := encoder.Encode(item.Text, "")
	if err != nil {
		t.Fatalf("encoder.Encode() error = %v", err)
	}

	if got, want := inputIDs, intsToInt64(item.InputIDs); !equalInt64s(got, want) {
		t.Fatalf("encoded input ids = %v, want %v", got, want)
	}
	if got, want := attentionMask, intsToInt64(item.AttentionMask); !equalInt64s(got, want) {
		t.Fatalf("encoded attention mask = %v, want %v", got, want)
	}
}

type alphaTextBundleFixtureOptions struct {
	bundleVersion         string
	labels                []string
	rawTextSupported      bool
	tokenizerKind         string
	skipLabelsArtifact    bool
	skipTokenizerManifest bool
}

func writeAlphaTextClassificationBundleFixture(t *testing.T, opts alphaTextBundleFixtureOptions) string {
	t.Helper()

	if opts.bundleVersion == "" {
		opts.bundleVersion = "1.0"
	}
	if len(opts.labels) == 0 {
		opts.labels = []string{"NEGATIVE", "POSITIVE"}
	}

	sourceDir := filepath.Clean("../../testdata/native/text-classification/distilbert-sst2-embedding-masked-avg-pool")
	oldMetadataRaw, err := os.ReadFile(filepath.Join(sourceDir, "metadata.json"))
	if err != nil {
		t.Fatalf("ReadFile(old metadata) error = %v", err)
	}

	var oldMetadata struct {
		FeatureMode       string `json:"feature_mode"`
		FeatureTokenIDs   []int  `json:"feature_token_ids"`
		EmbeddingArtifact string `json:"embedding_artifact"`
		Artifact          string `json:"artifact"`
	}
	if err := json.Unmarshal(oldMetadataRaw, &oldMetadata); err != nil {
		t.Fatalf("json.Unmarshal(old metadata) error = %v", err)
	}

	bundleDir := t.TempDir()
	copyFileForTest(t, filepath.Join(sourceDir, "model.gob"), filepath.Join(bundleDir, "model.gob"))
	copyFileForTest(t, filepath.Join(sourceDir, "embeddings.gob"), filepath.Join(bundleDir, "embeddings.gob"))

	if !opts.skipLabelsArtifact {
		writeJSONForTest(t, filepath.Join(bundleDir, "labels.json"), map[string]any{
			"labels": opts.labels,
		})
	}

	if opts.rawTextSupported && !opts.skipTokenizerManifest {
		tokenizerDir := filepath.Join(bundleDir, "tokenizer")
		if err := os.MkdirAll(tokenizerDir, 0o755); err != nil {
			t.Fatalf("MkdirAll(tokenizer) error = %v", err)
		}
		if opts.tokenizerKind == "" {
			opts.tokenizerKind = "hf-tokenizer-json"
		}

		writeJSONForTest(t, filepath.Join(tokenizerDir, "manifest.json"), map[string]any{
			"kind":                opts.tokenizerKind,
			"raw_text_supported":  true,
			"pair_text_supported": false,
			"files": map[string]string{
				"tokenizer_json":     "tokenizer.json",
				"tokenizer_config":   "tokenizer_config.json",
				"special_tokens_map": "special_tokens_map.json",
			},
		})
		writeTokenizerJSONForFixture(t, filepath.Join(tokenizerDir, "tokenizer.json"))
		writeJSONForTest(t, filepath.Join(tokenizerDir, "tokenizer_config.json"), map[string]any{
			"do_lower_case": true,
		})
		writeJSONForTest(t, filepath.Join(tokenizerDir, "special_tokens_map.json"), map[string]any{
			"cls_token": "[CLS]",
			"sep_token": "[SEP]",
		})
	}

	metadata := map[string]any{
		"bundle_format":    "infergo-native",
		"bundle_version":   opts.bundleVersion,
		"family":           "encoder-text-classification",
		"task":             "text-classification",
		"backend":          "bionet",
		"backend_artifact": oldMetadata.Artifact,
		"model_id":         "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
		"source": map[string]any{
			"framework":      "pytorch",
			"ecosystem":      "transformers",
			"weights_format": "safetensors",
		},
		"inputs": map[string]any{
			"raw_text_supported":        opts.rawTextSupported,
			"pair_text_supported":       false,
			"tokenized_input_supported": true,
			"max_sequence_length":       128,
		},
		"outputs": map[string]any{
			"kind":            "label_logits",
			"labels_artifact": "labels.json",
			"positive_label":  "POSITIVE",
		},
		"backend_config": map[string]any{
			"feature_mode":       oldMetadata.FeatureMode,
			"feature_token_ids":  oldMetadata.FeatureTokenIDs,
			"embedding_artifact": oldMetadata.EmbeddingArtifact,
		},
		"created_at": "2026-04-08T00:00:00Z",
		"created_by": map[string]any{
			"tool":    "infergo-export",
			"version": "0.1.0-alpha",
		},
	}

	if opts.rawTextSupported {
		metadata["tokenizer"] = map[string]any{
			"manifest": "tokenizer/manifest.json",
		}
	}

	writeJSONForTest(t, filepath.Join(bundleDir, "metadata.json"), metadata)
	return bundleDir
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

func writeTokenizerJSONForFixture(t *testing.T, path string) {
	t.Helper()

	reference, err := parity.LoadTransformersTextClassificationReference("../../testdata/reference/text-classification/distilbert-sst2-reference.json")
	if err != nil {
		t.Fatalf("LoadTransformersTextClassificationReference() error = %v", err)
	}

	item := reference.Cases[0]
	vocab := map[string]int{
		"[PAD]": 0,
		"[UNK]": 100,
		"[CLS]": 101,
		"[SEP]": 102,
	}
	for i, token := range item.Tokens {
		vocab[token] = item.InputIDs[i]
	}

	writeJSONForTest(t, path, map[string]any{
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
			"vocab":                     vocab,
		},
	})
}

func equalInt64s(left, right []int64) bool {
	if len(left) != len(right) {
		return false
	}
	for i := range left {
		if left[i] != right[i] {
			return false
		}
	}
	return true
}
