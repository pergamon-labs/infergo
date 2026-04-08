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

type alphaTextBundleFixtureOptions struct {
	bundleVersion         string
	labels                []string
	rawTextSupported      bool
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

		writeJSONForTest(t, filepath.Join(tokenizerDir, "manifest.json"), map[string]any{
			"kind":                "hf-tokenizer-json",
			"raw_text_supported":  true,
			"pair_text_supported": false,
			"files": map[string]string{
				"tokenizer_json":     "tokenizer.json",
				"tokenizer_config":   "tokenizer_config.json",
				"special_tokens_map": "special_tokens_map.json",
			},
		})
		writeJSONForTest(t, filepath.Join(tokenizerDir, "tokenizer.json"), map[string]any{
			"version": "test",
		})
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
