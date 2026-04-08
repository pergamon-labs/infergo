package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/pergamon-labs/infergo/backends/bionet"
	"github.com/pergamon-labs/infergo/internal/nativebundlegen"
)

func TestRunTemplateSingle(t *testing.T) {
	t.Parallel()

	outputPath := filepath.Join(t.TempDir(), "single.json")
	if err := runTemplate([]string{"-kind", "single", "-out", outputPath}); err != nil {
		t.Fatalf("runTemplate(single) error = %v", err)
	}

	var payload inputTemplate
	raw := readFileForTest(t, outputPath)
	if err := json.Unmarshal(raw, &payload); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}

	if got := len(payload.Cases); got < 3 {
		t.Fatalf("single template case count = %d, want >= 3", got)
	}
	if payload.Cases[0].TextPair != "" {
		t.Fatal("single template unexpectedly populated text_pair")
	}
}

func TestRunTemplatePair(t *testing.T) {
	t.Parallel()

	outputPath := filepath.Join(t.TempDir(), "pair.json")
	if err := runTemplate([]string{"-kind", "pair", "-out", outputPath}); err != nil {
		t.Fatalf("runTemplate(pair) error = %v", err)
	}

	var payload inputTemplate
	raw := readFileForTest(t, outputPath)
	if err := json.Unmarshal(raw, &payload); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}

	if payload.Cases[0].TextPair == "" {
		t.Fatal("pair template should include text_pair")
	}
}

func TestBuildAlphaMetadata(t *testing.T) {
	t.Parallel()

	legacy := bionet.TextClassificationBundleMetadata{
		ModelID:           "textattack/bert-base-uncased-MRPC",
		Task:              "text-classification",
		Artifact:          nativebundlegen.DefaultArtifactName,
		EmbeddingArtifact: nativebundlegen.DefaultEmbeddingName,
		Labels:            []string{"not_duplicate", "duplicate"},
		FeatureMode:       bionet.TextClassificationFeatureModeEmbeddingMaskedAvgPool,
		FeatureTokenIDs:   []int{101, 102, 2009},
	}

	manifest := tokenizerManifest{
		Kind:              "hf-tokenizer-json",
		RawTextSupported:  true,
		PairTextSupported: true,
	}

	metadata := buildAlphaMetadata(
		"textattack/bert-base-uncased-MRPC",
		"textattack/bert-base-uncased-MRPC",
		"1.0",
		128,
		legacy,
		manifest,
		"duplicate",
		"not_duplicate",
	)

	if metadata.CreatedBy.Tool != "infergo-export" {
		t.Fatalf("CreatedBy.Tool = %q, want infergo-export", metadata.CreatedBy.Tool)
	}
	if !metadata.Inputs.RawTextSupported {
		t.Fatal("expected raw_text_supported=true")
	}
	if !metadata.Inputs.PairTextSupported {
		t.Fatal("expected pair_text_supported=true")
	}
	if metadata.Outputs.Threshold == nil {
		t.Fatal("expected binary threshold to be set")
	}
}

func readFileForTest(t *testing.T, path string) []byte {
	t.Helper()

	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("os.ReadFile(%q) error = %v", path, err)
	}
	return raw
}
