package torchscript

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLoadEntityResolutionBundleMetadata(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	path := filepath.Join(dir, "metadata.json")
	payload := `{
  "bundle_format": "infergo-torchscript-bridge",
  "bundle_version": "1.0",
  "family": "numeric-feature-scoring",
  "task": "entity-resolution-scoring",
  "backend": "torchscript",
  "artifact": "model.torchscript.pt",
  "model_id": "pergamon/entres-individual",
  "profile_kind": "individual",
  "inputs": {
    "vector_size": 128,
    "message_size": 128
  },
  "outputs": {
    "kind": "score_vector"
  }
}`

	if err := os.WriteFile(path, []byte(payload), 0o644); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	metadata, err := LoadEntityResolutionBundleMetadata(dir)
	if err != nil {
		t.Fatalf("LoadEntityResolutionBundleMetadata() error = %v", err)
	}

	if metadata.ModelID != "pergamon/entres-individual" {
		t.Fatalf("unexpected model id %q", metadata.ModelID)
	}
	if metadata.Inputs.VectorSize != 128 || metadata.Inputs.MessageSize != 128 {
		t.Fatalf("unexpected input sizes: %+v", metadata.Inputs)
	}
}

func TestLoadEntityResolutionBundleMetadataRejectsInvalidFamily(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	path := filepath.Join(dir, "metadata.json")
	payload := `{
  "bundle_format": "infergo-torchscript-bridge",
  "bundle_version": "1.0",
  "family": "wrong-family",
  "task": "entity-resolution-scoring",
  "backend": "torchscript",
  "artifact": "model.torchscript.pt",
  "model_id": "pergamon/entres-individual",
  "inputs": {
    "vector_size": 128,
    "message_size": 128
  },
  "outputs": {
    "kind": "score_vector"
  }
}`

	if err := os.WriteFile(path, []byte(payload), 0o644); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	_, err := LoadEntityResolutionBundleMetadata(dir)
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "unsupported family") {
		t.Fatalf("unexpected error %v", err)
	}
}
