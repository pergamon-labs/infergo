package torchscript

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadBundleMetadata(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	path := filepath.Join(dir, "metadata.json")
	payload := `{
  "name": "bundle",
  "source": "test",
  "model_id": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
  "task": "text-classification",
  "generated_at": "2026-03-23T00:00:00Z",
  "transformers_version": "5.3.0",
  "torch_version": "2.10.0",
  "artifact": "model.torchscript.pt",
  "labels": ["NEGATIVE", "POSITIVE"],
  "pad_token_id": 0,
  "sequence_length": 9
}`

	if err := os.WriteFile(path, []byte(payload), 0o644); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	metadata, err := LoadBundleMetadata(dir)
	if err != nil {
		t.Fatalf("LoadBundleMetadata() error = %v", err)
	}

	if metadata.ModelID != "distilbert/distilbert-base-uncased-finetuned-sst-2-english" {
		t.Fatalf("unexpected model id %q", metadata.ModelID)
	}

	if metadata.SequenceLength != 9 {
		t.Fatalf("unexpected sequence length %d", metadata.SequenceLength)
	}
}
