package torchscript

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// BundleMetadata describes the exported TorchScript bundle layout.
type BundleMetadata struct {
	Name                string   `json:"name"`
	Source              string   `json:"source"`
	ModelID             string   `json:"model_id"`
	Task                string   `json:"task"`
	GeneratedAt         string   `json:"generated_at"`
	TransformersVersion string   `json:"transformers_version"`
	TorchVersion        string   `json:"torch_version"`
	Artifact            string   `json:"artifact"`
	Labels              []string `json:"labels"`
	PadTokenID          int      `json:"pad_token_id"`
	SequenceLength      int      `json:"sequence_length"`
}

// LoadBundleMetadata loads TorchScript bundle metadata from a directory.
func LoadBundleMetadata(bundleDir string) (BundleMetadata, error) {
	metadataPath := filepath.Join(bundleDir, "metadata.json")
	raw, err := os.ReadFile(metadataPath)
	if err != nil {
		return BundleMetadata{}, fmt.Errorf("read torchscript metadata: %w", err)
	}

	var metadata BundleMetadata
	if err := json.Unmarshal(raw, &metadata); err != nil {
		return BundleMetadata{}, fmt.Errorf("decode torchscript metadata: %w", err)
	}

	if metadata.ModelID == "" {
		return BundleMetadata{}, fmt.Errorf("decode torchscript metadata: missing model id")
	}

	if metadata.Artifact == "" {
		return BundleMetadata{}, fmt.Errorf("decode torchscript metadata: missing artifact")
	}

	if metadata.SequenceLength <= 0 {
		return BundleMetadata{}, fmt.Errorf("decode torchscript metadata: invalid sequence length")
	}

	return metadata, nil
}
