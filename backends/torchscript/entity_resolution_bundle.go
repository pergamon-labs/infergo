package torchscript

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/pergamon-labs/infergo/backends/torchscript/binding"
)

const (
	entityResolutionBundleFormat = "infergo-torchscript-bridge"
	entityResolutionFamily       = "numeric-feature-scoring"
	entityResolutionTask         = "entity-resolution-scoring"
)

// EntityResolutionBundleMetadata describes the bundle layout for the internal
// family-2 numeric-feature TorchScript bridge.
type EntityResolutionBundleMetadata struct {
	BundleFormat  string `json:"bundle_format"`
	BundleVersion string `json:"bundle_version"`
	Family        string `json:"family"`
	Task          string `json:"task"`
	Backend       string `json:"backend"`
	Artifact      string `json:"artifact"`
	ModelID       string `json:"model_id"`
	ProfileKind   string `json:"profile_kind,omitempty"`
	Inputs        struct {
		VectorSize  int `json:"vector_size"`
		MessageSize int `json:"message_size"`
	} `json:"inputs"`
	Outputs struct {
		Kind string `json:"kind"`
	} `json:"outputs"`
}

// EntityResolutionBundle loads and validates the family-2 bridge bundle.
type EntityResolutionBundle struct {
	metadata EntityResolutionBundleMetadata
	module   *binding.Module
}

// LoadEntityResolutionBundleMetadata loads and validates family-2 bundle metadata.
func LoadEntityResolutionBundleMetadata(bundleDir string) (EntityResolutionBundleMetadata, error) {
	metadataPath := filepath.Join(bundleDir, "metadata.json")
	raw, err := os.ReadFile(metadataPath)
	if err != nil {
		return EntityResolutionBundleMetadata{}, fmt.Errorf("read entity-resolution metadata: %w", err)
	}

	var metadata EntityResolutionBundleMetadata
	if err := json.Unmarshal(raw, &metadata); err != nil {
		return EntityResolutionBundleMetadata{}, fmt.Errorf("decode entity-resolution metadata: %w", err)
	}

	if metadata.BundleFormat != entityResolutionBundleFormat {
		return EntityResolutionBundleMetadata{}, fmt.Errorf("decode entity-resolution metadata: unsupported bundle_format %q", metadata.BundleFormat)
	}
	if metadata.BundleVersion == "" {
		return EntityResolutionBundleMetadata{}, fmt.Errorf("decode entity-resolution metadata: missing bundle version")
	}
	if metadata.Family != entityResolutionFamily {
		return EntityResolutionBundleMetadata{}, fmt.Errorf("decode entity-resolution metadata: unsupported family %q", metadata.Family)
	}
	if metadata.Task != entityResolutionTask {
		return EntityResolutionBundleMetadata{}, fmt.Errorf("decode entity-resolution metadata: unsupported task %q", metadata.Task)
	}
	if metadata.Backend != (Backend{}).Name() {
		return EntityResolutionBundleMetadata{}, fmt.Errorf("decode entity-resolution metadata: unsupported backend %q", metadata.Backend)
	}
	if metadata.Artifact == "" {
		return EntityResolutionBundleMetadata{}, fmt.Errorf("decode entity-resolution metadata: missing artifact")
	}
	if metadata.ModelID == "" {
		return EntityResolutionBundleMetadata{}, fmt.Errorf("decode entity-resolution metadata: missing model id")
	}
	if metadata.Inputs.VectorSize <= 0 {
		return EntityResolutionBundleMetadata{}, fmt.Errorf("decode entity-resolution metadata: invalid vector_size")
	}
	if metadata.Inputs.MessageSize <= 0 {
		return EntityResolutionBundleMetadata{}, fmt.Errorf("decode entity-resolution metadata: invalid message_size")
	}
	if metadata.Outputs.Kind != "score_vector" {
		return EntityResolutionBundleMetadata{}, fmt.Errorf("decode entity-resolution metadata: unsupported outputs.kind %q", metadata.Outputs.Kind)
	}

	return metadata, nil
}

// LoadEntityResolutionBundle loads the experimental family-2 bundle from disk.
func LoadEntityResolutionBundle(bundleDir string) (*EntityResolutionBundle, error) {
	metadata, err := LoadEntityResolutionBundleMetadata(bundleDir)
	if err != nil {
		return nil, err
	}

	modulePath := filepath.Join(bundleDir, metadata.Artifact)
	module, err := binding.LoadModule(modulePath)
	if err != nil {
		return nil, fmt.Errorf("load entity-resolution torchscript module: %w", err)
	}

	return &EntityResolutionBundle{
		metadata: metadata,
		module:   module,
	}, nil
}

// PredictBatch returns one score per input vector row.
func (b *EntityResolutionBundle) PredictBatch(vectors [][]float64, message []float64) ([]float64, error) {
	if b == nil || b.module == nil {
		return nil, fmt.Errorf("predict batch: entity-resolution bundle is not initialized")
	}
	if len(vectors) == 0 {
		return nil, fmt.Errorf("predict batch: vectors batch must not be empty")
	}
	if len(message) != b.metadata.Inputs.MessageSize {
		return nil, fmt.Errorf("predict batch: message length mismatch (%d != %d)", len(message), b.metadata.Inputs.MessageSize)
	}

	for i := range vectors {
		if len(vectors[i]) != b.metadata.Inputs.VectorSize {
			return nil, fmt.Errorf("predict batch: vector %d length mismatch (%d != %d)", i, len(vectors[i]), b.metadata.Inputs.VectorSize)
		}
	}

	return b.module.ForwardFeatureScoring(vectors, message)
}

// Metadata returns a copy of the validated bundle metadata.
func (b *EntityResolutionBundle) Metadata() EntityResolutionBundleMetadata {
	return b.metadata
}

// ModelID returns the source model identifier for the bundle.
func (b *EntityResolutionBundle) ModelID() string {
	return b.metadata.ModelID
}

// Close releases native resources held by the bundle.
func (b *EntityResolutionBundle) Close() error {
	if b == nil || b.module == nil {
		return nil
	}

	err := b.module.Close()
	b.module = nil
	return err
}
