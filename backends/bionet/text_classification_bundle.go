package bionet

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/pergamon-labs/infergo/backends/bionet/runtime/tensor"
)

const (
	// TextClassificationFeatureModeTokenIDBag counts active token ids into a
	// fixed feature vector before running the BIOnet runtime module.
	TextClassificationFeatureModeTokenIDBag = "token-id-bag"
)

// TextClassificationPredictor is the narrow prediction surface used by the
// parity harness for InferGo-native text classification bundles.
type TextClassificationPredictor interface {
	PredictBatch(inputIDs, attentionMasks [][]int64) ([][]float64, error)
	Labels() []string
	ModelID() string
	Close() error
}

// TextClassificationBundleMetadata describes an InferGo-native text
// classification bundle layout built on top of the BIOnet runtime.
type TextClassificationBundleMetadata struct {
	Name            string   `json:"name"`
	Source          string   `json:"source"`
	ModelID         string   `json:"model_id"`
	Task            string   `json:"task"`
	GeneratedAt     string   `json:"generated_at"`
	Artifact        string   `json:"artifact"`
	Labels          []string `json:"labels"`
	FeatureMode     string   `json:"feature_mode"`
	FeatureTokenIDs []int    `json:"feature_token_ids"`
}

// TextClassificationBundle loads a BIOnet-native text classification artifact
// plus the feature metadata needed to project input token ids into model
// features.
type TextClassificationBundle struct {
	metadata         TextClassificationBundleMetadata
	model            *Model
	featureIndexByID map[int]int
}

// LoadTextClassificationBundle loads a BIOnet-native text classification
// bundle from disk.
func LoadTextClassificationBundle(bundleDir string) (*TextClassificationBundle, error) {
	metadata, err := LoadTextClassificationBundleMetadata(bundleDir)
	if err != nil {
		return nil, err
	}

	model, err := LoadModel(filepath.Join(bundleDir, metadata.Artifact))
	if err != nil {
		return nil, fmt.Errorf("load bionet text classification bundle: %w", err)
	}

	featureIndexByID := make(map[int]int, len(metadata.FeatureTokenIDs))
	for idx, tokenID := range metadata.FeatureTokenIDs {
		featureIndexByID[tokenID] = idx
	}

	return &TextClassificationBundle{
		metadata:         metadata,
		model:            model,
		featureIndexByID: featureIndexByID,
	}, nil
}

// LoadTextClassificationBundleMetadata loads bundle metadata from disk.
func LoadTextClassificationBundleMetadata(bundleDir string) (TextClassificationBundleMetadata, error) {
	metadataPath := filepath.Join(bundleDir, "metadata.json")
	raw, err := os.ReadFile(metadataPath)
	if err != nil {
		return TextClassificationBundleMetadata{}, fmt.Errorf("read bionet text classification metadata: %w", err)
	}

	var metadata TextClassificationBundleMetadata
	if err := json.Unmarshal(raw, &metadata); err != nil {
		return TextClassificationBundleMetadata{}, fmt.Errorf("decode bionet text classification metadata: %w", err)
	}

	if metadata.Name == "" {
		return TextClassificationBundleMetadata{}, fmt.Errorf("decode bionet text classification metadata: missing name")
	}

	if metadata.ModelID == "" {
		return TextClassificationBundleMetadata{}, fmt.Errorf("decode bionet text classification metadata: missing model id")
	}

	if metadata.Artifact == "" {
		return TextClassificationBundleMetadata{}, fmt.Errorf("decode bionet text classification metadata: missing artifact")
	}

	if metadata.Task != "text-classification" {
		return TextClassificationBundleMetadata{}, fmt.Errorf("decode bionet text classification metadata: unsupported task %q", metadata.Task)
	}

	if metadata.FeatureMode != TextClassificationFeatureModeTokenIDBag {
		return TextClassificationBundleMetadata{}, fmt.Errorf("decode bionet text classification metadata: unsupported feature mode %q", metadata.FeatureMode)
	}

	if len(metadata.Labels) == 0 {
		return TextClassificationBundleMetadata{}, fmt.Errorf("decode bionet text classification metadata: missing labels")
	}

	if len(metadata.FeatureTokenIDs) == 0 {
		return TextClassificationBundleMetadata{}, fmt.Errorf("decode bionet text classification metadata: missing feature token ids")
	}

	return metadata, nil
}

// PredictBatch projects token ids into the configured feature vector and runs
// the BIOnet runtime model over each example.
func (b *TextClassificationBundle) PredictBatch(inputIDs, attentionMasks [][]int64) ([][]float64, error) {
	if b == nil || b.model == nil {
		return nil, fmt.Errorf("predict batch: bundle is not initialized")
	}

	if len(inputIDs) != len(attentionMasks) {
		return nil, fmt.Errorf("predict batch: input batch size mismatch (%d != %d)", len(inputIDs), len(attentionMasks))
	}

	logitsBatch := make([][]float64, 0, len(inputIDs))
	for i := range inputIDs {
		features, err := b.featuresFor(inputIDs[i], attentionMasks[i])
		if err != nil {
			return nil, fmt.Errorf("predict batch: case %d: %w", i, err)
		}

		logits, err := b.model.PredictVector(features)
		if err != nil {
			return nil, fmt.Errorf("predict batch: case %d: %w", i, err)
		}

		logitsBatch = append(logitsBatch, logits)
	}

	return logitsBatch, nil
}

// Labels returns the configured class labels.
func (b *TextClassificationBundle) Labels() []string {
	if b == nil {
		return nil
	}

	return append([]string(nil), b.metadata.Labels...)
}

// ModelID returns the source model identifier for this bundle.
func (b *TextClassificationBundle) ModelID() string {
	if b == nil {
		return ""
	}

	return b.metadata.ModelID
}

// Close is a no-op for the pure-Go BIOnet-native bundle path.
func (b *TextClassificationBundle) Close() error {
	return nil
}

func (b *TextClassificationBundle) featuresFor(inputIDs, attentionMask []int64) ([]float64, error) {
	if len(inputIDs) != len(attentionMask) {
		return nil, fmt.Errorf("feature extraction: input and attention mask length mismatch (%d != %d)", len(inputIDs), len(attentionMask))
	}

	features := tensor.Zeros([]int{len(b.metadata.FeatureTokenIDs)})
	for idx, tokenID := range inputIDs {
		if attentionMask[idx] == 0 {
			continue
		}

		featureIdx, ok := b.featureIndexByID[int(tokenID)]
		if !ok {
			continue
		}

		currentValue := features.GetFlatValue(featureIdx)
		features.SetFlatValue(featureIdx, currentValue+1)
	}

	return append([]float64(nil), features.Values()...), nil
}
