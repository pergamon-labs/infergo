package bionet

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/pergamon-labs/infergo/backends/bionet/runtime/tensor"
)

const (
	// TokenClassificationFeatureModeEmbeddingLinear maps active token ids into a
	// compact embedding table and applies a BIOnet linear head per token.
	TokenClassificationFeatureModeEmbeddingLinear = "embedding-linear"
)

// TokenClassificationBundleMetadata describes an InferGo-native token
// classification bundle layout built on top of the BIOnet runtime.
type TokenClassificationBundleMetadata struct {
	Name              string   `json:"name"`
	Source            string   `json:"source"`
	ModelID           string   `json:"model_id"`
	Task              string   `json:"task"`
	GeneratedAt       string   `json:"generated_at"`
	Artifact          string   `json:"artifact"`
	EmbeddingArtifact string   `json:"embedding_artifact"`
	Labels            []string `json:"labels"`
	FeatureMode       string   `json:"feature_mode"`
	FeatureTokenIDs   []int    `json:"feature_token_ids"`
}

// TokenClassificationBundle loads a BIOnet-native token classification
// artifact plus the feature metadata needed to project input token ids into
// per-token model features.
type TokenClassificationBundle struct {
	metadata         TokenClassificationBundleMetadata
	model            *Model
	embeddingMatrix  tensor.Tensor
	featureIndexByID map[int]int
}

// LoadTokenClassificationBundle loads a BIOnet-native token classification
// bundle from disk.
func LoadTokenClassificationBundle(bundleDir string) (*TokenClassificationBundle, error) {
	metadata, err := LoadTokenClassificationBundleMetadata(bundleDir)
	if err != nil {
		return nil, err
	}

	model, err := LoadModel(filepath.Join(bundleDir, metadata.Artifact))
	if err != nil {
		return nil, fmt.Errorf("load bionet token classification bundle: %w", err)
	}

	embeddingMatrix, err := LoadTensorFromFile(filepath.Join(bundleDir, metadata.EmbeddingArtifact))
	if err != nil {
		return nil, fmt.Errorf("load bionet token classification embeddings: %w", err)
	}

	featureIndexByID := make(map[int]int, len(metadata.FeatureTokenIDs))
	for idx, tokenID := range metadata.FeatureTokenIDs {
		featureIndexByID[tokenID] = idx
	}

	return &TokenClassificationBundle{
		metadata:         metadata,
		model:            model,
		embeddingMatrix:  embeddingMatrix,
		featureIndexByID: featureIndexByID,
	}, nil
}

// LoadTokenClassificationBundleMetadata loads bundle metadata from disk.
func LoadTokenClassificationBundleMetadata(bundleDir string) (TokenClassificationBundleMetadata, error) {
	metadataPath := filepath.Join(bundleDir, "metadata.json")
	raw, err := os.ReadFile(metadataPath)
	if err != nil {
		return TokenClassificationBundleMetadata{}, fmt.Errorf("read bionet token classification metadata: %w", err)
	}

	var metadata TokenClassificationBundleMetadata
	if err := json.Unmarshal(raw, &metadata); err != nil {
		return TokenClassificationBundleMetadata{}, fmt.Errorf("decode bionet token classification metadata: %w", err)
	}

	if metadata.Name == "" {
		return TokenClassificationBundleMetadata{}, fmt.Errorf("decode bionet token classification metadata: missing name")
	}
	if metadata.ModelID == "" {
		return TokenClassificationBundleMetadata{}, fmt.Errorf("decode bionet token classification metadata: missing model id")
	}
	if metadata.Artifact == "" {
		return TokenClassificationBundleMetadata{}, fmt.Errorf("decode bionet token classification metadata: missing artifact")
	}
	if metadata.EmbeddingArtifact == "" {
		return TokenClassificationBundleMetadata{}, fmt.Errorf("decode bionet token classification metadata: missing embedding artifact")
	}
	if metadata.Task != "token-classification" {
		return TokenClassificationBundleMetadata{}, fmt.Errorf("decode bionet token classification metadata: unsupported task %q", metadata.Task)
	}
	if metadata.FeatureMode != TokenClassificationFeatureModeEmbeddingLinear {
		return TokenClassificationBundleMetadata{}, fmt.Errorf("decode bionet token classification metadata: unsupported feature mode %q", metadata.FeatureMode)
	}
	if len(metadata.Labels) == 0 {
		return TokenClassificationBundleMetadata{}, fmt.Errorf("decode bionet token classification metadata: missing labels")
	}
	if len(metadata.FeatureTokenIDs) == 0 {
		return TokenClassificationBundleMetadata{}, fmt.Errorf("decode bionet token classification metadata: missing feature token ids")
	}

	return metadata, nil
}

// PredictBatch projects token ids into compact token embeddings and runs the
// BIOnet runtime model over each active token independently.
func (b *TokenClassificationBundle) PredictBatch(inputIDs, attentionMasks [][]int64) ([][][]float64, error) {
	if b == nil || b.model == nil {
		return nil, fmt.Errorf("predict batch: bundle is not initialized")
	}
	if len(inputIDs) != len(attentionMasks) {
		return nil, fmt.Errorf("predict batch: input batch size mismatch (%d != %d)", len(inputIDs), len(attentionMasks))
	}

	batch := make([][][]float64, 0, len(inputIDs))
	for i := range inputIDs {
		if len(inputIDs[i]) != len(attentionMasks[i]) {
			return nil, fmt.Errorf("predict batch: case %d length mismatch (%d != %d)", i, len(inputIDs[i]), len(attentionMasks[i]))
		}

		sequence := make([][]float64, len(inputIDs[i]))
		for pos, tokenID := range inputIDs[i] {
			var embedding []float64
			if attentionMasks[i][pos] != 0 {
				embedding = b.embeddingForTokenID(int(tokenID))
			} else {
				embedding = make([]float64, b.embeddingDim())
			}

			logits, err := b.model.PredictVector(embedding)
			if err != nil {
				return nil, fmt.Errorf("predict batch: case %d token %d: %w", i, pos, err)
			}
			sequence[pos] = logits
		}

		batch = append(batch, sequence)
	}

	return batch, nil
}

// Labels returns the configured class labels.
func (b *TokenClassificationBundle) Labels() []string {
	if b == nil {
		return nil
	}
	return append([]string(nil), b.metadata.Labels...)
}

// ModelID returns the source model identifier for this bundle.
func (b *TokenClassificationBundle) ModelID() string {
	if b == nil {
		return ""
	}
	return b.metadata.ModelID
}

// Close is a no-op for the pure-Go BIOnet-native bundle path.
func (b *TokenClassificationBundle) Close() error {
	return nil
}

func (b *TokenClassificationBundle) embeddingDim() int {
	shape := b.embeddingMatrix.Shape()
	if len(shape) != 2 {
		return 0
	}
	return shape[1]
}

func (b *TokenClassificationBundle) embeddingForTokenID(tokenID int) []float64 {
	dim := b.embeddingDim()
	if dim == 0 {
		return nil
	}

	featureIdx, ok := b.featureIndexByID[tokenID]
	if !ok {
		return make([]float64, dim)
	}

	start := featureIdx * dim
	end := start + dim
	return append([]float64(nil), b.embeddingMatrix.Values()[start:end]...)
}

// PredictTokenEmbeddings is exposed for tests and future feature work so the
// token projection step can be validated independently of the classifier head.
func (b *TokenClassificationBundle) PredictTokenEmbeddings(inputIDs, attentionMasks []int64) (tensor.Tensor, error) {
	if len(inputIDs) != len(attentionMasks) {
		return tensor.Tensor{}, fmt.Errorf("predict token embeddings: input and attention length mismatch (%d != %d)", len(inputIDs), len(attentionMasks))
	}

	dim := b.embeddingDim()
	values := make([]float64, 0, len(inputIDs)*dim)
	for pos, tokenID := range inputIDs {
		if attentionMasks[pos] == 0 {
			values = append(values, make([]float64, dim)...)
			continue
		}
		values = append(values, b.embeddingForTokenID(int(tokenID))...)
	}

	return tensor.New(values, []int{len(inputIDs), dim}), nil
}

func tokenProbabilities(logits []float64) []float64 {
	maxValue := logits[0]
	for _, value := range logits[1:] {
		if value > maxValue {
			maxValue = value
		}
	}

	expValues := make([]float64, len(logits))
	sum := 0.0
	for i, value := range logits {
		expValues[i] = math.Exp(value - maxValue)
		sum += expValues[i]
	}

	for i := range expValues {
		expValues[i] /= sum
	}

	return expValues
}

// PredictTokenLabels is a small helper that maps the per-token logits into
// the configured label set.
func (b *TokenClassificationBundle) PredictTokenLabels(inputIDs, attentionMasks [][]int64) ([][]string, error) {
	logitsBatch, err := b.PredictBatch(inputIDs, attentionMasks)
	if err != nil {
		return nil, err
	}

	labels := b.Labels()
	output := make([][]string, len(logitsBatch))
	for i, sequence := range logitsBatch {
		output[i] = make([]string, len(sequence))
		for pos, logits := range sequence {
			output[i][pos] = tokenLabelAt(labels, tokenArgMax(tokenProbabilities(logits)))
		}
	}

	return output, nil
}

func tokenArgMax(values []float64) int {
	bestIndex := 0
	bestValue := values[0]
	for i := 1; i < len(values); i++ {
		if values[i] > bestValue {
			bestValue = values[i]
			bestIndex = i
		}
	}
	return bestIndex
}

func tokenLabelAt(labels []string, index int) string {
	if index < 0 || index >= len(labels) {
		return ""
	}
	return labels[index]
}
