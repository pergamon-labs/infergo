package bionet

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

const (
	alphaNativeBundleFormat                     = "infergo-native"
	alphaTextClassificationBundleFamily         = "encoder-text-classification"
	alphaTextClassificationTask                 = "text-classification"
	alphaTextClassificationOutputKindLabelLogit = "label_logits"
	alphaBackendNameBionet                      = "bionet"
)

// AlphaTextClassificationBundleMetadata is the versioned alpha-era metadata
// shape for supported native text-classification bundles.
type AlphaTextClassificationBundleMetadata struct {
	BundleFormat    string                                   `json:"bundle_format"`
	BundleVersion   string                                   `json:"bundle_version"`
	Family          string                                   `json:"family"`
	Task            string                                   `json:"task"`
	Backend         string                                   `json:"backend"`
	BackendArtifact string                                   `json:"backend_artifact"`
	ModelID         string                                   `json:"model_id"`
	Source          AlphaBundleSourceMetadata                `json:"source"`
	Inputs          AlphaTextClassificationInputContract     `json:"inputs"`
	Tokenizer       AlphaTextClassificationTokenizerContract `json:"tokenizer"`
	Outputs         AlphaTextClassificationOutputContract    `json:"outputs"`
	BackendConfig   AlphaTextClassificationBackendConfig     `json:"backend_config"`
	CreatedAt       string                                   `json:"created_at,omitempty"`
	CreatedBy       AlphaBundleCreatedBy                     `json:"created_by,omitempty"`
}

// AlphaBundleSourceMetadata describes the source model ecosystem.
type AlphaBundleSourceMetadata struct {
	Framework     string `json:"framework"`
	Ecosystem     string `json:"ecosystem,omitempty"`
	WeightsFormat string `json:"weights_format,omitempty"`
	RepoURL       string `json:"repo_url,omitempty"`
	Revision      string `json:"revision,omitempty"`
}

// AlphaTextClassificationInputContract describes supported input modes.
type AlphaTextClassificationInputContract struct {
	RawTextSupported        bool `json:"raw_text_supported"`
	PairTextSupported       bool `json:"pair_text_supported"`
	TokenizedInputSupported bool `json:"tokenized_input_supported"`
	MaxSequenceLength       int  `json:"max_sequence_length"`
}

// AlphaTextClassificationTokenizerContract points to tokenizer assets when the
// bundle supports raw-text input.
type AlphaTextClassificationTokenizerContract struct {
	Manifest             string `json:"manifest,omitempty"`
	Kind                 string `json:"kind,omitempty"`
	RawTextNormalization string `json:"raw_text_normalization,omitempty"`
}

// AlphaTextClassificationOutputContract describes prediction semantics.
type AlphaTextClassificationOutputContract struct {
	Kind           string   `json:"kind"`
	LabelsArtifact string   `json:"labels_artifact"`
	PositiveLabel  string   `json:"positive_label,omitempty"`
	NegativeLabel  string   `json:"negative_label,omitempty"`
	Threshold      *float64 `json:"threshold,omitempty"`
}

// AlphaBundleCreatedBy identifies the exporter that produced the bundle.
type AlphaBundleCreatedBy struct {
	Tool    string `json:"tool,omitempty"`
	Version string `json:"version,omitempty"`
}

// AlphaTextClassificationBackendConfig carries BIOnet-specific projection
// settings needed to run the current native text-classification backend.
type AlphaTextClassificationBackendConfig struct {
	FeatureMode       string `json:"feature_mode"`
	FeatureTokenIDs   []int  `json:"feature_token_ids"`
	EmbeddingArtifact string `json:"embedding_artifact,omitempty"`
}

// AlphaTextClassificationTokenizerManifest is the alpha tokenizer manifest
// format used to validate raw-text-capable bundles.
type AlphaTextClassificationTokenizerManifest struct {
	Kind              string            `json:"kind"`
	RawTextSupported  bool              `json:"raw_text_supported"`
	PairTextSupported bool              `json:"pair_text_supported"`
	SpecialTokens     map[string]string `json:"special_tokens,omitempty"`
	Files             map[string]string `json:"files,omitempty"`
}

type alphaTextClassificationLabelsArtifact struct {
	Labels []string `json:"labels"`
}

type alphaTextClassificationBundleContract struct {
	Metadata          AlphaTextClassificationBundleMetadata
	Labels            []string
	TokenizerManifest *AlphaTextClassificationTokenizerManifest
}

type textClassificationBundleFormatProbe struct {
	BundleFormat string `json:"bundle_format"`
}

func loadAlphaTextClassificationBundleContract(bundleDir string, raw []byte) (alphaTextClassificationBundleContract, error) {
	var metadata AlphaTextClassificationBundleMetadata
	if err := json.Unmarshal(raw, &metadata); err != nil {
		return alphaTextClassificationBundleContract{}, fmt.Errorf("decode alpha text classification metadata: %w", err)
	}

	if metadata.BundleFormat != alphaNativeBundleFormat {
		return alphaTextClassificationBundleContract{}, fmt.Errorf("decode alpha text classification metadata: unsupported bundle format %q", metadata.BundleFormat)
	}

	if _, _, err := parseAlphaBundleVersion(metadata.BundleVersion); err != nil {
		return alphaTextClassificationBundleContract{}, fmt.Errorf("decode alpha text classification metadata: %w", err)
	}

	if metadata.Family != alphaTextClassificationBundleFamily {
		return alphaTextClassificationBundleContract{}, fmt.Errorf("decode alpha text classification metadata: unsupported family %q", metadata.Family)
	}

	if metadata.Task != alphaTextClassificationTask {
		return alphaTextClassificationBundleContract{}, fmt.Errorf("decode alpha text classification metadata: unsupported task %q", metadata.Task)
	}

	if metadata.Backend != alphaBackendNameBionet {
		return alphaTextClassificationBundleContract{}, fmt.Errorf("decode alpha text classification metadata: unsupported backend %q", metadata.Backend)
	}

	if metadata.BackendArtifact == "" {
		return alphaTextClassificationBundleContract{}, fmt.Errorf("decode alpha text classification metadata: missing backend artifact")
	}
	if err := requireBundleFile(bundleDir, metadata.BackendArtifact, "backend artifact"); err != nil {
		return alphaTextClassificationBundleContract{}, err
	}

	if metadata.ModelID == "" {
		return alphaTextClassificationBundleContract{}, fmt.Errorf("decode alpha text classification metadata: missing model id")
	}

	if metadata.Source.Framework != "pytorch" {
		return alphaTextClassificationBundleContract{}, fmt.Errorf("decode alpha text classification metadata: unsupported source framework %q", metadata.Source.Framework)
	}

	if metadata.Inputs.MaxSequenceLength <= 0 {
		return alphaTextClassificationBundleContract{}, fmt.Errorf("decode alpha text classification metadata: invalid max sequence length %d", metadata.Inputs.MaxSequenceLength)
	}

	if !metadata.Inputs.TokenizedInputSupported {
		return alphaTextClassificationBundleContract{}, fmt.Errorf("decode alpha text classification metadata: tokenized input support is required")
	}

	if metadata.Inputs.PairTextSupported && !metadata.Inputs.RawTextSupported {
		return alphaTextClassificationBundleContract{}, fmt.Errorf("decode alpha text classification metadata: pair_text_supported requires raw_text_supported")
	}

	if metadata.Outputs.Kind != alphaTextClassificationOutputKindLabelLogit {
		return alphaTextClassificationBundleContract{}, fmt.Errorf("decode alpha text classification metadata: unsupported output kind %q", metadata.Outputs.Kind)
	}

	if metadata.Outputs.LabelsArtifact == "" {
		return alphaTextClassificationBundleContract{}, fmt.Errorf("decode alpha text classification metadata: missing labels artifact")
	}

	labels, err := loadAlphaTextClassificationLabels(bundleDir, metadata.Outputs.LabelsArtifact)
	if err != nil {
		return alphaTextClassificationBundleContract{}, err
	}

	if metadata.Outputs.PositiveLabel != "" && !containsString(labels, metadata.Outputs.PositiveLabel) {
		return alphaTextClassificationBundleContract{}, fmt.Errorf("decode alpha text classification metadata: positive label %q not present in labels artifact", metadata.Outputs.PositiveLabel)
	}

	if metadata.Outputs.NegativeLabel != "" && !containsString(labels, metadata.Outputs.NegativeLabel) {
		return alphaTextClassificationBundleContract{}, fmt.Errorf("decode alpha text classification metadata: negative label %q not present in labels artifact", metadata.Outputs.NegativeLabel)
	}

	tokenizerManifest, err := loadAlphaTextClassificationTokenizerManifest(bundleDir, metadata)
	if err != nil {
		return alphaTextClassificationBundleContract{}, err
	}

	if err := validateAlphaTextClassificationBackendConfig(bundleDir, metadata.BackendConfig); err != nil {
		return alphaTextClassificationBundleContract{}, err
	}

	return alphaTextClassificationBundleContract{
		Metadata:          metadata,
		Labels:            labels,
		TokenizerManifest: tokenizerManifest,
	}, nil
}

func parseAlphaBundleVersion(value string) (int, int, error) {
	if value == "" {
		return 0, 0, fmt.Errorf("missing bundle version")
	}

	parts := strings.Split(value, ".")
	if len(parts) != 2 {
		return 0, 0, fmt.Errorf("invalid bundle version %q", value)
	}

	major, err := strconv.Atoi(parts[0])
	if err != nil {
		return 0, 0, fmt.Errorf("invalid bundle version %q", value)
	}
	minor, err := strconv.Atoi(parts[1])
	if err != nil {
		return 0, 0, fmt.Errorf("invalid bundle version %q", value)
	}

	if major != 1 {
		return 0, 0, fmt.Errorf("unsupported bundle version major %d", major)
	}
	if minor < 0 {
		return 0, 0, fmt.Errorf("invalid bundle version %q", value)
	}

	return major, minor, nil
}

func loadAlphaTextClassificationLabels(bundleDir, relPath string) ([]string, error) {
	if err := requireBundleFile(bundleDir, relPath, "labels artifact"); err != nil {
		return nil, err
	}

	raw, err := os.ReadFile(filepath.Join(bundleDir, relPath))
	if err != nil {
		return nil, fmt.Errorf("read alpha text classification labels artifact: %w", err)
	}

	var artifact alphaTextClassificationLabelsArtifact
	if err := json.Unmarshal(raw, &artifact); err != nil {
		return nil, fmt.Errorf("decode alpha text classification labels artifact: %w", err)
	}

	if len(artifact.Labels) == 0 {
		return nil, fmt.Errorf("decode alpha text classification labels artifact: missing labels")
	}

	seen := make(map[string]struct{}, len(artifact.Labels))
	for _, label := range artifact.Labels {
		if label == "" {
			return nil, fmt.Errorf("decode alpha text classification labels artifact: labels must not be empty")
		}
		if _, ok := seen[label]; ok {
			return nil, fmt.Errorf("decode alpha text classification labels artifact: duplicate label %q", label)
		}
		seen[label] = struct{}{}
	}

	return artifact.Labels, nil
}

func loadAlphaTextClassificationTokenizerManifest(bundleDir string, metadata AlphaTextClassificationBundleMetadata) (*AlphaTextClassificationTokenizerManifest, error) {
	if !metadata.Inputs.RawTextSupported && metadata.Tokenizer.Manifest == "" {
		return nil, nil
	}

	if metadata.Tokenizer.Manifest == "" {
		return nil, fmt.Errorf("decode alpha text classification metadata: missing tokenizer manifest for raw-text-capable bundle")
	}

	if err := requireBundleFile(bundleDir, metadata.Tokenizer.Manifest, "tokenizer manifest"); err != nil {
		return nil, err
	}

	raw, err := os.ReadFile(filepath.Join(bundleDir, metadata.Tokenizer.Manifest))
	if err != nil {
		return nil, fmt.Errorf("read alpha text classification tokenizer manifest: %w", err)
	}

	var manifest AlphaTextClassificationTokenizerManifest
	if err := json.Unmarshal(raw, &manifest); err != nil {
		return nil, fmt.Errorf("decode alpha text classification tokenizer manifest: %w", err)
	}

	if manifest.Kind != "hf-tokenizer-json" {
		return nil, fmt.Errorf("decode alpha text classification tokenizer manifest: unsupported tokenizer kind %q (alpha supports only %q manifests)", manifest.Kind, "hf-tokenizer-json")
	}

	if metadata.Inputs.RawTextSupported && !manifest.RawTextSupported {
		return nil, fmt.Errorf("decode alpha text classification tokenizer manifest: raw_text_supported must be true")
	}
	if metadata.Inputs.PairTextSupported && !manifest.PairTextSupported {
		return nil, fmt.Errorf("decode alpha text classification tokenizer manifest: pair_text_supported must be true")
	}

	for key, relPath := range manifest.Files {
		if relPath == "" {
			return nil, fmt.Errorf("decode alpha text classification tokenizer manifest: file %q must not be empty", key)
		}
		if err := requireBundleFile(bundleDir, filepath.Join(filepath.Dir(metadata.Tokenizer.Manifest), relPath), fmt.Sprintf("tokenizer file %q", key)); err != nil {
			return nil, err
		}
	}

	return &manifest, nil
}

func validateAlphaTextClassificationBackendConfig(bundleDir string, config AlphaTextClassificationBackendConfig) error {
	switch config.FeatureMode {
	case TextClassificationFeatureModeTokenIDBag,
		TextClassificationFeatureModeEmbeddingAvgPool,
		TextClassificationFeatureModeEmbeddingMaskedAvgPool:
	default:
		return fmt.Errorf("decode alpha text classification metadata: unsupported backend feature mode %q", config.FeatureMode)
	}

	if len(config.FeatureTokenIDs) == 0 {
		return fmt.Errorf("decode alpha text classification metadata: missing backend feature token ids")
	}

	if usesEmbeddingArtifact(config.FeatureMode) {
		if config.EmbeddingArtifact == "" {
			return fmt.Errorf("decode alpha text classification metadata: missing backend embedding artifact")
		}
		if err := requireBundleFile(bundleDir, config.EmbeddingArtifact, "embedding artifact"); err != nil {
			return err
		}
	}

	return nil
}

func requireBundleFile(bundleDir, relPath, label string) error {
	if relPath == "" {
		return fmt.Errorf("load bundle: missing %s", label)
	}

	if _, err := os.Stat(filepath.Join(bundleDir, relPath)); err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf("load bundle: missing %s %q", label, relPath)
		}
		return fmt.Errorf("load bundle: stat %s %q: %w", label, relPath, err)
	}

	return nil
}

func containsString(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}
