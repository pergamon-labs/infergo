package packs

import (
	"fmt"
	"slices"

	runtimeTokenizer "github.com/pergamon-labs/infergo/backends/bionet/runtime/tokenizer"
	"github.com/pergamon-labs/infergo/infer"
	"github.com/pergamon-labs/infergo/internal/modelpacks"
	"github.com/pergamon-labs/infergo/internal/parity"
)

const preferredTextBundleKey = "embedding-masked-avg-pool"

// TextPackInfo describes one supported checked-in text pack.
type TextPackInfo struct {
	Key              string `json:"key"`
	ModelID          string `json:"model_id"`
	ReferencePath    string `json:"reference_path"`
	DefaultBundleKey string `json:"default_bundle_key"`
	DefaultBundleDir string `json:"default_bundle_dir"`
	SupportsRawText  bool   `json:"supports_raw_text"`
}

// TextPack wraps a supported checked-in text pack with convenience helpers.
type TextPack struct {
	info             TextPackInfo
	classifier       infer.TextClassifier
	caseByID         map[string]parity.TransformersTextClassificationReferenceCase
	contentTokenToID map[string]int64
	prefixIDs        []int64
	suffixIDs        []int64
	rawTokenizer     runtimeTokenizer.Tokenizer
	maxContentTokens int
}

// ListTextPacks returns the supported checked-in text packs.
func ListTextPacks() ([]TextPackInfo, error) {
	manifest, err := modelpacks.LoadTextClassificationManifest(modulePath(textManifestPath))
	if err != nil {
		return nil, err
	}

	infos := make([]TextPackInfo, 0, len(manifest.Packs))
	for _, entry := range manifest.Packs {
		defaultBundle, err := chooseDefaultTextBundle(entry)
		if err != nil {
			return nil, err
		}

		reference, err := parity.LoadTransformersTextClassificationReference(modulePath(entry.ReferencePath))
		if err != nil {
			return nil, fmt.Errorf("list text packs: load reference for %q: %w", entry.Key, err)
		}

		_, _, _, rawTokenizer, _, err := buildTextEncoder(reference)
		if err != nil {
			return nil, fmt.Errorf("list text packs: build encoder for %q: %w", entry.Key, err)
		}

		infos = append(infos, TextPackInfo{
			Key:              entry.Key,
			ModelID:          entry.ModelID,
			ReferencePath:    modulePath(entry.ReferencePath),
			DefaultBundleKey: defaultBundle.Key,
			DefaultBundleDir: modulePath(defaultBundle.OutputDir),
			SupportsRawText:  rawTokenizer != nil,
		})
	}

	return infos, nil
}

// LoadTextPack loads a supported checked-in text pack using its default native
// bundle.
func LoadTextPack(key string) (*TextPack, error) {
	manifest, err := modelpacks.LoadTextClassificationManifest(modulePath(textManifestPath))
	if err != nil {
		return nil, err
	}

	entry, err := findTextManifestEntry(manifest, key)
	if err != nil {
		return nil, err
	}

	defaultBundle, err := chooseDefaultTextBundle(entry)
	if err != nil {
		return nil, err
	}

	return loadTextPackEntry(entry, defaultBundle)
}

// Key returns the manifest key for the pack.
func (p *TextPack) Key() string {
	if p == nil {
		return ""
	}
	return p.info.Key
}

// ModelID returns the source model id for the pack.
func (p *TextPack) ModelID() string {
	if p == nil {
		return ""
	}
	return p.info.ModelID
}

// DefaultBundleKey returns the selected checked-in native bundle key.
func (p *TextPack) DefaultBundleKey() string {
	if p == nil {
		return ""
	}
	return p.info.DefaultBundleKey
}

// SupportsRawText reports whether this pack can honestly tokenize raw text
// with the checked-in native tokenizer helper.
func (p *TextPack) SupportsRawText() bool {
	return p != nil && p.rawTokenizer != nil
}

// PredictReferenceCase runs inference for one checked-in reference case id.
func (p *TextPack) PredictReferenceCase(caseID string) (infer.TextPrediction, error) {
	if p == nil {
		return infer.TextPrediction{}, fmt.Errorf("predict reference case: text pack is not initialized")
	}

	item, ok := p.caseByID[caseID]
	if !ok {
		return infer.TextPrediction{}, fmt.Errorf("predict reference case: unknown case %q", caseID)
	}

	return p.classifier.Predict(infer.TextInput{
		InputIDs:      intsToInt64(item.InputIDs),
		AttentionMask: intsToInt64(item.AttentionMask),
	})
}

// PredictTokens runs inference for a tokenized text payload. Tokens must
// already match the checked-in pack's tokenizer pieces.
func (p *TextPack) PredictTokens(tokens []string) (infer.TextPrediction, error) {
	if p == nil {
		return infer.TextPrediction{}, fmt.Errorf("predict tokens: text pack is not initialized")
	}
	if len(tokens) == 0 {
		return infer.TextPrediction{}, fmt.Errorf("predict tokens: at least one token is required")
	}

	inputIDs := make([]int64, 0, len(p.prefixIDs)+len(tokens)+len(p.suffixIDs))
	inputIDs = append(inputIDs, p.prefixIDs...)
	for _, token := range tokens {
		tokenID, ok := p.contentTokenToID[token]
		if !ok {
			return infer.TextPrediction{}, fmt.Errorf("predict tokens: unsupported token %q for pack %q", token, p.info.Key)
		}
		inputIDs = append(inputIDs, tokenID)
	}
	inputIDs = append(inputIDs, p.suffixIDs...)

	attentionMask := make([]int64, len(inputIDs))
	for i := range attentionMask {
		attentionMask[i] = 1
	}

	return p.classifier.Predict(infer.TextInput{
		InputIDs:      inputIDs,
		AttentionMask: attentionMask,
	})
}

// PredictText tokenizes and predicts one raw text input when the checked-in
// pack supports a native tokenizer helper.
func (p *TextPack) PredictText(text string) (infer.TextPrediction, error) {
	if p == nil {
		return infer.TextPrediction{}, fmt.Errorf("predict text: text pack is not initialized")
	}
	if p.rawTokenizer == nil {
		return infer.TextPrediction{}, fmt.Errorf("predict text: pack %q does not support raw-text tokenization", p.info.Key)
	}

	tokens := p.rawTokenizer(text, p.maxContentTokens)
	if len(tokens) == 0 {
		return infer.TextPrediction{}, fmt.Errorf("predict text: tokenizer produced no tokens")
	}

	return p.PredictTokens(tokens)
}

// Close releases the underlying classifier.
func (p *TextPack) Close() error {
	if p == nil || p.classifier == nil {
		return nil
	}
	return p.classifier.Close()
}

func loadTextPackEntry(entry modelpacks.TextClassificationManifestEntry, bundle modelpacks.TextClassificationNativeBundle) (*TextPack, error) {
	classifier, err := infer.LoadTextClassifier(modulePath(bundle.OutputDir))
	if err != nil {
		return nil, fmt.Errorf("load text pack %q: %w", entry.Key, err)
	}

	reference, err := parity.LoadTransformersTextClassificationReference(modulePath(entry.ReferencePath))
	if err != nil {
		_ = classifier.Close()
		return nil, fmt.Errorf("load text pack %q reference: %w", entry.Key, err)
	}

	contentTokenToID, prefixIDs, suffixIDs, rawTokenizer, maxContentTokens, err := buildTextEncoder(reference)
	if err != nil {
		_ = classifier.Close()
		return nil, fmt.Errorf("load text pack %q encoder: %w", entry.Key, err)
	}

	caseByID := make(map[string]parity.TransformersTextClassificationReferenceCase, len(reference.Cases))
	for _, item := range reference.Cases {
		caseByID[item.ID] = item
	}

	return &TextPack{
		info: TextPackInfo{
			Key:              entry.Key,
			ModelID:          entry.ModelID,
			ReferencePath:    modulePath(entry.ReferencePath),
			DefaultBundleKey: bundle.Key,
			DefaultBundleDir: modulePath(bundle.OutputDir),
			SupportsRawText:  rawTokenizer != nil,
		},
		classifier:       classifier,
		caseByID:         caseByID,
		contentTokenToID: contentTokenToID,
		prefixIDs:        prefixIDs,
		suffixIDs:        suffixIDs,
		rawTokenizer:     rawTokenizer,
		maxContentTokens: maxContentTokens,
	}, nil
}

func findTextManifestEntry(manifest modelpacks.TextClassificationManifest, key string) (modelpacks.TextClassificationManifestEntry, error) {
	for _, entry := range manifest.Packs {
		if entry.Key == key {
			return entry, nil
		}
	}
	return modelpacks.TextClassificationManifestEntry{}, fmt.Errorf("unsupported text pack %q", key)
}

func chooseDefaultTextBundle(entry modelpacks.TextClassificationManifestEntry) (modelpacks.TextClassificationNativeBundle, error) {
	for _, bundle := range entry.NativeBundles {
		if bundle.Key == preferredTextBundleKey {
			return bundle, nil
		}
	}
	if len(entry.NativeBundles) == 0 {
		return modelpacks.TextClassificationNativeBundle{}, fmt.Errorf("text pack %q has no native bundles", entry.Key)
	}
	return entry.NativeBundles[0], nil
}

func buildTextEncoder(reference parity.TransformersTextClassificationReference) (map[string]int64, []int64, []int64, runtimeTokenizer.Tokenizer, int, error) {
	tokenToID := make(map[string]int64)
	var prefixIDs []int64
	var suffixIDs []int64
	rawTextSupported := true
	maxContentTokens := 0

	for caseIdx, item := range reference.Cases {
		if len(item.Tokens) != len(item.InputIDs) || len(item.InputIDs) != len(item.AttentionMask) {
			return nil, nil, nil, nil, 0, fmt.Errorf("reference case %q has inconsistent token/input lengths", item.ID)
		}
		basicTokens := runtimeTokenizer.BasicTokenizer(item.Text, len(item.Tokens)+8)
		contentTokens := item.Tokens
		contentIDs := item.InputIDs
		casePrefixIDs := []int64{}
		caseSuffixIDs := []int64{}

		switch {
		case slices.Equal(basicTokens, item.Tokens):
			// Native raw-text pack with no boundary wrappers.
		case len(item.Tokens) >= 2 && slices.Equal(basicTokens, item.Tokens[1:len(item.Tokens)-1]):
			contentTokens = item.Tokens[1 : len(item.Tokens)-1]
			contentIDs = item.InputIDs[1 : len(item.InputIDs)-1]
			casePrefixIDs = []int64{int64(item.InputIDs[0])}
			caseSuffixIDs = []int64{int64(item.InputIDs[len(item.InputIDs)-1])}
		default:
			if len(item.Tokens) >= 2 {
				contentTokens = item.Tokens[1 : len(item.Tokens)-1]
				contentIDs = item.InputIDs[1 : len(item.InputIDs)-1]
				casePrefixIDs = []int64{int64(item.InputIDs[0])}
				caseSuffixIDs = []int64{int64(item.InputIDs[len(item.InputIDs)-1])}
			}
			rawTextSupported = false
		}

		if caseIdx == 0 {
			prefixIDs = casePrefixIDs
			suffixIDs = caseSuffixIDs
		} else if !slices.Equal(prefixIDs, casePrefixIDs) || !slices.Equal(suffixIDs, caseSuffixIDs) {
			return nil, nil, nil, nil, 0, fmt.Errorf("reference case %q uses inconsistent boundary tokens", item.ID)
		}

		if len(contentTokens) > maxContentTokens {
			maxContentTokens = len(contentTokens)
		}

		for i, token := range contentTokens {
			tokenID := int64(contentIDs[i])
			if existing, ok := tokenToID[token]; ok && existing != tokenID {
				return nil, nil, nil, nil, 0, fmt.Errorf("token %q maps to multiple ids (%d, %d)", token, existing, tokenID)
			}
			tokenToID[token] = tokenID
		}

		if !slices.Equal(runtimeTokenizer.BasicTokenizer(item.Text, maxContentTokens+8), contentTokens) {
			rawTextSupported = false
		}
	}

	var rawTokenizer runtimeTokenizer.Tokenizer
	if rawTextSupported {
		rawTokenizer = runtimeTokenizer.BasicTokenizer
	}

	return tokenToID, prefixIDs, suffixIDs, rawTokenizer, maxContentTokens, nil
}
