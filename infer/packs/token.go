package packs

import (
	"fmt"
	"slices"
	"unicode"

	runtimeTokenizer "github.com/pergamon-labs/infergo/backends/bionet/runtime/tokenizer"
	"github.com/pergamon-labs/infergo/infer"
	"github.com/pergamon-labs/infergo/internal/modelpacks"
	"github.com/pergamon-labs/infergo/internal/parity"
)

// TokenPackInfo describes one supported checked-in token-classification pack.
type TokenPackInfo struct {
	Key             string `json:"key"`
	ModelID         string `json:"model_id"`
	ReferencePath   string `json:"reference_path"`
	NativeBundleDir string `json:"native_bundle_dir"`
	SupportsRawText bool   `json:"supports_raw_text"`
}

// TokenPack wraps a supported checked-in token pack with convenience helpers.
type TokenPack struct {
	info             TokenPackInfo
	classifier       infer.TokenClassifier
	caseByID         map[string]parity.TransformersTokenClassificationReferenceCase
	contentTokenToID map[string]int64
	prefixIDs        []int64
	suffixIDs        []int64
	prefixLen        int
	suffixLen        int
	rawTokenizer     runtimeTokenizer.Tokenizer
	maxContentTokens int
}

// ListTokenPacks returns the supported checked-in token packs.
func ListTokenPacks() ([]TokenPackInfo, error) {
	manifest, err := modelpacks.LoadTokenClassificationManifest(modulePath(tokenManifestPath))
	if err != nil {
		return nil, err
	}

	infos := make([]TokenPackInfo, 0, len(manifest.Packs))
	for _, entry := range manifest.Packs {
		reference, err := parity.LoadTransformersTokenClassificationReference(modulePath(entry.ReferencePath))
		if err != nil {
			return nil, fmt.Errorf("list token packs: load reference for %q: %w", entry.Key, err)
		}
		_, _, _, _, _, rawTokenizer, _, err := buildTokenEncoder(reference)
		if err != nil {
			return nil, fmt.Errorf("list token packs: build encoder for %q: %w", entry.Key, err)
		}

		infos = append(infos, TokenPackInfo{
			Key:             entry.Key,
			ModelID:         entry.ModelID,
			ReferencePath:   modulePath(entry.ReferencePath),
			NativeBundleDir: modulePath(entry.NativeBundleDir),
			SupportsRawText: rawTokenizer != nil,
		})
	}

	return infos, nil
}

// LoadTokenPack loads a supported checked-in token-classification pack.
func LoadTokenPack(key string) (*TokenPack, error) {
	manifest, err := modelpacks.LoadTokenClassificationManifest(modulePath(tokenManifestPath))
	if err != nil {
		return nil, err
	}

	var entry modelpacks.TokenClassificationManifestEntry
	found := false
	for _, candidate := range manifest.Packs {
		if candidate.Key == key {
			entry = candidate
			found = true
			break
		}
	}
	if !found {
		return nil, fmt.Errorf("unsupported token pack %q", key)
	}

	classifier, err := infer.LoadTokenClassifier(modulePath(entry.NativeBundleDir))
	if err != nil {
		return nil, fmt.Errorf("load token pack %q: %w", key, err)
	}

	reference, err := parity.LoadTransformersTokenClassificationReference(modulePath(entry.ReferencePath))
	if err != nil {
		_ = classifier.Close()
		return nil, fmt.Errorf("load token pack %q reference: %w", key, err)
	}

	contentTokenToID, prefixIDs, suffixIDs, prefixLen, suffixLen, rawTokenizer, maxContentTokens, err := buildTokenEncoder(reference)
	if err != nil {
		_ = classifier.Close()
		return nil, fmt.Errorf("load token pack %q encoder: %w", key, err)
	}

	caseByID := make(map[string]parity.TransformersTokenClassificationReferenceCase, len(reference.Cases))
	for _, item := range reference.Cases {
		caseByID[item.ID] = item
	}

	return &TokenPack{
		info: TokenPackInfo{
			Key:             entry.Key,
			ModelID:         entry.ModelID,
			ReferencePath:   modulePath(entry.ReferencePath),
			NativeBundleDir: modulePath(entry.NativeBundleDir),
			SupportsRawText: rawTokenizer != nil,
		},
		classifier:       classifier,
		caseByID:         caseByID,
		contentTokenToID: contentTokenToID,
		prefixIDs:        prefixIDs,
		suffixIDs:        suffixIDs,
		prefixLen:        prefixLen,
		suffixLen:        suffixLen,
		rawTokenizer:     rawTokenizer,
		maxContentTokens: maxContentTokens,
	}, nil
}

// Key returns the manifest key for the pack.
func (p *TokenPack) Key() string {
	if p == nil {
		return ""
	}
	return p.info.Key
}

// ModelID returns the source model id for the pack.
func (p *TokenPack) ModelID() string {
	if p == nil {
		return ""
	}
	return p.info.ModelID
}

// SupportsRawText reports whether this pack can honestly tokenize raw text
// with the checked-in native tokenizer helper.
func (p *TokenPack) SupportsRawText() bool {
	return p != nil && p.rawTokenizer != nil
}

// PredictReferenceCase runs inference for one checked-in reference case id and
// trims the result to the scored tokens for that case.
func (p *TokenPack) PredictReferenceCase(caseID string) (infer.TokenPrediction, error) {
	if p == nil {
		return infer.TokenPrediction{}, fmt.Errorf("predict reference case: token pack is not initialized")
	}

	item, ok := p.caseByID[caseID]
	if !ok {
		return infer.TokenPrediction{}, fmt.Errorf("predict reference case: unknown case %q", caseID)
	}

	prediction, err := p.classifier.Predict(infer.TokenInput{
		InputIDs:      intsToInt64(item.InputIDs),
		AttentionMask: intsToInt64(item.AttentionMask),
	})
	if err != nil {
		return infer.TokenPrediction{}, err
	}

	return trimPredictionForScoringMask(prediction, item.ScoringMask), nil
}

// PredictTokens runs inference for already-tokenized content tokens. Tokens
// must already match the checked-in pack's tokenizer pieces.
func (p *TokenPack) PredictTokens(tokens []string) (infer.TokenPrediction, error) {
	if p == nil {
		return infer.TokenPrediction{}, fmt.Errorf("predict tokens: token pack is not initialized")
	}
	if len(tokens) == 0 {
		return infer.TokenPrediction{}, fmt.Errorf("predict tokens: at least one token is required")
	}

	inputIDs := make([]int64, 0, len(p.prefixIDs)+len(tokens)+len(p.suffixIDs))
	inputIDs = append(inputIDs, p.prefixIDs...)
	for _, token := range tokens {
		tokenID, ok := p.contentTokenToID[token]
		if !ok {
			return infer.TokenPrediction{}, fmt.Errorf("predict tokens: unsupported token %q for pack %q", token, p.info.Key)
		}
		inputIDs = append(inputIDs, tokenID)
	}
	inputIDs = append(inputIDs, p.suffixIDs...)

	attentionMask := make([]int64, len(inputIDs))
	for i := range attentionMask {
		attentionMask[i] = 1
	}

	prediction, err := p.classifier.Predict(infer.TokenInput{
		InputIDs:      inputIDs,
		AttentionMask: attentionMask,
	})
	if err != nil {
		return infer.TokenPrediction{}, err
	}

	return trimPredictionWindow(prediction, p.prefixLen, p.suffixLen), nil
}

// PredictText tokenizes and predicts one raw text input when the checked-in
// pack supports a native tokenizer helper.
func (p *TokenPack) PredictText(text string) (infer.TokenPrediction, error) {
	if p == nil {
		return infer.TokenPrediction{}, fmt.Errorf("predict text: token pack is not initialized")
	}
	if p.rawTokenizer == nil {
		return infer.TokenPrediction{}, fmt.Errorf("predict text: pack %q does not support raw-text tokenization", p.info.Key)
	}

	tokens := p.rawTokenizer(text, p.maxContentTokens+8)
	tokens = trimUnscoredEdgeTokens(tokens)
	if len(tokens) == 0 {
		return infer.TokenPrediction{}, fmt.Errorf("predict text: tokenizer produced no scored tokens")
	}

	return p.PredictTokens(tokens)
}

// Close releases the underlying classifier.
func (p *TokenPack) Close() error {
	if p == nil || p.classifier == nil {
		return nil
	}
	return p.classifier.Close()
}

func buildTokenEncoder(reference parity.TransformersTokenClassificationReference) (map[string]int64, []int64, []int64, int, int, runtimeTokenizer.Tokenizer, int, error) {
	tokenToID := make(map[string]int64)
	var prefixIDs []int64
	var suffixIDs []int64
	prefixLen := -1
	suffixLen := -1
	rawTextSupported := true
	maxContentTokens := 0

	for _, item := range reference.Cases {
		if len(item.Tokens) != len(item.InputIDs) || len(item.InputIDs) != len(item.AttentionMask) || len(item.InputIDs) != len(item.ScoringMask) {
			return nil, nil, nil, 0, 0, nil, 0, fmt.Errorf("reference case %q has inconsistent token/input lengths", item.ID)
		}

		firstScored := -1
		lastScored := -1
		for idx, value := range item.ScoringMask {
			if value != 0 {
				firstScored = idx
				break
			}
		}
		for idx := len(item.ScoringMask) - 1; idx >= 0; idx-- {
			if item.ScoringMask[idx] != 0 {
				lastScored = idx
				break
			}
		}
		if firstScored == -1 || lastScored == -1 || lastScored < firstScored {
			return nil, nil, nil, 0, 0, nil, 0, fmt.Errorf("reference case %q has no scored tokens", item.ID)
		}

		casePrefixIDs := intsToInt64(item.InputIDs[:firstScored])
		caseSuffixIDs := intsToInt64(item.InputIDs[lastScored+1:])
		contentTokens := append([]string(nil), item.Tokens[firstScored:lastScored+1]...)
		if prefixLen == -1 {
			prefixLen = len(casePrefixIDs)
			suffixLen = len(caseSuffixIDs)
			prefixIDs = casePrefixIDs
			suffixIDs = caseSuffixIDs
		} else if prefixLen != len(casePrefixIDs) || suffixLen != len(caseSuffixIDs) || !slices.Equal(prefixIDs, casePrefixIDs) || !slices.Equal(suffixIDs, caseSuffixIDs) {
			return nil, nil, nil, 0, 0, nil, 0, fmt.Errorf("reference case %q uses inconsistent boundary tokens", item.ID)
		}

		basicTokens := trimUnscoredEdgeTokens(runtimeTokenizer.BasicTokenizer(item.Text, len(item.Tokens)+8))
		if len(basicTokens) > maxContentTokens {
			maxContentTokens = len(basicTokens)
		}
		if !slices.Equal(basicTokens, contentTokens) {
			rawTextSupported = false
		}

		for idx := firstScored; idx <= lastScored; idx++ {
			token := item.Tokens[idx]
			tokenID := int64(item.InputIDs[idx])
			if existing, ok := tokenToID[token]; ok && existing != tokenID {
				return nil, nil, nil, 0, 0, nil, 0, fmt.Errorf("token %q maps to multiple ids (%d, %d)", token, existing, tokenID)
			}
			tokenToID[token] = tokenID
		}
	}

	var rawTokenizer runtimeTokenizer.Tokenizer
	if rawTextSupported {
		rawTokenizer = runtimeTokenizer.BasicTokenizer
	}

	return tokenToID, prefixIDs, suffixIDs, prefixLen, suffixLen, rawTokenizer, maxContentTokens, nil
}

func trimPredictionForScoringMask(prediction infer.TokenPrediction, scoringMask []int) infer.TokenPrediction {
	trimmedLabels := make([]string, 0, len(prediction.TokenLabels))
	trimmedLogits := make([][]float64, 0, len(prediction.TokenLogits))
	for idx, value := range scoringMask {
		if value == 0 {
			continue
		}
		trimmedLabels = append(trimmedLabels, prediction.TokenLabels[idx])
		trimmedLogits = append(trimmedLogits, append([]float64(nil), prediction.TokenLogits[idx]...))
	}
	prediction.TokenLabels = trimmedLabels
	prediction.TokenLogits = trimmedLogits
	return prediction
}

func trimPredictionWindow(prediction infer.TokenPrediction, trimStart, trimEnd int) infer.TokenPrediction {
	start := trimStart
	end := len(prediction.TokenLabels) - trimEnd
	if start < 0 {
		start = 0
	}
	if end < start {
		end = start
	}

	prediction.TokenLabels = append([]string(nil), prediction.TokenLabels[start:end]...)
	prediction.TokenLogits = slices.Clone(prediction.TokenLogits[start:end])
	for i := range prediction.TokenLogits {
		prediction.TokenLogits[i] = append([]float64(nil), prediction.TokenLogits[i]...)
	}
	return prediction
}

func intsToInt64(values []int) []int64 {
	output := make([]int64, len(values))
	for i, value := range values {
		output[i] = int64(value)
	}
	return output
}

func trimUnscoredEdgeTokens(tokens []string) []string {
	start := 0
	for start < len(tokens) && !shouldScoreToken(tokens[start]) {
		start++
	}

	end := len(tokens)
	for end > start && !shouldScoreToken(tokens[end-1]) {
		end--
	}

	return append([]string(nil), tokens[start:end]...)
}

func shouldScoreToken(token string) bool {
	for _, r := range token {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			return true
		}
	}
	return false
}
