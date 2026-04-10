package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"

	sugartokenizer "github.com/sugarme/tokenizer"
	sugarpretrained "github.com/sugarme/tokenizer/pretrained"
)

const (
	hfTokenizerSubsetWordPiece = "BERT-style WordPiece"
	hfTokenizerSubsetRoberta   = "RoBERTa-style ByteLevel BPE"
)

// LoadHFTokenizerJSONEncoder loads one of the currently supported
// `hf-tokenizer-json` runtime subsets for family-1 raw-text serving.
func LoadHFTokenizerJSONEncoder(path string) (TextEncoder, error) {
	spec, err := loadHFTokenizerJSONSpec(path)
	if err != nil {
		return nil, err
	}

	switch {
	case isSupportedHFWordPieceSpec(spec):
		return LoadHFWordPieceEncoder(path)
	case isSupportedHFRobertaBPESpec(spec):
		return loadHFRobertaBPEEncoder(path)
	default:
		return nil, fmt.Errorf(
			"decode hf tokenizer json: unsupported runtime subset (alpha supports only %s and %s tokenizer.json bundles)",
			hfTokenizerSubsetWordPiece,
			hfTokenizerSubsetRoberta,
		)
	}
}

func loadHFTokenizerJSONSpec(path string) (hfTokenizerJSON, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return hfTokenizerJSON{}, fmt.Errorf("load hf tokenizer json: %w", err)
	}

	var spec hfTokenizerJSON
	if err := json.Unmarshal(raw, &spec); err != nil {
		return hfTokenizerJSON{}, fmt.Errorf("decode hf tokenizer json: %w", err)
	}
	return spec, nil
}

func isSupportedHFWordPieceSpec(spec hfTokenizerJSON) bool {
	return spec.Normalizer.Type == "BertNormalizer" &&
		spec.PreTokenizer.Type == "BertPreTokenizer" &&
		spec.PostProcessor.Type == "TemplateProcessing" &&
		spec.Model.Type == "WordPiece"
}

func isSupportedHFRobertaBPESpec(spec hfTokenizerJSON) bool {
	return spec.PreTokenizer.Type == "ByteLevel" &&
		spec.PostProcessor.Type == "RobertaProcessing" &&
		spec.Model.Type == "BPE"
}

type hfTokenizerJSONExternalEncoder struct {
	tokenizer *sugartokenizer.Tokenizer
	mu        sync.Mutex
}

func loadHFRobertaBPEEncoder(path string) (TextEncoder, error) {
	tokenizer, err := sugarpretrained.FromFile(path)
	if err != nil {
		return nil, fmt.Errorf("load hf tokenizer json: %w", err)
	}
	return &hfTokenizerJSONExternalEncoder{tokenizer: tokenizer}, nil
}

func (e *hfTokenizerJSONExternalEncoder) Encode(text, textPair string, maxSequenceLength int) (EncodedInput, error) {
	if e == nil || e.tokenizer == nil {
		return EncodedInput{}, fmt.Errorf("encode text: encoder is not initialized")
	}
	if strings.TrimSpace(text) == "" {
		return EncodedInput{}, fmt.Errorf("encode text: text must not be empty")
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	if maxSequenceLength > 0 {
		e.tokenizer.WithTruncation(&sugartokenizer.TruncationParams{
			MaxLength: maxSequenceLength,
			Strategy:  sugartokenizer.LongestFirst,
		})
	} else {
		e.tokenizer.WithTruncation(nil)
	}

	var (
		encoding *sugartokenizer.Encoding
		err      error
	)
	if textPair == "" {
		encoding, err = e.tokenizer.EncodeSingle(text, true)
	} else {
		encoding, err = e.tokenizer.EncodePair(text, textPair, true)
	}
	if err != nil {
		return EncodedInput{}, fmt.Errorf("encode text: %w", err)
	}

	inputIDs := intsToInt64(encoding.GetIds())
	attentionMask := intsToInt64(encoding.GetAttentionMask())
	if len(attentionMask) == 0 {
		attentionMask = make([]int64, len(inputIDs))
		for i := range attentionMask {
			attentionMask[i] = 1
		}
	}

	return EncodedInput{
		InputIDs:      inputIDs,
		AttentionMask: attentionMask,
	}, nil
}

func intsToInt64(values []int) []int64 {
	out := make([]int64, len(values))
	for i, value := range values {
		out[i] = int64(value)
	}
	return out
}
