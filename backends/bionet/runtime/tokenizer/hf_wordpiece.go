package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"slices"
	"strings"
	"unicode"

	"golang.org/x/text/unicode/norm"
)

// EncodedInput is a tokenized model-ready input payload.
type EncodedInput struct {
	InputIDs      []int64
	AttentionMask []int64
}

// TextEncoder encodes single or paired raw text into model-ready ids.
type TextEncoder interface {
	Encode(text, textPair string, maxSequenceLength int) (EncodedInput, error)
}

type hfTokenizerJSON struct {
	Normalizer    hfNormalizer    `json:"normalizer"`
	PreTokenizer  hfPreTokenizer  `json:"pre_tokenizer"`
	PostProcessor hfPostProcessor `json:"post_processor"`
	Model         hfModel         `json:"model"`
}

type hfNormalizer struct {
	Type               string `json:"type"`
	CleanText          bool   `json:"clean_text"`
	HandleChineseChars bool   `json:"handle_chinese_chars"`
	StripAccents       *bool  `json:"strip_accents"`
	Lowercase          bool   `json:"lowercase"`
}

type hfPreTokenizer struct {
	Type string `json:"type"`
}

type hfPostProcessor struct {
	Type          string                          `json:"type"`
	Single        []hfPostProcessorStep           `json:"single"`
	Pair          []hfPostProcessorStep           `json:"pair"`
	SpecialTokens map[string]hfPostProcessorToken `json:"special_tokens"`
}

type hfPostProcessorStep struct {
	SpecialToken *hfPostProcessorStepToken `json:"SpecialToken,omitempty"`
	Sequence     *hfPostProcessorSequence  `json:"Sequence,omitempty"`
}

type hfPostProcessorStepToken struct {
	ID string `json:"id"`
}

type hfPostProcessorSequence struct {
	ID string `json:"id"`
}

type hfPostProcessorToken struct {
	ID     string   `json:"id"`
	IDs    []int64  `json:"ids"`
	Tokens []string `json:"tokens"`
}

type hfModel struct {
	Type                    string         `json:"type"`
	UnkToken                string         `json:"unk_token"`
	ContinuingSubwordPrefix string         `json:"continuing_subword_prefix"`
	MaxInputCharsPerWord    int            `json:"max_input_chars_per_word"`
	Vocab                   map[string]int `json:"vocab"`
}

// HFWordPieceEncoder encodes BERT-style tokenizer.json assets emitted by
// Hugging Face tokenizers.
type HFWordPieceEncoder struct {
	lowercase               bool
	cleanText               bool
	handleChineseChars      bool
	stripAccents            *bool
	continuingSubwordPrefix string
	maxInputCharsPerWord    int
	unkToken                string
	vocab                   map[string]int64
	postProcessor           hfPostProcessor
}

// LoadHFWordPieceEncoder loads a narrow supported tokenizer.json subset that
// matches the current family-1 alpha exporter targets.
func LoadHFWordPieceEncoder(path string) (*HFWordPieceEncoder, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("load hf tokenizer json: %w", err)
	}

	var spec hfTokenizerJSON
	if err := json.Unmarshal(raw, &spec); err != nil {
		return nil, fmt.Errorf("decode hf tokenizer json: %w", err)
	}

	if spec.Normalizer.Type != "BertNormalizer" {
		return nil, fmt.Errorf("decode hf tokenizer json: unsupported normalizer %q", spec.Normalizer.Type)
	}
	if spec.PreTokenizer.Type != "BertPreTokenizer" {
		return nil, fmt.Errorf("decode hf tokenizer json: unsupported pre_tokenizer %q", spec.PreTokenizer.Type)
	}
	if spec.PostProcessor.Type != "TemplateProcessing" {
		return nil, fmt.Errorf("decode hf tokenizer json: unsupported post_processor %q", spec.PostProcessor.Type)
	}
	if spec.Model.Type != "WordPiece" {
		return nil, fmt.Errorf("decode hf tokenizer json: unsupported model %q", spec.Model.Type)
	}
	if spec.Model.UnkToken == "" {
		return nil, fmt.Errorf("decode hf tokenizer json: missing unk_token")
	}
	if len(spec.Model.Vocab) == 0 {
		return nil, fmt.Errorf("decode hf tokenizer json: missing vocab")
	}
	if spec.Model.MaxInputCharsPerWord <= 0 {
		spec.Model.MaxInputCharsPerWord = 100
	}
	if spec.Model.ContinuingSubwordPrefix == "" {
		spec.Model.ContinuingSubwordPrefix = "##"
	}

	vocab := make(map[string]int64, len(spec.Model.Vocab))
	for token, id := range spec.Model.Vocab {
		vocab[token] = int64(id)
	}
	if _, ok := vocab[spec.Model.UnkToken]; !ok {
		return nil, fmt.Errorf("decode hf tokenizer json: unk token %q missing from vocab", spec.Model.UnkToken)
	}

	return &HFWordPieceEncoder{
		lowercase:               spec.Normalizer.Lowercase,
		cleanText:               spec.Normalizer.CleanText,
		handleChineseChars:      spec.Normalizer.HandleChineseChars,
		stripAccents:            spec.Normalizer.StripAccents,
		continuingSubwordPrefix: spec.Model.ContinuingSubwordPrefix,
		maxInputCharsPerWord:    spec.Model.MaxInputCharsPerWord,
		unkToken:                spec.Model.UnkToken,
		vocab:                   vocab,
		postProcessor:           spec.PostProcessor,
	}, nil
}

// Encode tokenizes one or two raw strings into a single sequence with special
// tokens and an attention mask.
func (e *HFWordPieceEncoder) Encode(text, textPair string, maxSequenceLength int) (EncodedInput, error) {
	if e == nil {
		return EncodedInput{}, fmt.Errorf("encode text: encoder is not initialized")
	}
	if strings.TrimSpace(text) == "" {
		return EncodedInput{}, fmt.Errorf("encode text: text must not be empty")
	}

	sequenceA, err := e.encodeSequence(text)
	if err != nil {
		return EncodedInput{}, err
	}

	var sequenceB []int64
	if textPair != "" {
		sequenceB, err = e.encodeSequence(textPair)
		if err != nil {
			return EncodedInput{}, err
		}
	}

	available := maxSequenceLength
	if available <= 0 {
		available = int(^uint(0) >> 1)
	}
	specialCount := e.specialTokenCount(textPair != "")
	if available <= specialCount {
		return EncodedInput{}, fmt.Errorf("encode text: max sequence length %d is too small for special tokens", maxSequenceLength)
	}

	sequenceA, sequenceB = truncateLongestFirst(sequenceA, sequenceB, available-specialCount)

	inputIDs, err := e.applyTemplate(sequenceA, sequenceB)
	if err != nil {
		return EncodedInput{}, err
	}

	attentionMask := make([]int64, len(inputIDs))
	for i := range attentionMask {
		attentionMask[i] = 1
	}

	return EncodedInput{
		InputIDs:      inputIDs,
		AttentionMask: attentionMask,
	}, nil
}

func (e *HFWordPieceEncoder) encodeSequence(text string) ([]int64, error) {
	tokens := e.preTokenize(text)
	if len(tokens) == 0 {
		return nil, fmt.Errorf("encode text: tokenizer produced no content tokens")
	}

	output := make([]int64, 0, len(tokens))
	for _, token := range tokens {
		output = append(output, e.wordPiece(token)...)
	}
	return output, nil
}

func (e *HFWordPieceEncoder) preTokenize(text string) []string {
	normalized := e.normalize(text)
	if normalized == "" {
		return nil
	}

	output := make([]string, 0, len(normalized)/8)
	for _, word := range strings.Fields(normalized) {
		runes := []rune(word)
		start := 0
		for i, r := range runes {
			if unicode.IsPunct(r) {
				if start < i {
					output = append(output, string(runes[start:i]))
				}
				output = append(output, string(r))
				start = i + 1
			}
		}
		if start < len(runes) {
			output = append(output, string(runes[start:]))
		}
	}

	return output
}

func (e *HFWordPieceEncoder) normalize(text string) string {
	var builder strings.Builder
	for _, r := range text {
		switch {
		case e.cleanText && isControlRune(r):
			continue
		case unicode.IsSpace(r):
			builder.WriteRune(' ')
		case e.handleChineseChars && isChineseRune(r):
			builder.WriteRune(' ')
			builder.WriteRune(r)
			builder.WriteRune(' ')
		default:
			builder.WriteRune(r)
		}
	}

	normalized := builder.String()
	if e.lowercase {
		normalized = strings.ToLower(normalized)
	}

	stripAccents := e.stripAccents != nil && *e.stripAccents
	if e.stripAccents == nil && e.lowercase {
		stripAccents = true
	}
	if stripAccents {
		normalized = removeAccents(normalized)
	}

	return strings.TrimSpace(normalized)
}

func (e *HFWordPieceEncoder) wordPiece(token string) []int64 {
	if token == "" {
		return nil
	}

	runes := []rune(token)
	if len(runes) > e.maxInputCharsPerWord {
		return []int64{e.vocab[e.unkToken]}
	}

	output := make([]int64, 0, len(runes))
	start := 0
	for start < len(runes) {
		end := len(runes)
		var current string
		for start < end {
			piece := string(runes[start:end])
			if start > 0 {
				piece = e.continuingSubwordPrefix + piece
			}
			if _, ok := e.vocab[piece]; ok {
				current = piece
				break
			}
			end--
		}
		if current == "" {
			return []int64{e.vocab[e.unkToken]}
		}
		output = append(output, e.vocab[current])
		start = end
	}

	return output
}

func (e *HFWordPieceEncoder) applyTemplate(sequenceA, sequenceB []int64) ([]int64, error) {
	steps := e.postProcessor.Single
	if len(sequenceB) > 0 {
		if len(e.postProcessor.Pair) == 0 {
			return nil, fmt.Errorf("encode text: tokenizer does not define pair post-processing")
		}
		steps = e.postProcessor.Pair
	}

	output := make([]int64, 0, len(sequenceA)+len(sequenceB)+e.specialTokenCount(len(sequenceB) > 0))
	for _, step := range steps {
		switch {
		case step.SpecialToken != nil:
			token, ok := e.postProcessor.SpecialTokens[step.SpecialToken.ID]
			if !ok || len(token.IDs) == 0 {
				return nil, fmt.Errorf("encode text: special token %q is not defined in tokenizer json", step.SpecialToken.ID)
			}
			output = append(output, token.IDs...)
		case step.Sequence != nil:
			switch step.Sequence.ID {
			case "A":
				output = append(output, sequenceA...)
			case "B":
				output = append(output, sequenceB...)
			default:
				return nil, fmt.Errorf("encode text: unsupported sequence template id %q", step.Sequence.ID)
			}
		default:
			return nil, fmt.Errorf("encode text: unsupported post-processor step")
		}
	}

	return output, nil
}

func (e *HFWordPieceEncoder) specialTokenCount(isPair bool) int {
	steps := e.postProcessor.Single
	if isPair && len(e.postProcessor.Pair) > 0 {
		steps = e.postProcessor.Pair
	}

	var count int
	for _, step := range steps {
		if step.SpecialToken == nil {
			continue
		}
		token := e.postProcessor.SpecialTokens[step.SpecialToken.ID]
		count += len(token.IDs)
	}
	return count
}

func truncateLongestFirst(sequenceA, sequenceB []int64, limit int) ([]int64, []int64) {
	left := slices.Clone(sequenceA)
	right := slices.Clone(sequenceB)
	for len(left)+len(right) > limit {
		switch {
		case len(right) > len(left):
			right = right[:len(right)-1]
		default:
			left = left[:len(left)-1]
		}
	}
	return left, right
}

func isControlRune(r rune) bool {
	if r == '\t' || r == '\n' || r == '\r' {
		return false
	}
	return unicode.IsControl(r)
}

func isChineseRune(r rune) bool {
	return (r >= 0x4E00 && r <= 0x9FFF) ||
		(r >= 0x3400 && r <= 0x4DBF) ||
		(r >= 0x20000 && r <= 0x2A6DF) ||
		(r >= 0x2A700 && r <= 0x2B73F) ||
		(r >= 0x2B740 && r <= 0x2B81F) ||
		(r >= 0x2B820 && r <= 0x2CEAF) ||
		(r >= 0xF900 && r <= 0xFAFF) ||
		(r >= 0x2F800 && r <= 0x2FA1F)
}

func removeAccents(text string) string {
	var builder strings.Builder
	for _, r := range norm.NFD.String(text) {
		if unicode.Is(unicode.Mn, r) {
			continue
		}
		builder.WriteRune(r)
	}
	return builder.String()
}
