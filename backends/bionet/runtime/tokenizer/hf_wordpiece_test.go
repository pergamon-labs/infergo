package tokenizer

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestLoadHFWordPieceEncoderEncodeSingleAndPair(t *testing.T) {
	t.Parallel()

	path := filepath.Join(t.TempDir(), "tokenizer.json")
	writeTokenizerJSON(t, path, map[string]any{
		"version": "1.0",
		"normalizer": map[string]any{
			"type":                 "BertNormalizer",
			"clean_text":           true,
			"handle_chinese_chars": true,
			"strip_accents":        nil,
			"lowercase":            true,
		},
		"pre_tokenizer": map[string]any{
			"type": "BertPreTokenizer",
		},
		"post_processor": map[string]any{
			"type": "TemplateProcessing",
			"single": []map[string]any{
				{"SpecialToken": map[string]any{"id": "[CLS]", "type_id": 0}},
				{"Sequence": map[string]any{"id": "A", "type_id": 0}},
				{"SpecialToken": map[string]any{"id": "[SEP]", "type_id": 0}},
			},
			"pair": []map[string]any{
				{"SpecialToken": map[string]any{"id": "[CLS]", "type_id": 0}},
				{"Sequence": map[string]any{"id": "A", "type_id": 0}},
				{"SpecialToken": map[string]any{"id": "[SEP]", "type_id": 0}},
				{"Sequence": map[string]any{"id": "B", "type_id": 1}},
				{"SpecialToken": map[string]any{"id": "[SEP]", "type_id": 1}},
			},
			"special_tokens": map[string]any{
				"[CLS]": map[string]any{"id": "[CLS]", "ids": []int{101}, "tokens": []string{"[CLS]"}},
				"[SEP]": map[string]any{"id": "[SEP]", "ids": []int{102}, "tokens": []string{"[SEP]"}},
			},
		},
		"model": map[string]any{
			"type":                      "WordPiece",
			"unk_token":                 "[UNK]",
			"continuing_subword_prefix": "##",
			"max_input_chars_per_word":  100,
			"vocab": map[string]int{
				"[PAD]":       0,
				"[UNK]":       100,
				"[CLS]":       101,
				"[SEP]":       102,
				"this":        200,
				"product":     201,
				"is":          202,
				"excellent":   203,
				"and":         204,
				"reliable":    205,
				".":           206,
				"the":         207,
				"company":     208,
				"said":        209,
				"deal":        210,
				"closed":      211,
				"acquisition": 212,
				"has":         213,
				"been":        214,
				"completed":   215,
				",":           216,
			},
		},
	})

	encoder, err := LoadHFWordPieceEncoder(path)
	if err != nil {
		t.Fatalf("LoadHFWordPieceEncoder() error = %v", err)
	}

	single, err := encoder.Encode("This product is excellent and reliable.", "", 16)
	if err != nil {
		t.Fatalf("Encode(single) error = %v", err)
	}
	if got, want := single.InputIDs, []int64{101, 200, 201, 202, 203, 204, 205, 206, 102}; !equalInt64s(got, want) {
		t.Fatalf("single input ids = %v, want %v", got, want)
	}

	pair, err := encoder.Encode("The company said the deal closed.", "The acquisition has been completed, the company said.", 18)
	if err != nil {
		t.Fatalf("Encode(pair) error = %v", err)
	}
	if got, want := pair.InputIDs, []int64{101, 207, 208, 209, 207, 210, 211, 206, 102, 207, 212, 213, 214, 215, 216, 207, 208, 102}; !equalInt64s(got, want) {
		t.Fatalf("pair input ids = %v, want %v", got, want)
	}
	if len(pair.AttentionMask) != len(pair.InputIDs) {
		t.Fatalf("pair attention mask length = %d, want %d", len(pair.AttentionMask), len(pair.InputIDs))
	}
}

func TestLoadHFWordPieceEncoderRejectsUnsupportedSpec(t *testing.T) {
	t.Parallel()

	path := filepath.Join(t.TempDir(), "tokenizer.json")
	writeTokenizerJSON(t, path, map[string]any{
		"normalizer":     map[string]any{"type": "Lowercase"},
		"pre_tokenizer":  map[string]any{"type": "BertPreTokenizer"},
		"post_processor": map[string]any{"type": "TemplateProcessing"},
		"model": map[string]any{
			"type":      "WordPiece",
			"unk_token": "[UNK]",
			"vocab":     map[string]int{"[UNK]": 0},
		},
	})

	if _, err := LoadHFWordPieceEncoder(path); err == nil {
		t.Fatalf("LoadHFWordPieceEncoder() error = nil, want unsupported normalizer failure")
	}
}

func writeTokenizerJSON(t *testing.T, path string, payload any) {
	t.Helper()

	raw, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		t.Fatalf("json.MarshalIndent() error = %v", err)
	}
	raw = append(raw, '\n')
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatalf("WriteFile(%s) error = %v", path, err)
	}
}

func equalInt64s(left, right []int64) bool {
	if len(left) != len(right) {
		return false
	}
	for i := range left {
		if left[i] != right[i] {
			return false
		}
	}
	return true
}
