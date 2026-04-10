package tokenizer

import (
	"path/filepath"
	"testing"
)

func TestLoadHFTokenizerJSONEncoderEncodeRobertaBPE(t *testing.T) {
	t.Parallel()

	path := filepath.Join(t.TempDir(), "tokenizer.json")
	writeRobertaTokenizerJSON(t, path)

	encoder, err := LoadHFTokenizerJSONEncoder(path)
	if err != nil {
		t.Fatalf("LoadHFTokenizerJSONEncoder() error = %v", err)
	}

	single, err := encoder.Encode("Hello world", "", 8)
	if err != nil {
		t.Fatalf("Encode(single) error = %v", err)
	}
	if got, want := single.InputIDs, []int64{0, 33, 38, 2}; !equalInt64s(got, want) {
		t.Fatalf("single input ids = %v, want %v", got, want)
	}

	pair, err := encoder.Encode("Hello world", "Again", 7)
	if err != nil {
		t.Fatalf("Encode(pair) error = %v", err)
	}
	if got, want := pair.InputIDs, []int64{0, 33, 38, 2, 2, 42, 2}; !equalInt64s(got, want) {
		t.Fatalf("pair input ids = %v, want %v", got, want)
	}
	if got, want := pair.AttentionMask, []int64{1, 1, 1, 1, 1, 1, 1}; !equalInt64s(got, want) {
		t.Fatalf("pair attention mask = %v, want %v", got, want)
	}

	truncated, err := encoder.Encode("Hello world", "Again", 6)
	if err != nil {
		t.Fatalf("Encode(truncated pair) error = %v", err)
	}
	if got, want := truncated.InputIDs, []int64{0, 33, 2, 2, 42, 2}; !equalInt64s(got, want) {
		t.Fatalf("truncated pair input ids = %v, want %v", got, want)
	}
}

func TestLoadHFTokenizerJSONEncoderRejectsUnsupportedSubset(t *testing.T) {
	t.Parallel()

	path := filepath.Join(t.TempDir(), "tokenizer.json")
	writeTokenizerJSON(t, path, map[string]any{
		"version": "1.0",
		"pre_tokenizer": map[string]any{
			"type": "Whitespace",
		},
		"post_processor": map[string]any{
			"type": "TemplateProcessing",
		},
		"model": map[string]any{
			"type":      "BPE",
			"unk_token": "<unk>",
			"vocab":     map[string]int{"<unk>": 0},
			"merges":    []any{},
		},
	})

	if _, err := LoadHFTokenizerJSONEncoder(path); err == nil {
		t.Fatal("LoadHFTokenizerJSONEncoder() error = nil, want unsupported subset failure")
	}
}

func writeRobertaTokenizerJSON(t *testing.T, path string) {
	t.Helper()

	writeTokenizerJSON(t, path, map[string]any{
		"version":    "1.0",
		"truncation": nil,
		"padding":    nil,
		"added_tokens": []map[string]any{
			{"id": 0, "content": "<s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
			{"id": 1, "content": "<pad>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
			{"id": 2, "content": "</s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
			{"id": 3, "content": "<unk>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
		},
		"normalizer": nil,
		"pre_tokenizer": map[string]any{
			"type":             "ByteLevel",
			"add_prefix_space": false,
			"trim_offsets":     true,
			"use_regex":        true,
		},
		"post_processor": map[string]any{
			"type":             "RobertaProcessing",
			"sep":              []any{"</s>", 2},
			"cls":              []any{"<s>", 0},
			"trim_offsets":     true,
			"add_prefix_space": false,
		},
		"decoder": map[string]any{
			"type":             "ByteLevel",
			"add_prefix_space": false,
			"trim_offsets":     true,
			"use_regex":        true,
		},
		"model": map[string]any{
			"type":                      "BPE",
			"dropout":                   nil,
			"unk_token":                 "<unk>",
			"continuing_subword_prefix": "",
			"end_of_word_suffix":        "",
			"fuse_unk":                  false,
			"byte_fallback":             false,
			"ignore_merges":             false,
			"vocab": map[string]int{
				"<s>":    0,
				"<pad>":  1,
				"</s>":   2,
				"<unk>":  3,
				"H":      10,
				"e":      11,
				"l":      12,
				"o":      13,
				"w":      14,
				"r":      15,
				"d":      16,
				"A":      17,
				"g":      18,
				"a":      19,
				"i":      20,
				"n":      21,
				"Ġ":      22,
				"He":     30,
				"Hel":    31,
				"Hell":   32,
				"Hello":  33,
				"Ġw":     34,
				"Ġwo":    35,
				"Ġwor":   36,
				"Ġworl":  37,
				"Ġworld": 38,
				"Ag":     39,
				"Aga":    40,
				"Agai":   41,
				"Again":  42,
			},
			"merges": [][]string{
				{"H", "e"},
				{"He", "l"},
				{"Hel", "l"},
				{"Hell", "o"},
				{"Ġ", "w"},
				{"Ġw", "o"},
				{"Ġwo", "r"},
				{"Ġwor", "l"},
				{"Ġworl", "d"},
				{"A", "g"},
				{"Ag", "a"},
				{"Aga", "i"},
				{"Agai", "n"},
			},
		},
	})
}
