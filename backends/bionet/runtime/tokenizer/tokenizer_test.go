package tokenizer

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestBasicTokenizer(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name          string
		text          string
		maxOutputSize int
		expected      []string
	}{
		{
			name:          "basic sentence",
			text:          "Hello, world!",
			maxOutputSize: 10,
			expected:      []string{"hello", ",", "world", "!"},
		},
		{
			name:          "sentence with extra whitespace",
			text:          "Hello,  \tworld! ",
			maxOutputSize: 10,
			expected:      []string{"hello", ",", "world", "!"},
		},
		{
			name:          "multiple words with punctuation",
			text:          "This is a test. It has multiple sentences!",
			maxOutputSize: 20,
			expected:      []string{"this", "is", "a", "test", ".", "it", "has", "multiple", "sentences", "!"},
		},
		{
			name:          "sentence with accents",
			text:          "ben est un développeur C++",
			maxOutputSize: 20,
			expected:      []string{"ben", "est", "un", "développeur", "c++"},
		},
		{
			name:          "respects max output size",
			text:          "This sentence has more words than the maximum size allows.",
			maxOutputSize: 5,
			expected:      []string{"this", "sentence", "has", "more", "words"},
		},
		{
			name:          "empty string",
			text:          "",
			maxOutputSize: 10,
			expected:      []string{},
		},
		{
			name:          "only punctuation",
			text:          "!@#$%^&*().",
			maxOutputSize: 10,
			expected:      []string{"!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "."},
		},
		{
			name:          "zero max output size means no limit",
			text:          "This is a test. It has multiple sentences!",
			maxOutputSize: 0,
			expected:      []string{"this", "is", "a", "test", ".", "it", "has", "multiple", "sentences", "!"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, tt.expected, BasicTokenizer(tt.text, tt.maxOutputSize))
			assert.Equal(t, tt.expected, SentencePiece(tt.text, tt.maxOutputSize))
		})
	}
}

func TestBasicTokenizerWithOffsets(t *testing.T) {
	t.Parallel()

	text := "Sophie Tremblay a parlé avec Hydro-Québec à Montréal."
	spans := BasicTokenizerWithOffsets(text, 0)
	assert.Len(t, spans, 11)

	assert.Equal(t, TokenSpan{
		Token:     "sophie",
		StartByte: 0,
		EndByte:   6,
		StartChar: 0,
		EndChar:   6,
	}, spans[0])
	assert.Equal(t, TokenSpan{
		Token:     "hydro",
		StartByte: 30,
		EndByte:   35,
		StartChar: 29,
		EndChar:   34,
	}, spans[5])
	assert.Equal(t, TokenSpan{
		Token:     "québec",
		StartByte: 36,
		EndByte:   43,
		StartChar: 35,
		EndChar:   41,
	}, spans[7])
	assert.Equal(t, TokenSpan{
		Token:     "montréal",
		StartByte: 47,
		EndByte:   56,
		StartChar: 44,
		EndChar:   52,
	}, spans[9])
	assert.Equal(t, TokenSpan{
		Token:     ".",
		StartByte: 56,
		EndByte:   57,
		StartChar: 52,
		EndChar:   53,
	}, spans[10])
}
