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
