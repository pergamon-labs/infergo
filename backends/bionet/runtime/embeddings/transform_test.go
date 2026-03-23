package embeddings

import (
	"strings"
	"testing"

	"github.com/minervaai/infergo/backends/bionet/runtime/tensor"
	"github.com/minervaai/infergo/backends/bionet/runtime/tokenizer"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestTokensToIndices(t *testing.T) {
	t.Parallel()

	vocab := map[string]int{
		"hello": 0,
		"world": 1,
		"test":  2,
	}

	tests := []struct {
		name       string
		tokenBatch [][]string
		expected   tensor.Tensor
		expectErr  bool
		vocab      map[string]int
	}{
		{
			name: "single sentence",
			tokenBatch: [][]string{
				{"hello", "world"},
			},
			expected: tensor.New([]float64{0, 1}, []int{2, 1, 1}),
			vocab:    vocab,
		},
		{
			name: "multiple sentences",
			tokenBatch: [][]string{
				{"hello", "world"},
				{"test", "hello"},
				{"world", "test", "hello"},
			},
			expected: tensor.New([]float64{
				0, 2, 1,
				1, 0, 2,
				-1, -1, 0,
			}, []int{3, 3, 1}),
			vocab: vocab,
		},
		{
			name:       "empty batch",
			tokenBatch: [][]string{},
			expected:   tensor.Tensor{},
			vocab:      vocab,
		},
		{
			name: "nil vocab",
			tokenBatch: [][]string{
				{"hello", "world"},
			},
			expected:  tensor.Tensor{},
			expectErr: true,
			vocab:     nil,
		},
		{
			name: "unknown token",
			tokenBatch: [][]string{
				{"hello", "unknown"},
			},
			expected: tensor.New([]float64{0, -1}, []int{2, 1, 1}),
			vocab:    vocab,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got, err := TokensToIndices(tt.tokenBatch, tt.vocab)
			if tt.expectErr {
				require.Error(t, err)
				return
			}

			require.NoError(t, err)
			assert.Equal(t, tt.expected.Shape(), got.Shape())
			assert.InDeltaSlice(t, tt.expected.Values(), got.Values(), 1e-6)
		})
	}
}

func TestTextBatchToTokens(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name          string
		textBatch     []string
		tokenizerFn   tokenizer.Tokenizer
		maxOutputSize int
		expected      [][]string
	}{
		{
			name:          "basic tokenization",
			textBatch:     []string{"Hello world!", "How are you?"},
			tokenizerFn:   tokenizer.BasicTokenizer,
			maxOutputSize: 10,
			expected:      [][]string{{"hello", "world", "!"}, {"how", "are", "you", "?"}},
		},
		{
			name:          "empty input",
			textBatch:     []string{},
			tokenizerFn:   tokenizer.BasicTokenizer,
			maxOutputSize: 10,
			expected:      [][]string{},
		},
		{
			name:          "max output size limit",
			textBatch:     []string{"This is a long sentence that exceeds the max output size."},
			tokenizerFn:   tokenizer.BasicTokenizer,
			maxOutputSize: 5,
			expected:      [][]string{{"this", "is", "a", "long", "sentence"}},
		},
		{
			name:      "custom tokenizer",
			textBatch: []string{"custom,tokenizer,test"},
			tokenizerFn: func(text string, maxOutputSize int) []string {
				return strings.Split(text, ",")
			},
			maxOutputSize: 10,
			expected:      [][]string{{"custom", "tokenizer", "test"}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, tt.expected, TextBatchToTokens(tt.textBatch, tt.tokenizerFn, tt.maxOutputSize))
		})
	}
}
