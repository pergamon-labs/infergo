package tokenizer

import (
	"math"
	"strings"
	"unicode"
)

// Tokenizer tokenizes raw text into a bounded token slice.
type Tokenizer func(text string, maxOutputSize int) []string

// BasicTokenizer lowercases text, splits on whitespace, and breaks out
// punctuation as its own token. This is intentionally simple and is not a
// SentencePiece implementation.
func BasicTokenizer(text string, maxOutputSize int) []string {
	if maxOutputSize <= 0 {
		maxOutputSize = math.MaxInt
	}

	output := make([]string, 0, len(text)/8)
	words := strings.Fields(text)

	for _, word := range words {
		if len(output) >= maxOutputSize {
			break
		}

		runes := []rune(word)
		start := 0
		for i, r := range runes {
			if unicode.IsPunct(r) {
				if start < i {
					output = append(output, strings.ToLower(string(runes[start:i])))
				}
				output = append(output, string(r))
				start = i + 1
			}
		}

		if start < len(runes) {
			output = append(output, strings.ToLower(string(runes[start:])))
		}

		if len(output) >= maxOutputSize {
			break
		}
	}

	return output
}

// SentencePiece is a compatibility alias kept for the original BIOnet API.
// Deprecated: this function is not a real SentencePiece tokenizer. Use
// BasicTokenizer instead.
func SentencePiece(text string, maxOutputSize int) []string {
	return BasicTokenizer(text, maxOutputSize)
}
