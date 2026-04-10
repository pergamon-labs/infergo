package tokenizer

import (
	"math"
	"strings"
	"unicode"
	"unicode/utf8"
)

// Tokenizer tokenizes raw text into a bounded token slice.
type Tokenizer func(text string, maxOutputSize int) []string

// TokenSpan describes one tokenized span in the original source text.
// Offsets are start-inclusive and end-exclusive.
type TokenSpan struct {
	Token     string
	StartByte int
	EndByte   int
	StartChar int
	EndChar   int
}

// BasicTokenizer lowercases text, splits on whitespace, and breaks out
// punctuation as its own token. This is intentionally simple and is not a
// SentencePiece implementation.
func BasicTokenizer(text string, maxOutputSize int) []string {
	spans := BasicTokenizerWithOffsets(text, maxOutputSize)
	output := make([]string, len(spans))
	for i, span := range spans {
		output[i] = span.Token
	}
	return output
}

// BasicTokenizerWithOffsets tokenizes raw text using the same rules as
// BasicTokenizer and also reports the source-text span for each token.
func BasicTokenizerWithOffsets(text string, maxOutputSize int) []TokenSpan {
	if maxOutputSize <= 0 {
		maxOutputSize = math.MaxInt
	}

	output := make([]TokenSpan, 0, len(text)/8)
	processField := func(field string, baseByte, baseChar int) {
		var (
			wordRunes      []rune
			wordStartByte  = -1
			wordStartChars = -1
		)
		flushWord := func(endByte, endChar int) {
			if len(wordRunes) == 0 {
				return
			}
			output = append(output, TokenSpan{
				Token:     strings.ToLower(string(wordRunes)),
				StartByte: baseByte + wordStartByte,
				EndByte:   baseByte + endByte,
				StartChar: baseChar + wordStartChars,
				EndChar:   baseChar + endChar,
			})
			wordRunes = wordRunes[:0]
			wordStartByte = -1
			wordStartChars = -1
		}

		charPos := 0
		for bytePos, r := range field {
			if unicode.IsPunct(r) {
				flushWord(bytePos, charPos)
				output = append(output, TokenSpan{
					Token:     string(r),
					StartByte: baseByte + bytePos,
					EndByte:   baseByte + bytePos + utf8.RuneLen(r),
					StartChar: baseChar + charPos,
					EndChar:   baseChar + charPos + 1,
				})
			} else {
				if len(wordRunes) == 0 {
					wordStartByte = bytePos
					wordStartChars = charPos
				}
				wordRunes = append(wordRunes, r)
			}
			charPos++
		}
		flushWord(len(field), charPos)
	}

	inField := false
	fieldStartByte := 0
	fieldStartChar := 0
	charPos := 0
	for bytePos, r := range text {
		if unicode.IsSpace(r) {
			if inField {
				processField(text[fieldStartByte:bytePos], fieldStartByte, fieldStartChar)
				if len(output) >= maxOutputSize {
					return output
				}
				inField = false
			}
		} else if !inField {
			inField = true
			fieldStartByte = bytePos
			fieldStartChar = charPos
		}
		charPos++
	}
	if inField {
		processField(text[fieldStartByte:], fieldStartByte, fieldStartChar)
	}

	return output
}

// SentencePiece is a compatibility alias kept for the original BIOnet API.
// Deprecated: this function is not a real SentencePiece tokenizer. Use
// BasicTokenizer instead.
func SentencePiece(text string, maxOutputSize int) []string {
	return BasicTokenizer(text, maxOutputSize)
}
