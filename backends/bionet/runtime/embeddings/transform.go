package embeddings

import (
	"fmt"

	"github.com/pergamon-labs/infergo/backends/bionet/runtime/tensor"
	"github.com/pergamon-labs/infergo/backends/bionet/runtime/tokenizer"
)

// TextBatchToTokens tokenizes a batch of texts with the provided tokenizer.
func TextBatchToTokens(textBatch []string, tokenizerFn tokenizer.Tokenizer, maxOutputSize int) [][]string {
	output := make([][]string, len(textBatch))
	for i, text := range textBatch {
		output[i] = tokenizerFn(text, maxOutputSize)
	}

	return output
}

// TokensToIndices converts token batches into a [seq, batch, 1] tensor of vocab
// indices. Unknown tokens are encoded as -1.
func TokensToIndices(tokenBatch [][]string, vocab map[string]int) (tensor.Tensor, error) {
	if vocab == nil {
		return tensor.Tensor{}, fmt.Errorf("vocab is nil")
	}

	if len(tokenBatch) == 0 {
		return tensor.Tensor{}, nil
	}

	batchSize := len(tokenBatch)
	maxSeqLen := 0
	for _, tokens := range tokenBatch {
		if len(tokens) > maxSeqLen {
			maxSeqLen = len(tokens)
		}
	}

	output := tensor.Zeros([]int{maxSeqLen, batchSize, 1})
	output.Fill(-1)

	for batchIdx, tokens := range tokenBatch {
		for tokenIdx, token := range tokens {
			if vocabIdx, ok := vocab[token]; ok {
				if err := output.SetValue([]int{tokenIdx, batchIdx, 0}, float64(vocabIdx)); err != nil {
					return tensor.Tensor{}, fmt.Errorf("set token index: %w", err)
				}
			}
		}
	}

	return output, nil
}
