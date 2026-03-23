package embeddings

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/minervaai/infergo/backends/bionet/runtime/tensor"
)

// ImportGloVe loads a text GloVe embedding file into a tensor and vocabulary map.
func ImportGloVe(path string) (tensor.Tensor, map[string]int, error) {
	file, err := os.Open(path)
	if err != nil {
		return tensor.Tensor{}, nil, fmt.Errorf("open GloVe file: %w", err)
	}
	defer file.Close()

	var (
		embeddings    []float64
		embeddingDim  int
		vocabularyMap = make(map[string]int)
	)

	scanner := bufio.NewScanner(file)
	index := 0
	for scanner.Scan() {
		line := scanner.Text()
		fields := strings.Fields(line)
		if len(fields) < 2 {
			return tensor.Tensor{}, nil, fmt.Errorf("invalid GloVe line %d", index+1)
		}

		word := fields[0]
		if embeddingDim == 0 {
			embeddingDim = len(fields) - 1
		} else if len(fields)-1 != embeddingDim {
			return tensor.Tensor{}, nil, fmt.Errorf("inconsistent embedding dimension on line %d", index+1)
		}

		for _, raw := range fields[1:] {
			value, err := strconv.ParseFloat(raw, 64)
			if err != nil {
				return tensor.Tensor{}, nil, fmt.Errorf("parse float on line %d: %w", index+1, err)
			}
			embeddings = append(embeddings, value)
		}

		vocabularyMap[word] = index
		index++
	}

	if err := scanner.Err(); err != nil {
		return tensor.Tensor{}, nil, fmt.Errorf("scan GloVe file: %w", err)
	}

	if len(embeddings) == 0 {
		return tensor.Tensor{}, nil, fmt.Errorf("no embeddings found in file")
	}

	return tensor.New(embeddings, []int{len(vocabularyMap), embeddingDim}), vocabularyMap, nil
}
