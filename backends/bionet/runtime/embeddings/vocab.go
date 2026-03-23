package embeddings

import (
	"encoding/gob"
	"fmt"
	"os"
)

// LoadVocab loads a gob-encoded vocabulary map from disk.
func LoadVocab(vocabFile string) (map[string]int, error) {
	file, err := os.Open(vocabFile)
	if err != nil {
		return nil, fmt.Errorf("open vocab file: %w", err)
	}
	defer file.Close()

	var vocab map[string]int
	decoder := gob.NewDecoder(file)
	if err := decoder.Decode(&vocab); err != nil {
		return nil, fmt.Errorf("decode vocab file: %w", err)
	}

	return vocab, nil
}

// SaveVocab saves a vocabulary map to a gob file.
func SaveVocab(vocab map[string]int, vocabFile string) error {
	file, err := os.Create(vocabFile)
	if err != nil {
		return fmt.Errorf("create vocab file: %w", err)
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	if err := encoder.Encode(vocab); err != nil {
		return fmt.Errorf("encode vocab file: %w", err)
	}

	return nil
}
