package parity

import (
	"encoding/json"
	"fmt"
	"os"
)

type transformersReferenceHeader struct {
	Task string `json:"task"`
}

// DetectTransformersReferenceTask reads a reference JSON file just far enough
// to determine which parity flow should handle it.
func DetectTransformersReferenceTask(path string) (string, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("read transformers reference header: %w", err)
	}

	var header transformersReferenceHeader
	if err := json.Unmarshal(raw, &header); err != nil {
		return "", fmt.Errorf("decode transformers reference header: %w", err)
	}

	if header.Task == "" {
		return "", fmt.Errorf("decode transformers reference header: missing task")
	}

	return header.Task, nil
}
