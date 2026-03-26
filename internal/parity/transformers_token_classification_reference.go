package parity

import (
	"encoding/json"
	"fmt"
	"os"
)

// TransformersTokenClassificationInputSet is the public-safe input set used to
// generate an external token-classification reference file from Transformers.
type TransformersTokenClassificationInputSet struct {
	Name  string                                     `json:"name"`
	Cases []TransformersTokenClassificationInputCase `json:"cases"`
}

// TransformersTokenClassificationInputCase is a single text input in the
// public token-classification input set.
type TransformersTokenClassificationInputCase struct {
	ID   string `json:"id"`
	Text string `json:"text"`
}

// TransformersTokenClassificationReference stores the outputs produced by a
// Transformers token-classification model for a fixed public input set.
type TransformersTokenClassificationReference struct {
	Name                string                                         `json:"name"`
	Source              string                                         `json:"source"`
	ModelID             string                                         `json:"model_id"`
	Task                string                                         `json:"task"`
	GeneratedAt         string                                         `json:"generated_at"`
	TransformersVersion string                                         `json:"transformers_version"`
	TorchVersion        string                                         `json:"torch_version"`
	Labels              []string                                       `json:"labels"`
	Cases               []TransformersTokenClassificationReferenceCase `json:"cases"`
}

// TransformersTokenClassificationReferenceCase stores the encoded inputs and
// expected token-level outputs for one text example.
type TransformersTokenClassificationReferenceCase struct {
	ID                    string      `json:"id"`
	Text                  string      `json:"text"`
	Tokens                []string    `json:"tokens"`
	InputIDs              []int       `json:"input_ids"`
	AttentionMask         []int       `json:"attention_mask"`
	ScoringMask           []int       `json:"scoring_mask"`
	ExpectedLogits        [][]float64 `json:"expected_logits"`
	ExpectedProbabilities [][]float64 `json:"expected_probabilities"`
	ExpectedLabels        []string    `json:"expected_labels"`
}

// LoadTransformersTokenClassificationInputSet loads the JSON input set used by
// the Python token-classification reference runner.
func LoadTransformersTokenClassificationInputSet(path string) (TransformersTokenClassificationInputSet, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return TransformersTokenClassificationInputSet{}, fmt.Errorf("read transformers token-classification input set: %w", err)
	}

	var inputSet TransformersTokenClassificationInputSet
	if err := json.Unmarshal(raw, &inputSet); err != nil {
		return TransformersTokenClassificationInputSet{}, fmt.Errorf("decode transformers token-classification input set: %w", err)
	}

	if inputSet.Name == "" {
		return TransformersTokenClassificationInputSet{}, fmt.Errorf("decode transformers token-classification input set: missing name")
	}

	if len(inputSet.Cases) == 0 {
		return TransformersTokenClassificationInputSet{}, fmt.Errorf("decode transformers token-classification input set: no cases defined")
	}

	return inputSet, nil
}

// LoadTransformersTokenClassificationReference loads a generated token
// classification reference JSON file for parity work.
func LoadTransformersTokenClassificationReference(path string) (TransformersTokenClassificationReference, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return TransformersTokenClassificationReference{}, fmt.Errorf("read transformers token-classification reference: %w", err)
	}

	var reference TransformersTokenClassificationReference
	if err := json.Unmarshal(raw, &reference); err != nil {
		return TransformersTokenClassificationReference{}, fmt.Errorf("decode transformers token-classification reference: %w", err)
	}

	if reference.Name == "" {
		return TransformersTokenClassificationReference{}, fmt.Errorf("decode transformers token-classification reference: missing name")
	}

	if reference.ModelID == "" {
		return TransformersTokenClassificationReference{}, fmt.Errorf("decode transformers token-classification reference: missing model id")
	}

	if reference.Task != "token-classification" {
		return TransformersTokenClassificationReference{}, fmt.Errorf("decode transformers token-classification reference: unsupported task %q", reference.Task)
	}

	if len(reference.Cases) == 0 {
		return TransformersTokenClassificationReference{}, fmt.Errorf("decode transformers token-classification reference: no cases defined")
	}

	return reference, nil
}
