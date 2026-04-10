package parity

import (
	"encoding/json"
	"fmt"
	"os"
)

// TransformersTextClassificationInputSet is the public-safe input set used to
// generate an external reference file from Hugging Face Transformers.
type TransformersTextClassificationInputSet struct {
	Name  string                                    `json:"name"`
	Cases []TransformersTextClassificationInputCase `json:"cases"`
}

// TransformersTextClassificationInputCase is a single text input in the public
// reference input set.
type TransformersTextClassificationInputCase struct {
	ID       string `json:"id"`
	Text     string `json:"text"`
	TextPair string `json:"text_pair,omitempty"`
}

// TransformersTextClassificationReference stores the outputs produced by a
// Transformers model for a fixed public input set.
type TransformersTextClassificationReference struct {
	Name                string                                        `json:"name"`
	Source              string                                        `json:"source"`
	ModelID             string                                        `json:"model_id"`
	Task                string                                        `json:"task"`
	GeneratedAt         string                                        `json:"generated_at"`
	TransformersVersion string                                        `json:"transformers_version"`
	TorchVersion        string                                        `json:"torch_version"`
	Labels              []string                                      `json:"labels"`
	Cases               []TransformersTextClassificationReferenceCase `json:"cases"`
}

// TransformersTextClassificationReferenceCase stores the encoded inputs and
// expected outputs for one text example.
type TransformersTextClassificationReferenceCase struct {
	ID                    string    `json:"id"`
	Text                  string    `json:"text"`
	TextPair              string    `json:"text_pair,omitempty"`
	Tokens                []string  `json:"tokens"`
	InputIDs              []int     `json:"input_ids"`
	AttentionMask         []int     `json:"attention_mask"`
	ExpectedLogits        []float64 `json:"expected_logits"`
	ExpectedProbabilities []float64 `json:"expected_probabilities"`
	ExpectedLabel         string    `json:"expected_label"`
}

// LoadTransformersTextClassificationInputSet loads the JSON input set used by
// the Python reference runner.
func LoadTransformersTextClassificationInputSet(path string) (TransformersTextClassificationInputSet, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return TransformersTextClassificationInputSet{}, fmt.Errorf("read transformers input set: %w", err)
	}

	var inputSet TransformersTextClassificationInputSet
	if err := json.Unmarshal(raw, &inputSet); err != nil {
		return TransformersTextClassificationInputSet{}, fmt.Errorf("decode transformers input set: %w", err)
	}

	if inputSet.Name == "" {
		return TransformersTextClassificationInputSet{}, fmt.Errorf("decode transformers input set: missing name")
	}

	if len(inputSet.Cases) == 0 {
		return TransformersTextClassificationInputSet{}, fmt.Errorf("decode transformers input set: no cases defined")
	}

	for _, item := range inputSet.Cases {
		if item.ID == "" {
			return TransformersTextClassificationInputSet{}, fmt.Errorf("decode transformers input set: case missing id")
		}
		if item.Text == "" {
			return TransformersTextClassificationInputSet{}, fmt.Errorf("decode transformers input set: case %q missing text", item.ID)
		}
	}

	return inputSet, nil
}

// LoadTransformersTextClassificationReference loads a generated reference JSON
// file so later parity work can compare InferGo outputs against it.
func LoadTransformersTextClassificationReference(path string) (TransformersTextClassificationReference, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return TransformersTextClassificationReference{}, fmt.Errorf("read transformers reference: %w", err)
	}

	var reference TransformersTextClassificationReference
	if err := json.Unmarshal(raw, &reference); err != nil {
		return TransformersTextClassificationReference{}, fmt.Errorf("decode transformers reference: %w", err)
	}

	if reference.Name == "" {
		return TransformersTextClassificationReference{}, fmt.Errorf("decode transformers reference: missing name")
	}

	if reference.ModelID == "" {
		return TransformersTextClassificationReference{}, fmt.Errorf("decode transformers reference: missing model id")
	}

	if len(reference.Cases) == 0 {
		return TransformersTextClassificationReference{}, fmt.Errorf("decode transformers reference: no cases defined")
	}

	for _, item := range reference.Cases {
		if item.ID == "" {
			return TransformersTextClassificationReference{}, fmt.Errorf("decode transformers reference: case missing id")
		}
		if item.Text == "" {
			return TransformersTextClassificationReference{}, fmt.Errorf("decode transformers reference: case %q missing text", item.ID)
		}
		if len(item.InputIDs) == 0 {
			return TransformersTextClassificationReference{}, fmt.Errorf("decode transformers reference: case %q missing input ids", item.ID)
		}
		if len(item.InputIDs) != len(item.AttentionMask) {
			return TransformersTextClassificationReference{}, fmt.Errorf("decode transformers reference: case %q input_ids and attention_mask length mismatch", item.ID)
		}
	}

	return reference, nil
}
