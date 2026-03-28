package infer

import (
	"fmt"
	"slices"

	"github.com/pergamon-labs/infergo/backends/bionet"
)

// LoadTextClassifier loads a checked-in native text-classification bundle using
// InferGo's stable public API.
func LoadTextClassifier(bundleDir string) (TextClassifier, error) {
	bundle, err := bionet.LoadTextClassificationBundle(bundleDir)
	if err != nil {
		return nil, err
	}

	return &nativeTextClassifier{bundle: bundle}, nil
}

// LoadTokenClassifier loads a checked-in native token-classification bundle
// using InferGo's stable public API.
func LoadTokenClassifier(bundleDir string) (TokenClassifier, error) {
	bundle, err := bionet.LoadTokenClassificationBundle(bundleDir)
	if err != nil {
		return nil, err
	}

	return &nativeTokenClassifier{bundle: bundle}, nil
}

type nativeTextClassifier struct {
	bundle *bionet.TextClassificationBundle
}

func (c *nativeTextClassifier) BackendName() string {
	return bionet.Backend{}.Name()
}

func (c *nativeTextClassifier) ModelID() string {
	if c == nil || c.bundle == nil {
		return ""
	}
	return c.bundle.ModelID()
}

func (c *nativeTextClassifier) Labels() []string {
	if c == nil || c.bundle == nil {
		return nil
	}
	return c.bundle.Labels()
}

func (c *nativeTextClassifier) Predict(input TextInput) (TextPrediction, error) {
	results, err := c.PredictBatch([]TextInput{input})
	if err != nil {
		return TextPrediction{}, err
	}
	return results[0], nil
}

func (c *nativeTextClassifier) PredictBatch(inputs []TextInput) ([]TextPrediction, error) {
	if c == nil || c.bundle == nil {
		return nil, fmt.Errorf("predict batch: text classifier is not initialized")
	}
	if len(inputs) == 0 {
		return nil, fmt.Errorf("predict batch: at least one input is required")
	}

	inputIDs := make([][]int64, len(inputs))
	attentionMasks := make([][]int64, len(inputs))
	for i, input := range inputs {
		normalized, err := normalizeInput(input.InputIDs, input.AttentionMask)
		if err != nil {
			return nil, fmt.Errorf("predict batch: text input %d: %w", i, err)
		}
		inputIDs[i] = normalized.inputIDs
		attentionMasks[i] = normalized.attentionMask
	}

	logitsBatch, err := c.bundle.PredictBatch(inputIDs, attentionMasks)
	if err != nil {
		return nil, err
	}

	labels := c.bundle.Labels()
	output := make([]TextPrediction, len(logitsBatch))
	for i, logits := range logitsBatch {
		labelIdx := argMax(logits)
		output[i] = TextPrediction{
			Backend: c.BackendName(),
			ModelID: c.bundle.ModelID(),
			Labels:  labels,
			Logits:  slices.Clone(logits),
			Label:   labels[labelIdx],
		}
	}

	return output, nil
}

func (c *nativeTextClassifier) Close() error {
	if c == nil || c.bundle == nil {
		return nil
	}
	return c.bundle.Close()
}

type nativeTokenClassifier struct {
	bundle *bionet.TokenClassificationBundle
}

func (c *nativeTokenClassifier) BackendName() string {
	return bionet.Backend{}.Name()
}

func (c *nativeTokenClassifier) ModelID() string {
	if c == nil || c.bundle == nil {
		return ""
	}
	return c.bundle.ModelID()
}

func (c *nativeTokenClassifier) Labels() []string {
	if c == nil || c.bundle == nil {
		return nil
	}
	return c.bundle.Labels()
}

func (c *nativeTokenClassifier) Predict(input TokenInput) (TokenPrediction, error) {
	results, err := c.PredictBatch([]TokenInput{input})
	if err != nil {
		return TokenPrediction{}, err
	}
	return results[0], nil
}

func (c *nativeTokenClassifier) PredictBatch(inputs []TokenInput) ([]TokenPrediction, error) {
	if c == nil || c.bundle == nil {
		return nil, fmt.Errorf("predict batch: token classifier is not initialized")
	}
	if len(inputs) == 0 {
		return nil, fmt.Errorf("predict batch: at least one input is required")
	}

	inputIDs := make([][]int64, len(inputs))
	attentionMasks := make([][]int64, len(inputs))
	for i, input := range inputs {
		normalized, err := normalizeInput(input.InputIDs, input.AttentionMask)
		if err != nil {
			return nil, fmt.Errorf("predict batch: token input %d: %w", i, err)
		}
		inputIDs[i] = normalized.inputIDs
		attentionMasks[i] = normalized.attentionMask
	}

	logitsBatch, err := c.bundle.PredictBatch(inputIDs, attentionMasks)
	if err != nil {
		return nil, err
	}

	labels := c.bundle.Labels()
	output := make([]TokenPrediction, len(logitsBatch))
	for i, tokenLogits := range logitsBatch {
		tokenLabels := make([]string, len(tokenLogits))
		clonedLogits := make([][]float64, len(tokenLogits))
		for pos, logits := range tokenLogits {
			labelIdx := argMax(logits)
			tokenLabels[pos] = labels[labelIdx]
			clonedLogits[pos] = slices.Clone(logits)
		}

		output[i] = TokenPrediction{
			Backend:     c.BackendName(),
			ModelID:     c.bundle.ModelID(),
			Labels:      labels,
			TokenLabels: tokenLabels,
			TokenLogits: clonedLogits,
		}
	}

	return output, nil
}

func (c *nativeTokenClassifier) Close() error {
	if c == nil || c.bundle == nil {
		return nil
	}
	return c.bundle.Close()
}

type normalizedInput struct {
	inputIDs      []int64
	attentionMask []int64
}

func normalizeInput(inputIDs, attentionMask []int64) (normalizedInput, error) {
	if len(inputIDs) == 0 {
		return normalizedInput{}, fmt.Errorf("input_ids must not be empty")
	}

	normalizedIDs := slices.Clone(inputIDs)
	switch {
	case len(attentionMask) == 0:
		mask := make([]int64, len(inputIDs))
		for i := range mask {
			mask[i] = 1
		}
		return normalizedInput{inputIDs: normalizedIDs, attentionMask: mask}, nil
	case len(attentionMask) != len(inputIDs):
		return normalizedInput{}, fmt.Errorf("input_ids and attention_mask length mismatch (%d != %d)", len(inputIDs), len(attentionMask))
	default:
		return normalizedInput{
			inputIDs:      normalizedIDs,
			attentionMask: slices.Clone(attentionMask),
		}, nil
	}
}

func argMax(values []float64) int {
	bestIdx := 0
	bestValue := values[0]
	for i := 1; i < len(values); i++ {
		if values[i] > bestValue {
			bestValue = values[i]
			bestIdx = i
		}
	}
	return bestIdx
}
