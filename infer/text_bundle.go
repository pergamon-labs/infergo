package infer

import (
	"fmt"

	"github.com/pergamon-labs/infergo/backends/bionet"
)

// TextBundle is the higher-level family-1 bundle surface for exported
// text-classification artifacts. It keeps the tokenized classifier path while
// also exposing tokenizer-backed raw-text helpers when the bundle supports
// them.
type TextBundle struct {
	info       bionet.TextClassificationBundleInfo
	classifier TextClassifier
	rawEncoder *bionet.TextClassificationRawTextEncoder
}

// LoadTextBundle loads an InferGo-native text-classification bundle and, when
// available, its tokenizer-backed raw-text encoder.
func LoadTextBundle(bundleDir string) (*TextBundle, error) {
	classifier, err := LoadTextClassifier(bundleDir)
	if err != nil {
		return nil, err
	}

	info, err := bionet.InspectTextClassificationBundle(bundleDir)
	if err != nil {
		_ = classifier.Close()
		return nil, err
	}

	var rawEncoder *bionet.TextClassificationRawTextEncoder
	if info.SupportsRawText {
		rawEncoder, err = bionet.LoadTextClassificationRawTextEncoder(bundleDir)
		if err != nil {
			_ = classifier.Close()
			return nil, err
		}
	}

	return &TextBundle{
		info:       info,
		classifier: classifier,
		rawEncoder: rawEncoder,
	}, nil
}

// BackendName reports the backing runtime name.
func (b *TextBundle) BackendName() string {
	if b == nil || b.classifier == nil {
		return ""
	}
	return b.classifier.BackendName()
}

// ModelID reports the source model id captured in the bundle metadata.
func (b *TextBundle) ModelID() string {
	if b == nil {
		return ""
	}
	return b.info.ModelID
}

// Labels reports the ordered output labels captured in the bundle metadata.
func (b *TextBundle) Labels() []string {
	if b == nil || b.classifier == nil {
		return nil
	}
	return b.classifier.Labels()
}

// SupportsRawText reports whether the bundle can tokenize raw text directly.
func (b *TextBundle) SupportsRawText() bool {
	return b != nil && b.info.SupportsRawText && b.rawEncoder != nil
}

// SupportsPairText reports whether the bundle can tokenize paired text
// directly.
func (b *TextBundle) SupportsPairText() bool {
	return b != nil && b.info.SupportsPairText && b.rawEncoder != nil
}

// SupportsTokenizedInput reports whether the bundle accepts direct tokenized
// input ids.
func (b *TextBundle) SupportsTokenizedInput() bool {
	return b != nil && b.info.SupportsTokenizedInput
}

// Predict runs tokenized inference through the stable text classifier path.
func (b *TextBundle) Predict(input TextInput) (TextPrediction, error) {
	if b == nil || b.classifier == nil {
		return TextPrediction{}, fmt.Errorf("predict: text bundle is not initialized")
	}
	return b.classifier.Predict(input)
}

// PredictText tokenizes one raw input string and runs text classification.
func (b *TextBundle) PredictText(text string) (TextPrediction, error) {
	if !b.SupportsRawText() {
		return TextPrediction{}, fmt.Errorf("predict text: bundle %q does not support raw text", b.ModelID())
	}

	inputIDs, attentionMask, err := b.rawEncoder.Encode(text, "")
	if err != nil {
		return TextPrediction{}, err
	}
	return b.classifier.Predict(TextInput{
		InputIDs:      inputIDs,
		AttentionMask: attentionMask,
	})
}

// PredictTextPair tokenizes a paired-text request and runs text
// classification.
func (b *TextBundle) PredictTextPair(text, textPair string) (TextPrediction, error) {
	if !b.SupportsRawText() {
		return TextPrediction{}, fmt.Errorf("predict text pair: bundle %q does not support raw text", b.ModelID())
	}
	if !b.SupportsPairText() {
		return TextPrediction{}, fmt.Errorf("predict text pair: bundle %q does not support paired text", b.ModelID())
	}

	inputIDs, attentionMask, err := b.rawEncoder.Encode(text, textPair)
	if err != nil {
		return TextPrediction{}, err
	}
	return b.classifier.Predict(TextInput{
		InputIDs:      inputIDs,
		AttentionMask: attentionMask,
	})
}

// Close releases the underlying classifier resources.
func (b *TextBundle) Close() error {
	if b == nil || b.classifier == nil {
		return nil
	}
	return b.classifier.Close()
}
