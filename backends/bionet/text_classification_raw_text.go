package bionet

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	runtimeTokenizer "github.com/pergamon-labs/infergo/backends/bionet/runtime/tokenizer"
)

// TextClassificationRawTextEncoder encodes alpha family-1 raw text into the
// token ids expected by the native BIOnet text bundle.
type TextClassificationRawTextEncoder struct {
	maxSequenceLength int
	supportsPairText  bool
	encoder           runtimeTokenizer.TextEncoder
}

// SupportsPairText reports whether the encoder accepts paired text input.
func (e *TextClassificationRawTextEncoder) SupportsPairText() bool {
	return e != nil && e.supportsPairText
}

// Encode turns one or two raw strings into model-ready input ids.
func (e *TextClassificationRawTextEncoder) Encode(text, textPair string) ([]int64, []int64, error) {
	if e == nil || e.encoder == nil {
		return nil, nil, fmt.Errorf("encode raw text: encoder is not initialized")
	}
	if textPair != "" && !e.supportsPairText {
		return nil, nil, fmt.Errorf("encode raw text: this bundle does not support paired text input")
	}

	encoded, err := e.encoder.Encode(text, textPair, e.maxSequenceLength)
	if err != nil {
		return nil, nil, err
	}
	return encoded.InputIDs, encoded.AttentionMask, nil
}

// LoadTextClassificationRawTextEncoder loads the tokenizer assets for an
// alpha-format bundle when it claims raw-text support.
func LoadTextClassificationRawTextEncoder(bundleDir string) (*TextClassificationRawTextEncoder, error) {
	metadataPath := filepath.Join(bundleDir, "metadata.json")
	raw, err := os.ReadFile(metadataPath)
	if err != nil {
		return nil, fmt.Errorf("read bionet text classification metadata: %w", err)
	}

	var probe textClassificationBundleFormatProbe
	if err := json.Unmarshal(raw, &probe); err != nil {
		return nil, fmt.Errorf("decode bionet text classification metadata probe: %w", err)
	}
	if probe.BundleFormat != alphaNativeBundleFormat {
		return nil, fmt.Errorf("load text raw encoder: legacy bundles do not support tokenizer-backed raw text")
	}

	contract, err := loadAlphaTextClassificationBundleContract(bundleDir, raw)
	if err != nil {
		return nil, err
	}
	if !contract.Metadata.Inputs.RawTextSupported {
		return nil, fmt.Errorf("load text raw encoder: bundle %q does not support raw text", contract.Metadata.ModelID)
	}
	if contract.TokenizerManifest == nil {
		return nil, fmt.Errorf("load text raw encoder: bundle %q is missing tokenizer metadata", contract.Metadata.ModelID)
	}

	manifestDir := filepath.Dir(filepath.Join(bundleDir, contract.Metadata.Tokenizer.Manifest))
	switch contract.TokenizerManifest.Kind {
	case "hf-tokenizer-json":
		tokenizerJSON, ok := contract.TokenizerManifest.Files["tokenizer_json"]
		if !ok || tokenizerJSON == "" {
			return nil, fmt.Errorf("load text raw encoder: tokenizer_json is required for hf-tokenizer-json bundles")
		}
		encoder, err := runtimeTokenizer.LoadHFWordPieceEncoder(filepath.Join(manifestDir, tokenizerJSON))
		if err != nil {
			return nil, err
		}
		return &TextClassificationRawTextEncoder{
			maxSequenceLength: contract.Metadata.Inputs.MaxSequenceLength,
			supportsPairText:  contract.Metadata.Inputs.PairTextSupported,
			encoder:           encoder,
		}, nil
	default:
		return nil, fmt.Errorf("load text raw encoder: unsupported tokenizer kind %q", contract.TokenizerManifest.Kind)
	}
}
