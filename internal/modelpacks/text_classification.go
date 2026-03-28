package modelpacks

import (
	"encoding/json"
	"fmt"
	"os"
)

// TextClassificationManifest describes the supported public-safe
// text-classification model packs in this repo.
type TextClassificationManifest struct {
	Name         string                            `json:"name"`
	InputSetPath string                            `json:"input_set_path"`
	Packs        []TextClassificationManifestEntry `json:"packs"`
}

// TextClassificationManifestEntry describes one supported pack plus the native
// bundles we keep checked in for it.
type TextClassificationManifestEntry struct {
	Key           string                           `json:"key"`
	ModelID       string                           `json:"model_id"`
	InputSetPath  string                           `json:"input_set_path,omitempty"`
	ReferencePath string                           `json:"reference_path"`
	NativeBundles []TextClassificationNativeBundle `json:"native_bundles"`
}

// TextClassificationNativeBundle describes one checked-in InferGo-native text
// bundle for a supported pack.
type TextClassificationNativeBundle struct {
	Key          string `json:"key"`
	OutputDir    string `json:"output_dir"`
	Mode         string `json:"mode"`
	UseLayerNorm bool   `json:"use_layernorm,omitempty"`
}

// LoadTextClassificationManifest loads the supported text-classification
// manifest from disk.
func LoadTextClassificationManifest(path string) (TextClassificationManifest, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return TextClassificationManifest{}, fmt.Errorf("read text-classification pack manifest: %w", err)
	}

	var manifest TextClassificationManifest
	if err := json.Unmarshal(raw, &manifest); err != nil {
		return TextClassificationManifest{}, fmt.Errorf("decode text-classification pack manifest: %w", err)
	}

	if manifest.Name == "" {
		return TextClassificationManifest{}, fmt.Errorf("decode text-classification pack manifest: missing name")
	}
	if manifest.InputSetPath == "" {
		return TextClassificationManifest{}, fmt.Errorf("decode text-classification pack manifest: missing input_set_path")
	}
	if len(manifest.Packs) == 0 {
		return TextClassificationManifest{}, fmt.Errorf("decode text-classification pack manifest: no packs defined")
	}

	seenPackKeys := make(map[string]struct{}, len(manifest.Packs))
	for i, pack := range manifest.Packs {
		if pack.Key == "" {
			return TextClassificationManifest{}, fmt.Errorf("decode text-classification pack manifest: missing pack key")
		}
		if _, exists := seenPackKeys[pack.Key]; exists {
			return TextClassificationManifest{}, fmt.Errorf("decode text-classification pack manifest: duplicate pack key %q", pack.Key)
		}
		seenPackKeys[pack.Key] = struct{}{}

		if pack.ModelID == "" {
			return TextClassificationManifest{}, fmt.Errorf("decode text-classification pack manifest: pack %q missing model_id", pack.Key)
		}
		if pack.InputSetPath == "" {
			manifest.Packs[i].InputSetPath = manifest.InputSetPath
			pack.InputSetPath = manifest.InputSetPath
		}
		if pack.ReferencePath == "" {
			return TextClassificationManifest{}, fmt.Errorf("decode text-classification pack manifest: pack %q missing reference_path", pack.Key)
		}
		if len(pack.NativeBundles) == 0 {
			return TextClassificationManifest{}, fmt.Errorf("decode text-classification pack manifest: pack %q missing native_bundles", pack.Key)
		}

		seenBundleKeys := make(map[string]struct{}, len(pack.NativeBundles))
		for _, bundle := range pack.NativeBundles {
			if bundle.Key == "" {
				return TextClassificationManifest{}, fmt.Errorf("decode text-classification pack manifest: pack %q has native bundle without key", pack.Key)
			}
			if _, exists := seenBundleKeys[bundle.Key]; exists {
				return TextClassificationManifest{}, fmt.Errorf("decode text-classification pack manifest: pack %q duplicate native bundle key %q", pack.Key, bundle.Key)
			}
			seenBundleKeys[bundle.Key] = struct{}{}

			if bundle.OutputDir == "" {
				return TextClassificationManifest{}, fmt.Errorf("decode text-classification pack manifest: pack %q bundle %q missing output_dir", pack.Key, bundle.Key)
			}
			if bundle.Mode == "" {
				return TextClassificationManifest{}, fmt.Errorf("decode text-classification pack manifest: pack %q bundle %q missing mode", pack.Key, bundle.Key)
			}
		}
	}

	return manifest, nil
}
