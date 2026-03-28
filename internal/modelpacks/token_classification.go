package modelpacks

import (
	"encoding/json"
	"fmt"
	"os"
)

// TokenClassificationManifest describes the supported public-safe
// token-classification model packs in this repo.
type TokenClassificationManifest struct {
	Name         string                             `json:"name"`
	InputSetPath string                             `json:"input_set_path"`
	Packs        []TokenClassificationManifestEntry `json:"packs"`
}

// TokenClassificationManifestEntry describes one supported token-classification
// reference pack.
type TokenClassificationManifestEntry struct {
	InputSetPath    string `json:"input_set_path,omitempty"`
	Key             string `json:"key"`
	ModelID         string `json:"model_id"`
	ReferencePath   string `json:"reference_path"`
	NativeBundleDir string `json:"native_bundle_dir"`
	NativeMode      string `json:"native_mode"`
}

// LoadTokenClassificationManifest loads the supported token-classification
// manifest from disk.
func LoadTokenClassificationManifest(path string) (TokenClassificationManifest, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return TokenClassificationManifest{}, fmt.Errorf("read token-classification pack manifest: %w", err)
	}

	var manifest TokenClassificationManifest
	if err := json.Unmarshal(raw, &manifest); err != nil {
		return TokenClassificationManifest{}, fmt.Errorf("decode token-classification pack manifest: %w", err)
	}

	if manifest.Name == "" {
		return TokenClassificationManifest{}, fmt.Errorf("decode token-classification pack manifest: missing name")
	}
	if manifest.InputSetPath == "" {
		return TokenClassificationManifest{}, fmt.Errorf("decode token-classification pack manifest: missing input_set_path")
	}
	if len(manifest.Packs) == 0 {
		return TokenClassificationManifest{}, fmt.Errorf("decode token-classification pack manifest: no packs defined")
	}

	seenKeys := make(map[string]struct{}, len(manifest.Packs))
	for idx, entry := range manifest.Packs {
		if entry.Key == "" {
			return TokenClassificationManifest{}, fmt.Errorf("decode token-classification pack manifest: missing pack key")
		}
		if _, exists := seenKeys[entry.Key]; exists {
			return TokenClassificationManifest{}, fmt.Errorf("decode token-classification pack manifest: duplicate pack key %q", entry.Key)
		}
		seenKeys[entry.Key] = struct{}{}

		if entry.ModelID == "" {
			return TokenClassificationManifest{}, fmt.Errorf("decode token-classification pack manifest: pack %q missing model_id", entry.Key)
		}
		if entry.ReferencePath == "" {
			return TokenClassificationManifest{}, fmt.Errorf("decode token-classification pack manifest: pack %q missing reference_path", entry.Key)
		}
		if entry.NativeBundleDir == "" {
			return TokenClassificationManifest{}, fmt.Errorf("decode token-classification pack manifest: pack %q missing native_bundle_dir", entry.Key)
		}
		if entry.NativeMode == "" {
			return TokenClassificationManifest{}, fmt.Errorf("decode token-classification pack manifest: pack %q missing native_mode", entry.Key)
		}

		if entry.InputSetPath == "" {
			manifest.Packs[idx].InputSetPath = manifest.InputSetPath
		}
	}

	return manifest, nil
}
