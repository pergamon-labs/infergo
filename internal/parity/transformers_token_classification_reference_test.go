package parity

import (
	"testing"

	"github.com/pergamon-labs/infergo/internal/modelpacks"
)

const tokenClassificationPackManifestPath = "../../testdata/reference/token-classification/model-packs.json"

func TestLoadTransformersTokenClassificationInputSet(t *testing.T) {
	t.Parallel()

	inputSet, err := LoadTransformersTokenClassificationInputSet("../../testdata/reference/token-classification/ner-inputs.json")
	if err != nil {
		t.Fatalf("LoadTransformersTokenClassificationInputSet() error = %v", err)
	}

	if len(inputSet.Cases) < 6 {
		t.Fatalf("expected public token-classification input set, got %d cases", len(inputSet.Cases))
	}
}

func TestLoadTokenClassificationPackManifest(t *testing.T) {
	t.Parallel()

	manifest, err := modelpacks.LoadTokenClassificationManifest(tokenClassificationPackManifestPath)
	if err != nil {
		t.Fatalf("LoadTokenClassificationManifest() error = %v", err)
	}

	if len(manifest.Packs) < 4 {
		t.Fatalf("expected supported token-classification packs, got %d", len(manifest.Packs))
	}
}

func TestLoadTransformersTokenClassificationReferencesFromManifest(t *testing.T) {
	t.Parallel()

	manifest, err := modelpacks.LoadTokenClassificationManifest(tokenClassificationPackManifestPath)
	if err != nil {
		t.Fatalf("LoadTokenClassificationManifest() error = %v", err)
	}

	inputSet, err := LoadTransformersTokenClassificationInputSet("../../" + manifest.InputSetPath)
	if err != nil {
		t.Fatalf("LoadTransformersTokenClassificationInputSet() error = %v", err)
	}

	if len(inputSet.Cases) < 30 {
		t.Fatalf("expected widened token-classification input set, got %d cases", len(inputSet.Cases))
	}

	for _, pack := range manifest.Packs {
		reference, err := LoadTransformersTokenClassificationReference("../../" + pack.ReferencePath)
		if err != nil {
			t.Fatalf("LoadTransformersTokenClassificationReference(%q) error = %v", pack.ReferencePath, err)
		}

		if reference.ModelID != pack.ModelID {
			t.Fatalf("reference %q reported model id %q, want %q", pack.ReferencePath, reference.ModelID, pack.ModelID)
		}

		if got, want := len(reference.Cases), len(inputSet.Cases); got != want {
			t.Fatalf("reference %q case count = %d, want %d", pack.ReferencePath, got, want)
		}
	}
}
