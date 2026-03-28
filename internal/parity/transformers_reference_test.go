package parity

import (
	"testing"

	"github.com/pergamon-labs/infergo/internal/modelpacks"
)

const textClassificationPackManifestPath = "../../testdata/reference/text-classification/model-packs.json"

func TestLoadTransformersTextClassificationInputSet(t *testing.T) {
	t.Parallel()

	inputSet, err := LoadTransformersTextClassificationInputSet("../../testdata/reference/text-classification/sst2-inputs.json")
	if err != nil {
		t.Fatalf("LoadTransformersTextClassificationInputSet() error = %v", err)
	}

	if len(inputSet.Cases) < 12 {
		t.Fatalf("expected a widened public input set, got %d cases", len(inputSet.Cases))
	}
}

func TestLoadTextClassificationManifest(t *testing.T) {
	t.Parallel()

	manifest, err := modelpacks.LoadTextClassificationManifest(textClassificationPackManifestPath)
	if err != nil {
		t.Fatalf("LoadTextClassificationManifest() error = %v", err)
	}

	if len(manifest.Packs) < 2 {
		t.Fatalf("expected supported text-classification packs, got %d", len(manifest.Packs))
	}
}

func TestLoadTransformersTextClassificationReferencesFromManifest(t *testing.T) {
	t.Parallel()

	manifest, err := modelpacks.LoadTextClassificationManifest(textClassificationPackManifestPath)
	if err != nil {
		t.Fatalf("LoadTextClassificationManifest() error = %v", err)
	}

	inputSet, err := LoadTransformersTextClassificationInputSet("../../" + manifest.InputSetPath)
	if err != nil {
		t.Fatalf("LoadTransformersTextClassificationInputSet() error = %v", err)
	}

	if len(inputSet.Cases) < 12 {
		t.Fatalf("expected a widened public input set, got %d cases", len(inputSet.Cases))
	}

	for _, pack := range manifest.Packs {
		reference, err := LoadTransformersTextClassificationReference("../../" + pack.ReferencePath)
		if err != nil {
			t.Fatalf("LoadTransformersTextClassificationReference(%q) error = %v", pack.ReferencePath, err)
		}

		if reference.ModelID != pack.ModelID {
			t.Fatalf("reference %q reported model id %q, want %q", pack.ReferencePath, reference.ModelID, pack.ModelID)
		}

		if got, want := len(reference.Cases), len(inputSet.Cases); got != want {
			t.Fatalf("reference %q case count = %d, want %d", pack.ReferencePath, got, want)
		}
	}
}
