package parity

import "testing"

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

func TestLoadTransformersTextClassificationReference(t *testing.T) {
	t.Parallel()

	reference, err := LoadTransformersTextClassificationReference("../../testdata/reference/text-classification/distilbert-sst2-reference.json")
	if err != nil {
		t.Fatalf("LoadTransformersTextClassificationReference() error = %v", err)
	}

	if reference.ModelID != "distilbert/distilbert-base-uncased-finetuned-sst-2-english" {
		t.Fatalf("unexpected model id %q", reference.ModelID)
	}

	if len(reference.Cases) < 12 {
		t.Fatalf("expected a widened public reference set, got %d cases", len(reference.Cases))
	}
}
