package parity

import "testing"

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

func TestLoadTransformersTokenClassificationReference(t *testing.T) {
	t.Parallel()

	reference, err := LoadTransformersTokenClassificationReference("../../testdata/reference/token-classification/distilbert-ner-reference.json")
	if err != nil {
		t.Fatalf("LoadTransformersTokenClassificationReference() error = %v", err)
	}

	if reference.ModelID != "dslim/distilbert-NER" {
		t.Fatalf("unexpected model id %q", reference.ModelID)
	}

	if len(reference.Cases) < 6 {
		t.Fatalf("expected token-classification reference cases, got %d", len(reference.Cases))
	}
}

func TestLoadTransformersTokenClassificationReferenceBert(t *testing.T) {
	t.Parallel()

	reference, err := LoadTransformersTokenClassificationReference("../../testdata/reference/token-classification/bert-base-ner-reference.json")
	if err != nil {
		t.Fatalf("LoadTransformersTokenClassificationReference() error = %v", err)
	}

	if reference.ModelID != "dslim/bert-base-NER" {
		t.Fatalf("unexpected model id %q", reference.ModelID)
	}

	if len(reference.Cases) < 12 {
		t.Fatalf("expected widened token-classification reference cases, got %d", len(reference.Cases))
	}
}

func TestLoadTransformersTokenClassificationReferenceElasticDistilBert(t *testing.T) {
	t.Parallel()

	reference, err := LoadTransformersTokenClassificationReference("../../testdata/reference/token-classification/elastic-distilbert-conll03-reference.json")
	if err != nil {
		t.Fatalf("LoadTransformersTokenClassificationReference() error = %v", err)
	}

	if reference.ModelID != "elastic/distilbert-base-cased-finetuned-conll03-english" {
		t.Fatalf("unexpected model id %q", reference.ModelID)
	}

	if len(reference.Cases) < 12 {
		t.Fatalf("expected widened token-classification reference cases, got %d", len(reference.Cases))
	}
}

func TestLoadTransformersTokenClassificationReferenceRobertaLarge(t *testing.T) {
	t.Parallel()

	reference, err := LoadTransformersTokenClassificationReference("../../testdata/reference/token-classification/roberta-large-ner-english-reference.json")
	if err != nil {
		t.Fatalf("LoadTransformersTokenClassificationReference() error = %v", err)
	}

	if reference.ModelID != "Jean-Baptiste/roberta-large-ner-english" {
		t.Fatalf("unexpected model id %q", reference.ModelID)
	}

	if len(reference.Cases) < 12 {
		t.Fatalf("expected widened token-classification reference cases, got %d", len(reference.Cases))
	}
}
