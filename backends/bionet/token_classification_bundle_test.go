package bionet

import "testing"

func TestLoadTokenClassificationBundleMetadata(t *testing.T) {
	t.Parallel()

	metadata, err := LoadTokenClassificationBundleMetadata("../../testdata/native/token-classification/distilbert-ner-windowed-embedding-linear")
	if err != nil {
		t.Fatalf("LoadTokenClassificationBundleMetadata() error = %v", err)
	}

	if metadata.Task != "token-classification" {
		t.Fatalf("unexpected task %q", metadata.Task)
	}

	if metadata.FeatureMode != TokenClassificationFeatureModeWindowedEmbeddingLinear {
		t.Fatalf("unexpected feature mode %q", metadata.FeatureMode)
	}
}

func TestTokenClassificationBundlePredictBatch(t *testing.T) {
	t.Parallel()

	bundle, err := LoadTokenClassificationBundle("../../testdata/native/token-classification/distilbert-ner-windowed-embedding-linear")
	if err != nil {
		t.Fatalf("LoadTokenClassificationBundle() error = %v", err)
	}

	logitsBatch, err := bundle.PredictBatch(
		[][]int64{{101, 1287, 3044, 102}},
		[][]int64{{1, 1, 1, 1}},
	)
	if err != nil {
		t.Fatalf("PredictBatch() error = %v", err)
	}

	if len(logitsBatch) != 1 || len(logitsBatch[0]) != 4 {
		t.Fatalf("unexpected batch shape %#v", logitsBatch)
	}

	if got := len(logitsBatch[0][0]); got != len(bundle.Labels()) {
		t.Fatalf("unexpected label dimension %d", got)
	}
}
