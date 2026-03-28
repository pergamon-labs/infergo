package bionet

import (
	"testing"

	"github.com/pergamon-labs/infergo/internal/modelpacks"
)

func TestLoadTokenClassificationBundleMetadata(t *testing.T) {
	t.Parallel()

	manifest, err := modelpacks.LoadTokenClassificationManifest("../../testdata/reference/token-classification/model-packs.json")
	if err != nil {
		t.Fatalf("LoadTokenClassificationManifest() error = %v", err)
	}

	for _, pack := range manifest.Packs {
		bundleDir := "../../" + pack.NativeBundleDir
		metadata, err := LoadTokenClassificationBundleMetadata(bundleDir)
		if err != nil {
			t.Fatalf("LoadTokenClassificationBundleMetadata(%q) error = %v", bundleDir, err)
		}

		if metadata.Task != "token-classification" {
			t.Fatalf("unexpected task %q", metadata.Task)
		}

		if metadata.FeatureMode != TokenClassificationFeatureModeWindowedEmbeddingLinear {
			t.Fatalf("unexpected feature mode %q", metadata.FeatureMode)
		}
	}
}

func TestTokenClassificationBundlePredictBatch(t *testing.T) {
	t.Parallel()

	manifest, err := modelpacks.LoadTokenClassificationManifest("../../testdata/reference/token-classification/model-packs.json")
	if err != nil {
		t.Fatalf("LoadTokenClassificationManifest() error = %v", err)
	}

	for _, pack := range manifest.Packs {
		bundleDir := "../../" + pack.NativeBundleDir
		bundle, err := LoadTokenClassificationBundle(bundleDir)
		if err != nil {
			t.Fatalf("LoadTokenClassificationBundle(%q) error = %v", bundleDir, err)
		}

		logitsBatch, err := bundle.PredictBatch(
			[][]int64{{101, 1287, 3044, 102}},
			[][]int64{{1, 1, 1, 1}},
		)
		if err != nil {
			t.Fatalf("PredictBatch(%q) error = %v", bundleDir, err)
		}

		if len(logitsBatch) != 1 || len(logitsBatch[0]) != 4 {
			t.Fatalf("unexpected batch shape %#v", logitsBatch)
		}

		if got := len(logitsBatch[0][0]); got != len(bundle.Labels()) {
			t.Fatalf("unexpected label dimension %d", got)
		}
	}
}
