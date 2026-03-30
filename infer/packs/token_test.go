package packs_test

import (
	"path/filepath"
	"runtime"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/pergamon-labs/infergo/infer/packs"
	"github.com/pergamon-labs/infergo/internal/parity"
)

func TestListTokenPacks(t *testing.T) {
	items, err := packs.ListTokenPacks()
	require.NoError(t, err)
	require.NotEmpty(t, items)
}

func TestLoadTokenPackPredictReferenceCase(t *testing.T) {
	pack, err := packs.LoadTokenPack("distilcamembert-french-ner")
	require.NoError(t, err)
	defer pack.Close()

	_, currentFile, _, ok := runtime.Caller(0)
	require.True(t, ok)

	referencePath := filepath.Join(filepath.Dir(currentFile), "..", "..", "testdata", "reference", "token-classification", "distilcamembert-french-ner-reference.json")
	reference, err := parity.LoadTransformersTokenClassificationReference(referencePath)
	require.NoError(t, err)

	prediction, err := pack.PredictReferenceCase("frca-001")
	require.NoError(t, err)

	var expected []string
	for idx, label := range reference.Cases[0].ExpectedLabels {
		if reference.Cases[0].ScoringMask[idx] == 0 {
			continue
		}
		expected = append(expected, label)
	}
	require.Equal(t, expected, prediction.TokenLabels)
}

func TestLoadTokenPackPredictTokens(t *testing.T) {
	pack, err := packs.LoadTokenPack("distilcamembert-french-ner")
	require.NoError(t, err)
	defer pack.Close()

	prediction, err := pack.PredictTokens([]string{"▁Jean", "▁Dupont", "▁a", "▁rencontré", "▁Air", "bus", "▁à", "▁Paris"})
	require.NoError(t, err)
	require.Len(t, prediction.TokenLabels, 8)
}
