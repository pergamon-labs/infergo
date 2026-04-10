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

	var foundRawText bool
	for _, item := range items {
		if item.Key == "infergo-basic-french-ner" {
			require.True(t, item.SupportsRawText)
			foundRawText = true
		}
	}
	require.True(t, foundRawText)
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

func TestLoadTokenPackPredictText(t *testing.T) {
	pack, err := packs.LoadTokenPack("infergo-basic-french-ner")
	require.NoError(t, err)
	defer pack.Close()

	require.True(t, pack.SupportsRawText())

	_, currentFile, _, ok := runtime.Caller(0)
	require.True(t, ok)

	referencePath := filepath.Join(filepath.Dir(currentFile), "..", "..", "testdata", "reference", "token-classification", "infergo-basic-french-ner-reference.json")
	reference, err := parity.LoadTransformersTokenClassificationReference(referencePath)
	require.NoError(t, err)

	prediction, err := pack.PredictText("Sophie Tremblay a parlé avec Hydro-Québec à Montréal.")
	require.NoError(t, err)

	var expected []string
	var actual []string
	for idx, label := range reference.Cases[2].ExpectedLabels[:10] {
		if reference.Cases[2].ScoringMask[idx] == 0 {
			continue
		}
		expected = append(expected, label)
		actual = append(actual, prediction.TokenLabels[idx])
	}
	require.Equal(t, expected, actual)
}

func TestLoadTokenPackTokenizeText(t *testing.T) {
	pack, err := packs.LoadTokenPack("infergo-basic-french-ner")
	require.NoError(t, err)
	defer pack.Close()

	tokens, err := pack.TokenizeText("Sophie Tremblay a parlé avec Hydro-Québec à Montréal.")
	require.NoError(t, err)
	require.Equal(t, []string{"sophie", "tremblay", "a", "parlé", "avec", "hydro", "-", "québec", "à", "montréal"}, tokens)
}

func TestLoadTokenPackTokenizeTextWithOffsets(t *testing.T) {
	pack, err := packs.LoadTokenPack("infergo-basic-french-ner")
	require.NoError(t, err)
	defer pack.Close()

	spans, err := pack.TokenizeTextWithOffsets("Sophie Tremblay a parlé avec Hydro-Québec à Montréal.")
	require.NoError(t, err)
	require.Equal(t, []packs.TokenSpan{
		{Token: "sophie", StartByte: 0, EndByte: 6, StartChar: 0, EndChar: 6},
		{Token: "tremblay", StartByte: 7, EndByte: 15, StartChar: 7, EndChar: 15},
		{Token: "a", StartByte: 16, EndByte: 17, StartChar: 16, EndChar: 17},
		{Token: "parlé", StartByte: 18, EndByte: 24, StartChar: 18, EndChar: 23},
		{Token: "avec", StartByte: 25, EndByte: 29, StartChar: 24, EndChar: 28},
		{Token: "hydro", StartByte: 30, EndByte: 35, StartChar: 29, EndChar: 34},
		{Token: "-", StartByte: 35, EndByte: 36, StartChar: 34, EndChar: 35},
		{Token: "québec", StartByte: 36, EndByte: 43, StartChar: 35, EndChar: 41},
		{Token: "à", StartByte: 44, EndByte: 46, StartChar: 42, EndChar: 43},
		{Token: "montréal", StartByte: 47, EndByte: 56, StartChar: 44, EndChar: 52},
	}, spans)
}

func TestLegacyTokenPackDoesNotOverclaimRawText(t *testing.T) {
	pack, err := packs.LoadTokenPack("distilcamembert-french-ner")
	require.NoError(t, err)
	defer pack.Close()

	require.False(t, pack.SupportsRawText())

	_, err = pack.TokenizeText("Jean Dupont a rencontré Airbus à Paris.")
	require.Error(t, err)
}
