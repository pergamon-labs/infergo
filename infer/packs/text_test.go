package packs_test

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/pergamon-labs/infergo/infer/packs"
)

func TestListTextPacks(t *testing.T) {
	items, err := packs.ListTextPacks()
	require.NoError(t, err)
	require.NotEmpty(t, items)
}

func TestLoadTextPackPredictText(t *testing.T) {
	pack, err := packs.LoadTextPack("infergo-basic-sst2")
	require.NoError(t, err)
	defer pack.Close()

	require.True(t, pack.SupportsRawText())

	prediction, err := pack.PredictText("This product is excellent and reliable.")
	require.NoError(t, err)
	require.Equal(t, "POSITIVE", prediction.Label)
}

func TestLegacyTextPackDoesNotOverclaimRawText(t *testing.T) {
	pack, err := packs.LoadTextPack("distilbert-sst2")
	require.NoError(t, err)
	defer pack.Close()

	require.False(t, pack.SupportsRawText())
}

func TestLoadTextPackPredictTokens(t *testing.T) {
	pack, err := packs.LoadTextPack("twitter-xlm-roberta-sentiment-multilingual")
	require.NoError(t, err)
	defer pack.Close()

	require.False(t, pack.SupportsRawText())

	prediction, err := pack.PredictTokens([]string{"▁Este", "▁producto", "▁es", "▁excelente", "▁y", "▁muy", "▁facil", "▁de", "▁usar", "."})
	require.NoError(t, err)
	require.Equal(t, "positive", prediction.Label)
}
