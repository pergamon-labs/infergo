package embeddings

import (
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSaveAndLoadVocab(t *testing.T) {
	t.Parallel()

	vocabPath := filepath.Join(t.TempDir(), "vocab.gob")
	expected := map[string]int{
		"hello": 1,
		"world": 2,
		"test":  3,
	}

	require.NoError(t, SaveVocab(expected, vocabPath))
	loaded, err := LoadVocab(vocabPath)
	require.NoError(t, err)
	assert.Equal(t, expected, loaded)
}

func TestSaveVocabError(t *testing.T) {
	t.Parallel()
	err := SaveVocab(map[string]int{"test": 1}, "/nonexistent/directory/vocab.gob")
	require.Error(t, err)
}

func TestLoadVocabError(t *testing.T) {
	t.Parallel()
	_, err := LoadVocab("/nonexistent/file.gob")
	require.Error(t, err)
}
