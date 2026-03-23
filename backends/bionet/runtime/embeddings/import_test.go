package embeddings

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestImportGloVe(t *testing.T) {
	t.Parallel()

	t.Run("imports embeddings successfully", func(t *testing.T) {
		t.Parallel()

		dir := t.TempDir()
		glovePath := filepath.Join(dir, "mini-glove.txt")
		require.NoError(t, os.WriteFile(glovePath, []byte("hello 0.1 0.2\nworld 0.3 0.4\n"), 0o644))

		embeddingTensor, vocabulary, err := ImportGloVe(glovePath)
		require.NoError(t, err)
		assert.Equal(t, []int{2, 2}, embeddingTensor.Shape())
		assert.Equal(t, map[string]int{"hello": 0, "world": 1}, vocabulary)
		assert.Equal(t, []float64{0.1, 0.2, 0.3, 0.4}, embeddingTensor.Values())
	})

	t.Run("fails when file does not exist", func(t *testing.T) {
		t.Parallel()

		embeddingTensor, vocabulary, err := ImportGloVe("/tmp/does-not-exist-glove.txt")
		require.Error(t, err)
		assert.Empty(t, embeddingTensor.Values())
		assert.Nil(t, vocabulary)
	})

	t.Run("fails on inconsistent dimension", func(t *testing.T) {
		t.Parallel()

		dir := t.TempDir()
		glovePath := filepath.Join(dir, "bad-glove.txt")
		require.NoError(t, os.WriteFile(glovePath, []byte("hello 0.1 0.2\nworld 0.3\n"), 0o644))

		_, _, err := ImportGloVe(glovePath)
		require.Error(t, err)
	})
}
