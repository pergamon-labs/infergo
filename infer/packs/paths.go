package packs

import (
	"path/filepath"
	"runtime"
)

const (
	textManifestPath  = "testdata/reference/text-classification/model-packs.json"
	tokenManifestPath = "testdata/reference/token-classification/model-packs.json"
)

func moduleRoot() string {
	_, currentFile, _, ok := runtime.Caller(0)
	if !ok {
		return "."
	}

	return filepath.Clean(filepath.Join(filepath.Dir(currentFile), "..", ".."))
}

func modulePath(rel string) string {
	return filepath.Join(moduleRoot(), filepath.FromSlash(rel))
}
