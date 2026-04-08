package entres

import (
	"os"
	"path/filepath"
	"testing"
)

func TestScaffoldBundleCreatesMetadataAndSymlink(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	modelPath := filepath.Join(dir, "source.pt")
	if err := os.WriteFile(modelPath, []byte("fixture"), 0o644); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	outDir := filepath.Join(dir, "bundle")
	if err := ScaffoldBundle(BundleSpec{
		ModelPath:   modelPath,
		OutputDir:   outDir,
		ModelID:     "pergamon/entres-individual",
		ProfileKind: "individual",
	}); err != nil {
		t.Fatalf("ScaffoldBundle() error = %v", err)
	}

	metadataPath := filepath.Join(outDir, "metadata.json")
	if _, err := os.Stat(metadataPath); err != nil {
		t.Fatalf("metadata.json stat error = %v", err)
	}

	artifactPath := filepath.Join(outDir, DefaultArtifactName)
	info, err := os.Lstat(artifactPath)
	if err != nil {
		t.Fatalf("artifact stat error = %v", err)
	}
	if info.Mode()&os.ModeSymlink == 0 {
		t.Fatalf("expected symlink artifact, mode=%v", info.Mode())
	}
}

func TestScaffoldBundleRejectsUnknownProfileKind(t *testing.T) {
	t.Parallel()

	err := ScaffoldBundle(BundleSpec{
		ModelPath:   "/tmp/model.pt",
		OutputDir:   t.TempDir(),
		ProfileKind: "unknown",
	})
	if err == nil {
		t.Fatal("expected error")
	}
}
