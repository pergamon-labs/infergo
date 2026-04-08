package main

import (
	"flag"
	"log"
	"path/filepath"

	"github.com/pergamon-labs/infergo/infer/experimental/entres"
)

func main() {
	modelPath := flag.String("model", "", "path to the source TorchScript .pt artifact")
	outDir := flag.String("out", "", "output family-2 bundle directory")
	modelID := flag.String("model-id", "", "source model identifier to record in metadata")
	profileKind := flag.String("profile-kind", "individual", "entity profile kind: individual or organization")
	copyArtifact := flag.Bool("copy-artifact", false, "copy the .pt artifact into the bundle instead of symlinking it")
	flag.Parse()

	if *modelPath == "" {
		log.Fatal("model path is required")
	}
	if *outDir == "" {
		log.Fatal("output dir is required")
	}

	spec := entres.BundleSpec{
		ModelPath:    *modelPath,
		OutputDir:    *outDir,
		ModelID:      *modelID,
		ProfileKind:  *profileKind,
		CopyArtifact: *copyArtifact,
	}
	if err := entres.ScaffoldBundle(spec); err != nil {
		log.Fatal(err)
	}

	log.Printf("InferGo entres bridge bundle created at %s", filepath.Clean(*outDir))
}
