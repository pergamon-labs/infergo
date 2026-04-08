package entres

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

const (
	IndividualVectorSize   = 268
	OrganizationVectorSize = 212
	DefaultArtifactName    = "model.torchscript.pt"
)

// BundleSpec describes the minimal family-2 bridge bundle to scaffold.
type BundleSpec struct {
	ModelPath    string
	OutputDir    string
	ModelID      string
	ProfileKind  string
	CopyArtifact bool
}

// ScaffoldBundle creates a family-2 bridge bundle directory around a real
// TorchScript artifact.
func ScaffoldBundle(spec BundleSpec) error {
	if spec.ModelPath == "" {
		return fmt.Errorf("scaffold entres bundle: model path is required")
	}
	if spec.OutputDir == "" {
		return fmt.Errorf("scaffold entres bundle: output dir is required")
	}
	vectorSize, err := vectorSizeForProfileKind(spec.ProfileKind)
	if err != nil {
		return err
	}

	if spec.ModelID == "" {
		spec.ModelID = filepath.Base(spec.ModelPath)
	}

	if err := os.MkdirAll(spec.OutputDir, 0o755); err != nil {
		return fmt.Errorf("scaffold entres bundle: create output dir: %w", err)
	}

	artifactPath := filepath.Join(spec.OutputDir, DefaultArtifactName)
	if err := os.RemoveAll(artifactPath); err != nil {
		return fmt.Errorf("scaffold entres bundle: clear artifact path: %w", err)
	}
	if err := materializeArtifact(spec.ModelPath, artifactPath, spec.CopyArtifact); err != nil {
		return err
	}

	metadata := map[string]any{
		"bundle_format":  "infergo-torchscript-bridge",
		"bundle_version": "1.0",
		"family":         "numeric-feature-scoring",
		"task":           "entity-resolution-scoring",
		"backend":        "torchscript",
		"artifact":       DefaultArtifactName,
		"model_id":       spec.ModelID,
		"profile_kind":   spec.ProfileKind,
		"source": map[string]any{
			"framework": "pytorch",
			"format":    "torchscript",
		},
		"inputs": map[string]any{
			"vector_size":        vectorSize,
			"message_size":       vectorSize,
			"input_layout":       "stacked_sample_message_channels",
			"message_strategy":   "caller_supplied_consensus_vector",
			"message_projection": "legacy_first_value_broadcast",
		},
		"outputs": map[string]any{
			"kind":           "score_vector",
			"interpretation": "confidence",
		},
	}

	metadataPath := filepath.Join(spec.OutputDir, "metadata.json")
	data, err := json.MarshalIndent(metadata, "", "  ")
	if err != nil {
		return fmt.Errorf("scaffold entres bundle: encode metadata: %w", err)
	}
	data = append(data, '\n')
	if err := os.WriteFile(metadataPath, data, 0o644); err != nil {
		return fmt.Errorf("scaffold entres bundle: write metadata: %w", err)
	}

	return nil
}

func vectorSizeForProfileKind(profileKind string) (int, error) {
	switch profileKind {
	case "individual":
		return IndividualVectorSize, nil
	case "organization":
		return OrganizationVectorSize, nil
	default:
		return 0, fmt.Errorf("scaffold entres bundle: unsupported profile kind %q", profileKind)
	}
}

func materializeArtifact(src, dst string, copyArtifact bool) error {
	absSrc, err := filepath.Abs(src)
	if err != nil {
		return fmt.Errorf("scaffold entres bundle: resolve source artifact path: %w", err)
	}
	absDst, err := filepath.Abs(dst)
	if err != nil {
		return fmt.Errorf("scaffold entres bundle: resolve destination artifact path: %w", err)
	}

	if copyArtifact {
		in, err := os.Open(absSrc)
		if err != nil {
			return fmt.Errorf("scaffold entres bundle: open source artifact: %w", err)
		}
		defer in.Close()

		out, err := os.Create(absDst)
		if err != nil {
			return fmt.Errorf("scaffold entres bundle: create artifact copy: %w", err)
		}
		defer out.Close()

		if _, err := io.Copy(out, in); err != nil {
			return fmt.Errorf("scaffold entres bundle: copy artifact: %w", err)
		}
		return nil
	}

	relSrc, err := filepath.Rel(filepath.Dir(absDst), absSrc)
	if err != nil {
		return fmt.Errorf("scaffold entres bundle: compute symlink path: %w", err)
	}
	if err := os.Symlink(relSrc, absDst); err != nil {
		return fmt.Errorf("scaffold entres bundle: create symlink: %w", err)
	}
	return nil
}
