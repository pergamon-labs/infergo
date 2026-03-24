package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"time"

	"github.com/pergamon-labs/infergo/backends/bionet"
	"github.com/pergamon-labs/infergo/backends/bionet/runtime/functional"
	"github.com/pergamon-labs/infergo/backends/bionet/runtime/module"
	"github.com/pergamon-labs/infergo/backends/bionet/runtime/tensor"
	"github.com/pergamon-labs/infergo/internal/parity"
)

const (
	defaultReferencePath = "testdata/reference/text-classification/distilbert-sst2-reference.json"
	defaultOutputDir     = "testdata/native/text-classification/distilbert-sst2-token-id-bag"
	defaultArtifactName  = "model.gob"
	ridgeLambda          = 1e-9
)

func main() {
	referencePath := flag.String("reference", defaultReferencePath, "path to the Transformers reference JSON file")
	outputDir := flag.String("output-dir", defaultOutputDir, "directory to write the InferGo-native bundle into")
	flag.Parse()

	reference, err := parity.LoadTransformersTextClassificationReference(*referencePath)
	if err != nil {
		fatalf("load reference: %v", err)
	}

	metadata, classifier, err := buildTokenIDBagBundle(reference, defaultArtifactName)
	if err != nil {
		fatalf("build bundle: %v", err)
	}

	if err := os.MkdirAll(*outputDir, 0o755); err != nil {
		fatalf("create output dir: %v", err)
	}

	if err := module.SaveToFile(classifier, filepath.Join(*outputDir, metadata.Artifact)); err != nil {
		fatalf("save model artifact: %v", err)
	}

	if err := writeMetadata(filepath.Join(*outputDir, "metadata.json"), metadata); err != nil {
		fatalf("save metadata: %v", err)
	}

	fmt.Printf("wrote infergo-native bundle to %s\n", *outputDir)
}

func buildTokenIDBagBundle(reference parity.TransformersTextClassificationReference, artifactName string) (bionet.TextClassificationBundleMetadata, *module.Module, error) {
	featureTokenIDs := collectFeatureTokenIDs(reference)
	if len(featureTokenIDs) == 0 {
		return bionet.TextClassificationBundleMetadata{}, nil, fmt.Errorf("no feature token ids discovered in reference cases")
	}

	numCases := len(reference.Cases)
	numFeatures := len(featureTokenIDs)
	numLabels := len(reference.Labels)
	if numLabels == 0 {
		return bionet.TextClassificationBundleMetadata{}, nil, fmt.Errorf("reference labels are empty")
	}

	featureIndexByID := make(map[int]int, len(featureTokenIDs))
	for idx, tokenID := range featureTokenIDs {
		featureIndexByID[tokenID] = idx
	}

	design := make([][]float64, numCases)
	targets := make([][]float64, numCases)
	for caseIdx, item := range reference.Cases {
		features, err := featuresForCase(item, featureIndexByID, numFeatures)
		if err != nil {
			return bionet.TextClassificationBundleMetadata{}, nil, fmt.Errorf("build features for %q: %w", item.ID, err)
		}

		design[caseIdx] = append(features, 1.0)
		targets[caseIdx] = append([]float64(nil), item.ExpectedLogits...)
	}

	coefficients, err := solveRidgeRegression(design, targets, ridgeLambda)
	if err != nil {
		return bionet.TextClassificationBundleMetadata{}, nil, fmt.Errorf("fit linear classifier: %w", err)
	}

	weights := make([]float64, 0, numLabels*numFeatures)
	bias := make([]float64, numLabels)
	for labelIdx := 0; labelIdx < numLabels; labelIdx++ {
		for featureIdx := 0; featureIdx < numFeatures; featureIdx++ {
			weights = append(weights, coefficients[featureIdx][labelIdx])
		}
		bias[labelIdx] = coefficients[numFeatures][labelIdx]
	}

	classifier := module.New(functional.ActivationLinear, functional.ActivationParams{
		Weights: tensor.New(weights, []int{numLabels, numFeatures}),
		Bias:    tensor.New(bias, []int{numLabels}),
	})
	classifier.ModuleType = module.ModuleTypeFullyConnected

	metadata := bionet.TextClassificationBundleMetadata{
		Name:            fmt.Sprintf("%s InferGo-native bundle", reference.Name),
		Source:          "infergo-native-bundlegen",
		ModelID:         reference.ModelID,
		Task:            reference.Task,
		GeneratedAt:     time.Now().UTC().Format(time.RFC3339),
		Artifact:        artifactName,
		Labels:          append([]string(nil), reference.Labels...),
		FeatureMode:     bionet.TextClassificationFeatureModeTokenIDBag,
		FeatureTokenIDs: featureTokenIDs,
	}

	return metadata, classifier, nil
}

func collectFeatureTokenIDs(reference parity.TransformersTextClassificationReference) []int {
	seen := make(map[int]struct{})
	for _, item := range reference.Cases {
		for idx, tokenID := range item.InputIDs {
			if idx >= len(item.AttentionMask) || item.AttentionMask[idx] == 0 {
				continue
			}

			seen[tokenID] = struct{}{}
		}
	}

	featureTokenIDs := make([]int, 0, len(seen))
	for tokenID := range seen {
		featureTokenIDs = append(featureTokenIDs, tokenID)
	}
	sort.Ints(featureTokenIDs)
	return featureTokenIDs
}

func featuresForCase(item parity.TransformersTextClassificationReferenceCase, featureIndexByID map[int]int, numFeatures int) ([]float64, error) {
	if len(item.InputIDs) != len(item.AttentionMask) {
		return nil, fmt.Errorf("input_ids and attention_mask length mismatch (%d != %d)", len(item.InputIDs), len(item.AttentionMask))
	}

	features := make([]float64, numFeatures)
	for idx, tokenID := range item.InputIDs {
		if item.AttentionMask[idx] == 0 {
			continue
		}

		featureIdx, ok := featureIndexByID[tokenID]
		if !ok {
			continue
		}

		features[featureIdx]++
	}

	return features, nil
}

func solveRidgeRegression(design, targets [][]float64, lambda float64) ([][]float64, error) {
	if len(design) == 0 || len(targets) == 0 {
		return nil, fmt.Errorf("design and targets must not be empty")
	}

	if len(design) != len(targets) {
		return nil, fmt.Errorf("design/target row mismatch (%d != %d)", len(design), len(targets))
	}

	numSamples := len(design)
	numFeatures := len(design[0])
	numOutputs := len(targets[0])

	for rowIdx := range design {
		if len(design[rowIdx]) != numFeatures {
			return nil, fmt.Errorf("design row %d length mismatch", rowIdx)
		}
		if len(targets[rowIdx]) != numOutputs {
			return nil, fmt.Errorf("target row %d length mismatch", rowIdx)
		}
	}

	normal := make([][]float64, numFeatures)
	rhs := make([][]float64, numFeatures)
	for i := 0; i < numFeatures; i++ {
		normal[i] = make([]float64, numFeatures)
		rhs[i] = make([]float64, numOutputs)
	}

	for sampleIdx := 0; sampleIdx < numSamples; sampleIdx++ {
		for i := 0; i < numFeatures; i++ {
			for j := 0; j < numFeatures; j++ {
				normal[i][j] += design[sampleIdx][i] * design[sampleIdx][j]
			}
			for outputIdx := 0; outputIdx < numOutputs; outputIdx++ {
				rhs[i][outputIdx] += design[sampleIdx][i] * targets[sampleIdx][outputIdx]
			}
		}
	}

	for i := 0; i < numFeatures; i++ {
		normal[i][i] += lambda
	}

	coefficients := make([][]float64, numFeatures)
	for i := range coefficients {
		coefficients[i] = make([]float64, numOutputs)
	}

	for outputIdx := 0; outputIdx < numOutputs; outputIdx++ {
		column := make([]float64, numFeatures)
		for rowIdx := 0; rowIdx < numFeatures; rowIdx++ {
			column[rowIdx] = rhs[rowIdx][outputIdx]
		}

		solution, err := solveLinearSystem(copyMatrix(normal), column)
		if err != nil {
			return nil, err
		}

		for rowIdx := 0; rowIdx < numFeatures; rowIdx++ {
			coefficients[rowIdx][outputIdx] = solution[rowIdx]
		}
	}

	return coefficients, nil
}

func solveLinearSystem(a [][]float64, b []float64) ([]float64, error) {
	if len(a) == 0 || len(a) != len(b) {
		return nil, fmt.Errorf("invalid linear system dimensions")
	}

	n := len(a)
	augmented := make([][]float64, n)
	for i := 0; i < n; i++ {
		if len(a[i]) != n {
			return nil, fmt.Errorf("linear system is not square")
		}

		augmented[i] = append(append([]float64(nil), a[i]...), b[i])
	}

	for pivot := 0; pivot < n; pivot++ {
		bestRow := pivot
		bestValue := abs(augmented[pivot][pivot])
		for row := pivot + 1; row < n; row++ {
			value := abs(augmented[row][pivot])
			if value > bestValue {
				bestValue = value
				bestRow = row
			}
		}

		if bestValue < 1e-12 {
			return nil, fmt.Errorf("linear system is singular")
		}

		if bestRow != pivot {
			augmented[pivot], augmented[bestRow] = augmented[bestRow], augmented[pivot]
		}

		pivotValue := augmented[pivot][pivot]
		for col := pivot; col <= n; col++ {
			augmented[pivot][col] /= pivotValue
		}

		for row := 0; row < n; row++ {
			if row == pivot {
				continue
			}

			factor := augmented[row][pivot]
			if factor == 0 {
				continue
			}

			for col := pivot; col <= n; col++ {
				augmented[row][col] -= factor * augmented[pivot][col]
			}
		}
	}

	solution := make([]float64, n)
	for i := 0; i < n; i++ {
		solution[i] = augmented[i][n]
	}

	return solution, nil
}

func copyMatrix(input [][]float64) [][]float64 {
	output := make([][]float64, len(input))
	for i := range input {
		output[i] = append([]float64(nil), input[i]...)
	}
	return output
}

func writeMetadata(path string, metadata bionet.TextClassificationBundleMetadata) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	return encoder.Encode(metadata)
}

func abs(value float64) float64 {
	if value < 0 {
		return -value
	}
	return value
}

func fatalf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "infergo-native-bundlegen: "+format+"\n", args...)
	os.Exit(1)
}
