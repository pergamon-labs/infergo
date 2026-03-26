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
	defaultReferencePath = "testdata/reference/token-classification/distilbert-ner-reference.json"
	defaultOutputDir     = "testdata/native/token-classification/distilbert-ner-embedding-linear"
	defaultArtifactName  = "model.gob"
	defaultEmbeddingName = "embeddings.gob"
	ridgeLambda          = 1e-9
)

func main() {
	referencePath := flag.String("reference", defaultReferencePath, "path to the Transformers token-classification reference JSON file")
	outputDir := flag.String("output-dir", defaultOutputDir, "directory to write the InferGo-native token-classification bundle into")
	flag.Parse()

	reference, err := parity.LoadTransformersTokenClassificationReference(*referencePath)
	if err != nil {
		fatalf("load reference: %v", err)
	}

	metadata, classifier, embeddingMatrix, err := buildBundle(reference, defaultArtifactName, defaultEmbeddingName)
	if err != nil {
		fatalf("build bundle: %v", err)
	}

	if err := os.MkdirAll(*outputDir, 0o755); err != nil {
		fatalf("create output dir: %v", err)
	}

	if err := module.SaveToFile(classifier, filepath.Join(*outputDir, metadata.Artifact)); err != nil {
		fatalf("save model artifact: %v", err)
	}

	if err := bionet.SaveTensorToFile(embeddingMatrix, filepath.Join(*outputDir, metadata.EmbeddingArtifact)); err != nil {
		fatalf("save embedding artifact: %v", err)
	}

	if err := writeMetadata(filepath.Join(*outputDir, "metadata.json"), metadata); err != nil {
		fatalf("save metadata: %v", err)
	}

	fmt.Printf("wrote infergo-native token-classification bundle to %s\n", *outputDir)
}

func buildBundle(reference parity.TransformersTokenClassificationReference, artifactName, embeddingArtifactName string) (bionet.TokenClassificationBundleMetadata, *module.Module, tensor.Tensor, error) {
	featureTokenIDs := collectFeatureTokenIDs(reference)
	if len(featureTokenIDs) == 0 {
		return bionet.TokenClassificationBundleMetadata{}, nil, tensor.Tensor{}, fmt.Errorf("no feature token ids discovered in reference cases")
	}

	numLabels := len(reference.Labels)
	if numLabels == 0 {
		return bionet.TokenClassificationBundleMetadata{}, nil, tensor.Tensor{}, fmt.Errorf("reference labels are empty")
	}

	numFeatures := len(featureTokenIDs)
	featureIndexByID := make(map[int]int, len(featureTokenIDs))
	for idx, tokenID := range featureTokenIDs {
		featureIndexByID[tokenID] = idx
	}

	var design [][]float64
	var targets [][]float64
	for _, item := range reference.Cases {
		if len(item.InputIDs) != len(item.ScoringMask) || len(item.InputIDs) != len(item.ExpectedLogits) {
			return bionet.TokenClassificationBundleMetadata{}, nil, tensor.Tensor{}, fmt.Errorf("reference case %q has inconsistent lengths", item.ID)
		}

		for pos, tokenID := range item.InputIDs {
			if item.ScoringMask[pos] == 0 {
				continue
			}

			row := make([]float64, numFeatures+1)
			row[featureIndexByID[tokenID]] = 1
			row[numFeatures] = 1

			design = append(design, row)
			targets = append(targets, append([]float64(nil), item.ExpectedLogits[pos]...))
		}
	}

	coefficients, err := solveRidgeRegression(design, targets, ridgeLambda)
	if err != nil {
		return bionet.TokenClassificationBundleMetadata{}, nil, tensor.Tensor{}, fmt.Errorf("fit token classifier: %w", err)
	}

	embeddingMatrix, headWeights, bias := denseEmbeddingArtifactsFromCoefficients(coefficients, numFeatures, numLabels)

	classifier := module.New(functional.ActivationLinear, functional.ActivationParams{
		Weights: headWeights,
		Bias:    tensor.New(bias, []int{numLabels}),
	})
	classifier.ModuleType = module.ModuleTypeFullyConnected

	metadata := bionet.TokenClassificationBundleMetadata{
		Name:              fmt.Sprintf("%s InferGo-native bundle", reference.Name),
		Source:            "infergo-native-token-bundlegen",
		ModelID:           reference.ModelID,
		Task:              reference.Task,
		GeneratedAt:       time.Now().UTC().Format(time.RFC3339),
		Artifact:          artifactName,
		EmbeddingArtifact: embeddingArtifactName,
		Labels:            append([]string(nil), reference.Labels...),
		FeatureMode:       bionet.TokenClassificationFeatureModeEmbeddingLinear,
		FeatureTokenIDs:   featureTokenIDs,
	}

	return metadata, classifier, embeddingMatrix, nil
}

func collectFeatureTokenIDs(reference parity.TransformersTokenClassificationReference) []int {
	seen := make(map[int]struct{})
	for _, item := range reference.Cases {
		for pos, tokenID := range item.InputIDs {
			if item.ScoringMask[pos] == 0 {
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

func denseEmbeddingArtifactsFromCoefficients(coefficients [][]float64, numFeatures, numLabels int) (tensor.Tensor, tensor.Tensor, []float64) {
	embeddingValues := make([]float64, 0, numFeatures*numLabels)
	for featureIdx := 0; featureIdx < numFeatures; featureIdx++ {
		for labelIdx := 0; labelIdx < numLabels; labelIdx++ {
			embeddingValues = append(embeddingValues, coefficients[featureIdx][labelIdx])
		}
	}

	headValues := make([]float64, numLabels*numLabels)
	for labelIdx := 0; labelIdx < numLabels; labelIdx++ {
		headValues[labelIdx*numLabels+labelIdx] = 1
	}

	bias := make([]float64, numLabels)
	for labelIdx := 0; labelIdx < numLabels; labelIdx++ {
		bias[labelIdx] = coefficients[numFeatures][labelIdx]
	}

	return tensor.New(embeddingValues, []int{numFeatures, numLabels}), tensor.New(headValues, []int{numLabels, numLabels}), bias
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

func solveLinearSystem(matrix [][]float64, values []float64) ([]float64, error) {
	n := len(matrix)
	if n == 0 {
		return nil, fmt.Errorf("matrix must not be empty")
	}

	for pivot := 0; pivot < n; pivot++ {
		bestRow := pivot
		bestValue := abs(matrix[pivot][pivot])
		for row := pivot + 1; row < n; row++ {
			value := abs(matrix[row][pivot])
			if value > bestValue {
				bestValue = value
				bestRow = row
			}
		}

		if bestValue == 0 {
			return nil, fmt.Errorf("matrix is singular")
		}

		if bestRow != pivot {
			matrix[pivot], matrix[bestRow] = matrix[bestRow], matrix[pivot]
			values[pivot], values[bestRow] = values[bestRow], values[pivot]
		}

		diag := matrix[pivot][pivot]
		for col := pivot; col < n; col++ {
			matrix[pivot][col] /= diag
		}
		values[pivot] /= diag

		for row := 0; row < n; row++ {
			if row == pivot {
				continue
			}
			factor := matrix[row][pivot]
			if factor == 0 {
				continue
			}
			for col := pivot; col < n; col++ {
				matrix[row][col] -= factor * matrix[pivot][col]
			}
			values[row] -= factor * values[pivot]
		}
	}

	return values, nil
}

func copyMatrix(matrix [][]float64) [][]float64 {
	out := make([][]float64, len(matrix))
	for i := range matrix {
		out[i] = append([]float64(nil), matrix[i]...)
	}
	return out
}

func writeMetadata(path string, metadata bionet.TokenClassificationBundleMetadata) error {
	raw, err := json.MarshalIndent(metadata, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal metadata: %w", err)
	}

	if err := os.WriteFile(path, append(raw, '\n'), 0o644); err != nil {
		return fmt.Errorf("write metadata: %w", err)
	}
	return nil
}

func abs(value float64) float64 {
	if value < 0 {
		return -value
	}
	return value
}

func fatalf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(1)
}
