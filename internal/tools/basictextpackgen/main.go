package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"time"

	runtimeTokenizer "github.com/pergamon-labs/infergo/backends/bionet/runtime/tokenizer"
	"github.com/pergamon-labs/infergo/internal/parity"
)

func main() {
	sourceReferencePath := flag.String("source-reference", "", "path to the source text-classification reference JSON file")
	outputPath := flag.String("output", "", "path to write the projected native text-classification reference JSON file")
	modelID := flag.String("model-id", "", "model id to record in the projected reference")
	name := flag.String("name", "", "reference name to record in the projected reference")
	flag.Parse()

	if *sourceReferencePath == "" {
		fatalf("source-reference is required")
	}
	if *outputPath == "" {
		fatalf("output is required")
	}
	if *modelID == "" {
		fatalf("model-id is required")
	}

	sourceReference, err := parity.LoadTransformersTextClassificationReference(*sourceReferencePath)
	if err != nil {
		fatalf("load source reference: %v", err)
	}

	projected, err := projectReference(sourceReference, *modelID, *name)
	if err != nil {
		fatalf("project reference: %v", err)
	}

	if err := os.MkdirAll(filepath.Dir(*outputPath), 0o755); err != nil {
		fatalf("create output dir: %v", err)
	}

	raw, err := json.MarshalIndent(projected, "", "  ")
	if err != nil {
		fatalf("encode projected reference: %v", err)
	}

	if err := os.WriteFile(*outputPath, append(raw, '\n'), 0o644); err != nil {
		fatalf("write projected reference: %v", err)
	}

	fmt.Printf("wrote projected text reference to %s\n", *outputPath)
}

func projectReference(source parity.TransformersTextClassificationReference, modelID, name string) (parity.TransformersTextClassificationReference, error) {
	if name == "" {
		name = fmt.Sprintf("%s BasicTokenizer Projection", source.Name)
	}

	vocab, err := collectVocabulary(source)
	if err != nil {
		return parity.TransformersTextClassificationReference{}, err
	}

	tokenIDs := make(map[string]int, len(vocab))
	for idx, token := range vocab {
		tokenIDs[token] = idx
	}

	projectedCases := make([]parity.TransformersTextClassificationReferenceCase, 0, len(source.Cases))
	for _, item := range source.Cases {
		if item.TextPair != "" {
			return parity.TransformersTextClassificationReference{}, fmt.Errorf("source case %q uses text_pair; basic tokenizer projection only supports single-text references", item.ID)
		}

		tokens := runtimeTokenizer.BasicTokenizer(item.Text, 0)
		if len(tokens) == 0 {
			return parity.TransformersTextClassificationReference{}, fmt.Errorf("source case %q tokenized to zero tokens", item.ID)
		}

		inputIDs := make([]int, len(tokens))
		attentionMask := make([]int, len(tokens))
		for i, token := range tokens {
			tokenID, ok := tokenIDs[token]
			if !ok {
				return parity.TransformersTextClassificationReference{}, fmt.Errorf("source case %q token %q missing from projected vocab", item.ID, token)
			}
			inputIDs[i] = tokenID
			attentionMask[i] = 1
		}

		projectedCases = append(projectedCases, parity.TransformersTextClassificationReferenceCase{
			ID:                    item.ID,
			Text:                  item.Text,
			Tokens:                tokens,
			InputIDs:              inputIDs,
			AttentionMask:         attentionMask,
			ExpectedLogits:        append([]float64(nil), item.ExpectedLogits...),
			ExpectedProbabilities: append([]float64(nil), item.ExpectedProbabilities...),
			ExpectedLabel:         item.ExpectedLabel,
		})
	}

	return parity.TransformersTextClassificationReference{
		Name:                name,
		Source:              fmt.Sprintf("infergo-basic-tokenizer-projection:%s", source.ModelID),
		ModelID:             modelID,
		Task:                source.Task,
		GeneratedAt:         time.Now().UTC().Format(time.RFC3339),
		TransformersVersion: source.TransformersVersion,
		TorchVersion:        source.TorchVersion,
		Labels:              append([]string(nil), source.Labels...),
		Cases:               projectedCases,
	}, nil
}

func collectVocabulary(source parity.TransformersTextClassificationReference) ([]string, error) {
	seen := make(map[string]struct{})
	for _, item := range source.Cases {
		tokens := runtimeTokenizer.BasicTokenizer(item.Text, 0)
		if len(tokens) == 0 {
			return nil, fmt.Errorf("source case %q tokenized to zero tokens", item.ID)
		}
		for _, token := range tokens {
			seen[token] = struct{}{}
		}
	}

	vocab := make([]string, 0, len(seen))
	for token := range seen {
		vocab = append(vocab, token)
	}
	sort.Strings(vocab)
	return vocab, nil
}

func fatalf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(1)
}
