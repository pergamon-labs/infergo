package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
	"unicode"

	runtimeTokenizer "github.com/pergamon-labs/infergo/backends/bionet/runtime/tokenizer"
	"github.com/pergamon-labs/infergo/internal/parity"
)

type sourceToken struct {
	normalized string
	logits     []float64
}

func main() {
	sourceReferencePath := flag.String("source-reference", "", "path to the source token-classification reference JSON file")
	outputPath := flag.String("output", "", "path to write the projected native token-classification reference JSON file")
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

	sourceReference, err := parity.LoadTransformersTokenClassificationReference(*sourceReferencePath)
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

	fmt.Printf("wrote projected token reference to %s\n", *outputPath)
}

func projectReference(source parity.TransformersTokenClassificationReference, modelID, name string) (parity.TransformersTokenClassificationReference, error) {
	if name == "" {
		name = fmt.Sprintf("%s BasicTokenizer Projection", source.Name)
	}

	vocab, err := collectVocabulary(source)
	if err != nil {
		return parity.TransformersTokenClassificationReference{}, err
	}

	tokenIDs := make(map[string]int, len(vocab))
	for idx, token := range vocab {
		tokenIDs[token] = idx
	}

	projectedCases := make([]parity.TransformersTokenClassificationReferenceCase, 0, len(source.Cases))
	for _, item := range source.Cases {
		projectedCase, err := projectCase(item, source.Labels, tokenIDs)
		if err != nil {
			return parity.TransformersTokenClassificationReference{}, fmt.Errorf("project case %q: %w", item.ID, err)
		}
		projectedCases = append(projectedCases, projectedCase)
	}

	return parity.TransformersTokenClassificationReference{
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

func projectCase(item parity.TransformersTokenClassificationReferenceCase, labels []string, tokenIDs map[string]int) (parity.TransformersTokenClassificationReferenceCase, error) {
	basicTokens := runtimeTokenizer.BasicTokenizer(item.Text, 0)
	if len(basicTokens) == 0 {
		return parity.TransformersTokenClassificationReferenceCase{}, fmt.Errorf("tokenized to zero tokens")
	}

	sourceTokens, err := collectSourceTokens(item)
	if err != nil {
		return parity.TransformersTokenClassificationReferenceCase{}, err
	}

	inputIDs := make([]int, len(basicTokens))
	attentionMask := make([]int, len(basicTokens))
	scoringMask := make([]int, len(basicTokens))
	expectedLogits := make([][]float64, len(basicTokens))
	expectedProbabilities := make([][]float64, len(basicTokens))
	expectedLabels := make([]string, len(basicTokens))

	cursor := 0
	for idx, token := range basicTokens {
		tokenID, ok := tokenIDs[token]
		if !ok {
			return parity.TransformersTokenClassificationReferenceCase{}, fmt.Errorf("token %q missing from projected vocab", token)
		}

		group, nextCursor, err := matchSourceGroup(sourceTokens, cursor, token)
		if err != nil {
			return parity.TransformersTokenClassificationReferenceCase{}, fmt.Errorf("align token %q: %w", token, err)
		}
		cursor = nextCursor

		logits := averageLogits(group)
		probabilities := softmax(logits)
		label := labels[argmax(logits)]

		inputIDs[idx] = tokenID
		attentionMask[idx] = 1
		scoringMask[idx] = scoreToken(token)
		expectedLogits[idx] = logits
		expectedProbabilities[idx] = probabilities
		expectedLabels[idx] = label
	}

	if cursor != len(sourceTokens) {
		return parity.TransformersTokenClassificationReferenceCase{}, fmt.Errorf("left %d unmatched source token groups", len(sourceTokens)-cursor)
	}

	return parity.TransformersTokenClassificationReferenceCase{
		ID:                    item.ID,
		Text:                  item.Text,
		Tokens:                basicTokens,
		InputIDs:              inputIDs,
		AttentionMask:         attentionMask,
		ScoringMask:           scoringMask,
		ExpectedLogits:        expectedLogits,
		ExpectedProbabilities: expectedProbabilities,
		ExpectedLabels:        expectedLabels,
	}, nil
}

func collectVocabulary(source parity.TransformersTokenClassificationReference) ([]string, error) {
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

func collectSourceTokens(item parity.TransformersTokenClassificationReferenceCase) ([]sourceToken, error) {
	if len(item.Tokens) != len(item.ExpectedLogits) {
		return nil, fmt.Errorf("token/logit length mismatch")
	}

	output := make([]sourceToken, 0, len(item.Tokens))
	for idx, token := range item.Tokens {
		normalized := normalizeSourceToken(token)
		if normalized == "" {
			continue
		}
		output = append(output, sourceToken{
			normalized: normalized,
			logits:     append([]float64(nil), item.ExpectedLogits[idx]...),
		})
	}
	if len(output) == 0 {
		return nil, fmt.Errorf("no non-special source tokens")
	}
	return output, nil
}

func matchSourceGroup(tokens []sourceToken, start int, target string) ([]sourceToken, int, error) {
	normalizedTarget := strings.ToLower(target)
	if start >= len(tokens) {
		return nil, start, fmt.Errorf("no source tokens remaining")
	}

	combined := ""
	group := make([]sourceToken, 0, 2)
	cursor := start
	for cursor < len(tokens) {
		next := combined + tokens[cursor].normalized
		if !strings.HasPrefix(normalizedTarget, next) {
			break
		}
		group = append(group, tokens[cursor])
		combined = next
		cursor++
		if combined == normalizedTarget {
			return group, cursor, nil
		}
	}

	return nil, start, fmt.Errorf("could not align target %q", target)
}

func normalizeSourceToken(token string) string {
	if isSpecialToken(token) {
		return ""
	}

	normalized := token
	for {
		changed := false
		for _, prefix := range []string{"##", "▁", "Ġ", "Ċ"} {
			if strings.HasPrefix(normalized, prefix) {
				normalized = strings.TrimPrefix(normalized, prefix)
				changed = true
			}
		}
		if !changed {
			break
		}
	}

	normalized = strings.TrimSuffix(normalized, "</w>")
	return strings.ToLower(normalized)
}

func isSpecialToken(token string) bool {
	if token == "" {
		return true
	}
	if strings.HasPrefix(token, "<") && strings.HasSuffix(token, ">") {
		return true
	}
	if strings.HasPrefix(token, "[") && strings.HasSuffix(token, "]") {
		return true
	}
	return false
}

func averageLogits(group []sourceToken) []float64 {
	output := make([]float64, len(group[0].logits))
	for _, item := range group {
		for idx, value := range item.logits {
			output[idx] += value
		}
	}
	for idx := range output {
		output[idx] /= float64(len(group))
	}
	return output
}

func softmax(values []float64) []float64 {
	maxValue := values[0]
	for _, value := range values[1:] {
		if value > maxValue {
			maxValue = value
		}
	}

	output := make([]float64, len(values))
	total := 0.0
	for idx, value := range values {
		output[idx] = math.Exp(value - maxValue)
		total += output[idx]
	}
	for idx := range output {
		output[idx] /= total
	}
	return output
}

func argmax(values []float64) int {
	bestIdx := 0
	bestValue := values[0]
	for idx, value := range values[1:] {
		if value > bestValue {
			bestIdx = idx + 1
			bestValue = value
		}
	}
	return bestIdx
}

func scoreToken(token string) int {
	for _, r := range token {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			return 1
		}
	}
	return 0
}

func fatalf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(1)
}
