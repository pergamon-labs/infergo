package main

import (
	"bytes"
	"embed"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/pergamon-labs/infergo/backends/bionet"
	"github.com/pergamon-labs/infergo/internal/nativebundlegen"
	"github.com/pergamon-labs/infergo/internal/parity"
)

//go:embed assets/family1_reference_and_tokenizer.py
var embeddedAssets embed.FS

const (
	exportToolVersion = "0.1.0-alpha"

	defaultMaxLength = 128
	defaultRunner    = "uv"

	exportReadmePath          = "cmd/infergo-export/README.md"
	alphaTokenizerRuntimeKind = "hf-tokenizer-json"
)

var (
	commonPositiveLabels = []string{"positive", "match", "entailment", "duplicate"}
	commonNegativeLabels = []string{"negative", "non_match", "contradiction", "not_duplicate"}
)

type tokenizerManifest struct {
	Kind              string            `json:"kind"`
	RawTextSupported  bool              `json:"raw_text_supported"`
	PairTextSupported bool              `json:"pair_text_supported"`
	SpecialTokens     map[string]string `json:"special_tokens"`
	Files             map[string]string `json:"files"`
}

type alphaMetadata struct {
	BundleFormat    string             `json:"bundle_format"`
	BundleVersion   string             `json:"bundle_version"`
	Family          string             `json:"family"`
	Task            string             `json:"task"`
	Backend         string             `json:"backend"`
	BackendArtifact string             `json:"backend_artifact"`
	ModelID         string             `json:"model_id"`
	Source          alphaSource        `json:"source"`
	Inputs          alphaInputs        `json:"inputs"`
	Tokenizer       alphaTokenizer     `json:"tokenizer,omitempty"`
	Outputs         alphaOutputs       `json:"outputs"`
	BackendConfig   alphaBackendConfig `json:"backend_config"`
	CreatedAt       string             `json:"created_at"`
	CreatedBy       alphaCreatedBy     `json:"created_by"`
}

type alphaSource struct {
	Framework string `json:"framework"`
	Ecosystem string `json:"ecosystem"`
	RepoURL   string `json:"repo_url,omitempty"`
}

type alphaInputs struct {
	RawTextSupported        bool `json:"raw_text_supported"`
	PairTextSupported       bool `json:"pair_text_supported"`
	TokenizedInputSupported bool `json:"tokenized_input_supported"`
	MaxSequenceLength       int  `json:"max_sequence_length"`
}

type alphaTokenizer struct {
	Manifest string `json:"manifest"`
}

type alphaOutputs struct {
	Kind           string   `json:"kind"`
	LabelsArtifact string   `json:"labels_artifact"`
	PositiveLabel  string   `json:"positive_label,omitempty"`
	NegativeLabel  string   `json:"negative_label,omitempty"`
	Threshold      *float64 `json:"threshold,omitempty"`
}

type alphaBackendConfig struct {
	FeatureMode       string `json:"feature_mode"`
	FeatureTokenIDs   []int  `json:"feature_token_ids"`
	EmbeddingArtifact string `json:"embedding_artifact,omitempty"`
}

type alphaCreatedBy struct {
	Tool    string `json:"tool"`
	Version string `json:"version"`
}

type inputTemplate struct {
	Name  string                                           `json:"name"`
	Cases []parity.TransformersTextClassificationInputCase `json:"cases"`
}

func main() {
	if len(os.Args) == 1 {
		printUsage(os.Stderr)
		os.Exit(2)
	}
	if len(os.Args) == 2 {
		switch os.Args[1] {
		case "help", "-h", "--help":
			printUsage(os.Stdout)
			return
		}
	}

	command := os.Args[1]
	args := os.Args[2:]
	if strings.HasPrefix(command, "-") {
		command = "export"
		args = os.Args[1:]
	}

	var err error
	switch command {
	case "template":
		err = runTemplate(args)
	case "export":
		err = runExport(args)
	case "help":
		if len(args) == 0 {
			printUsage(os.Stdout)
			return
		}
		switch args[0] {
		case "template":
			printTemplateUsage(os.Stdout)
		case "export":
			printExportUsage(os.Stdout)
		default:
			err = fmt.Errorf("unknown help topic %q", args[0])
		}
		return
	default:
		err = fmt.Errorf("unknown subcommand %q", command)
	}

	if err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return
		}
		fmt.Fprintf(os.Stderr, "infergo-export: %v\n", err)
		fmt.Fprintf(os.Stderr, "Next help: infergo-export help export\n")
		fmt.Fprintf(os.Stderr, "Docs: %s\n", exportReadmePath)
		os.Exit(1)
	}
}

func printUsage(w io.Writer) {
	fmt.Fprintln(w, "Usage:")
	fmt.Fprintln(w, "  infergo-export template -kind single|pair -out <path>")
	fmt.Fprintln(w, "  infergo-export export -model <hf-id-or-local-dir> -input <input.json> -out <bundle-dir> [flags]")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "The export path is installable without a repo checkout, but it still needs")
	fmt.Fprintln(w, "Python/Transformers tooling at export time. By default it uses:")
	fmt.Fprintln(w, "  uv run --with torch==2.10.0 --with transformers==5.3.0")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Examples:")
	fmt.Fprintln(w, "  infergo-export template -kind single -out ./family1-inputs.json")
	fmt.Fprintln(w, "  infergo-export export -model distilbert/distilbert-base-uncased-finetuned-sst-2-english -input ./family1-inputs.json -out ./artifacts/distilbert-sst2-alpha")
	fmt.Fprintln(w, "  infergo-export help export")
	fmt.Fprintln(w)
	fmt.Fprintf(w, "Docs: %s\n", exportReadmePath)
}

func printTemplateUsage(w io.Writer) {
	fmt.Fprintln(w, "Usage:")
	fmt.Fprintln(w, "  infergo-export template -kind single|pair -out <path> [-force]")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Use this first when you do not already have a public-safe input file.")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Examples:")
	fmt.Fprintln(w, "  infergo-export template -kind single -out ./family1-inputs.json")
	fmt.Fprintln(w, "  infergo-export template -kind pair -out ./family1-pairs.json")
	fmt.Fprintln(w)
	fmt.Fprintf(w, "Docs: %s\n", exportReadmePath)
}

func printExportUsage(w io.Writer) {
	fmt.Fprintln(w, "Usage:")
	fmt.Fprintln(w, "  infergo-export export -model <hf-id-or-local-dir> -input <input.json> -out <bundle-dir> [flags]")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "This is the no-repo-checkout family-1 BYOM path.")
	fmt.Fprintln(w, "It still needs Python/Transformers tooling at export time.")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Most users want:")
	fmt.Fprintln(w, "  1. infergo-export template -kind single -out ./family1-inputs.json")
	fmt.Fprintln(w, "  2. edit that file with public-safe representative examples")
	fmt.Fprintln(w, "  3. infergo-export export -model <model-id> -input ./family1-inputs.json -out ./artifacts/my-bundle")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Examples:")
	fmt.Fprintln(w, "  infergo-export export -model distilbert/distilbert-base-uncased-finetuned-sst-2-english -input ./family1-inputs.json -out ./artifacts/distilbert-sst2-alpha")
	fmt.Fprintln(w, "  infergo-export export -model textattack/bert-base-uncased-MRPC -input ./family1-pairs.json -out ./artifacts/mrpc-alpha")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Important flags:")
	fmt.Fprintln(w, "  -model       Hugging Face model id or local model directory")
	fmt.Fprintln(w, "  -model-id    required when -model points to a local directory")
	fmt.Fprintln(w, "  -input       public-safe validation/parity input set")
	fmt.Fprintln(w, "  -out         output bundle directory")
	fmt.Fprintln(w, "  -python-runner uv (default) or python")
	fmt.Fprintln(w)
	fmt.Fprintf(w, "Docs: %s\n", exportReadmePath)
}

func runTemplate(args []string) error {
	fs := flag.NewFlagSet("template", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	fs.Usage = func() { printTemplateUsage(fs.Output()) }
	kind := fs.String("kind", "single", "template kind: single or pair")
	out := fs.String("out", "", "path to write the example input json")
	force := fs.Bool("force", false, "overwrite the output file if it already exists")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if strings.TrimSpace(*out) == "" {
		return errors.New("template: -out is required\nnext step: infergo-export template -kind single -out ./family1-inputs.json")
	}

	var template inputTemplate
	switch *kind {
	case "single":
		template = inputTemplate{
			Name: "replace-with-your-single-text-validation-set",
			Cases: []parity.TransformersTextClassificationInputCase{
				{ID: "case-1", Text: "Replace this with a representative positive or matching example."},
				{ID: "case-2", Text: "Replace this with a representative negative or non-matching example."},
				{ID: "case-3", Text: "Keep inputs public-safe and representative of your deployment traffic."},
			},
		}
	case "pair":
		template = inputTemplate{
			Name: "replace-with-your-paired-text-validation-set",
			Cases: []parity.TransformersTextClassificationInputCase{
				{ID: "pair-1", Text: "Company confirms the acquisition has completed.", TextPair: "The deal has officially closed, the company said."},
				{ID: "pair-2", Text: "Customer reported duplicate billing after checkout.", TextPair: "A user says they were charged twice."},
				{ID: "pair-3", Text: "Replace these examples with public-safe representative pairs from your task.", TextPair: "Use several examples so parity has something meaningful to compare."},
			},
		}
	default:
		return fmt.Errorf("template: unsupported kind %q (expected \"single\" or \"pair\")\nnext step: use -kind single for one-text inputs or -kind pair for text-pair inputs", *kind)
	}

	outputPath := filepath.Clean(*out)
	if !*force {
		if _, err := os.Stat(outputPath); err == nil {
			return fmt.Errorf("template: output already exists: %s (use -force to overwrite)", outputPath)
		}
	}
	return writeJSON(outputPath, template)
}

func runExport(args []string) error {
	fs := flag.NewFlagSet("export", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	fs.Usage = func() { printExportUsage(fs.Output()) }

	model := fs.String("model", "", "Hugging Face model id or local model directory")
	modelIDOverride := fs.String("model-id", "", "canonical model id to record in metadata; required when -model is a local path")
	inputPathFlag := fs.String("input", "", "path to the public-safe input json")
	outputDirFlag := fs.String("out", "", "output bundle directory")
	referenceOutputFlag := fs.String("reference-output", "", "optional path to persist the generated source reference json")
	featureMode := fs.String("feature-mode", bionet.TextClassificationFeatureModeEmbeddingMaskedAvgPool, "native BIOnet feature mode")
	maxLength := fs.Int("max-length", defaultMaxLength, "max tokenizer length passed to the source reference generator")
	bundleVersion := fs.String("bundle-version", "1.0", "alpha bundle version written into metadata.json")
	positiveLabel := fs.String("positive-label", "", "optional positive label override for binary bundles")
	negativeLabel := fs.String("negative-label", "", "optional negative label override for binary bundles")
	runner := fs.String("python-runner", defaultRunner, "Python dependency runner: uv or python")
	pythonExec := fs.String("python-exec", "python3", "python executable to use when -python-runner=python")
	if err := fs.Parse(args); err != nil {
		return err
	}

	if strings.TrimSpace(*model) == "" {
		return errors.New("export: -model is required\nnext step: infergo-export export -model distilbert/distilbert-base-uncased-finetuned-sst-2-english -input ./family1-inputs.json -out ./artifacts/my-bundle")
	}
	if strings.TrimSpace(*inputPathFlag) == "" {
		return errors.New("export: -input is required\nnext step: start with infergo-export template -kind single -out ./family1-inputs.json")
	}
	if strings.TrimSpace(*outputDirFlag) == "" {
		return errors.New("export: -out is required\nnext step: choose an output directory such as ./artifacts/my-bundle")
	}

	inputPath := filepath.Clean(*inputPathFlag)
	if _, err := parity.LoadTransformersTextClassificationInputSet(inputPath); err != nil {
		return fmt.Errorf("export: validate input set: %w\nnext step: confirm the file exists and matches the template written by infergo-export template", err)
	}

	modelID, err := resolveModelID(*model, *modelIDOverride)
	if err != nil {
		return fmt.Errorf("export: %w", err)
	}

	withPairs, err := inputSetSupportsPairs(inputPath)
	if err != nil {
		return fmt.Errorf("export: inspect input set: %w", err)
	}

	outputDir := filepath.Clean(*outputDirFlag)
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		return fmt.Errorf("export: create output dir: %w", err)
	}

	tempDir, err := os.MkdirTemp("", "infergo-export-")
	if err != nil {
		return fmt.Errorf("export: create temp dir: %w", err)
	}
	defer os.RemoveAll(tempDir)

	sourceReferencePath := filepath.Join(tempDir, "source-reference.json")
	tokenizerDir := filepath.Join(tempDir, "tokenizer")
	legacyBundleDir := filepath.Join(tempDir, "legacy-bundle")
	helperScriptPath := filepath.Join(tempDir, "family1_reference_and_tokenizer.py")

	if err := materializeEmbeddedAsset("assets/family1_reference_and_tokenizer.py", helperScriptPath); err != nil {
		return fmt.Errorf("export: write embedded helper: %w", err)
	}

	if err := runPythonHelper(*runner, *pythonExec, helperScriptPath, *model, inputPath, sourceReferencePath, tokenizerDir, *maxLength); err != nil {
		return fmt.Errorf("export: generate source reference and tokenizer assets: %w", err)
	}

	reference, err := parity.LoadTransformersTextClassificationReference(sourceReferencePath)
	if err != nil {
		return fmt.Errorf("export: load generated source reference: %w", err)
	}

	if err := nativebundlegen.GenerateBundle(reference, legacyBundleDir, *featureMode, false); err != nil {
		return fmt.Errorf("export: fit native bundle: %w", err)
	}

	legacyMetadata, err := bionet.LoadTextClassificationBundleMetadata(legacyBundleDir)
	if err != nil {
		return fmt.Errorf("export: load generated bundle metadata: %w", err)
	}

	manifest, err := loadTokenizerManifest(filepath.Join(tokenizerDir, "manifest.json"))
	if err != nil {
		return fmt.Errorf("export: load tokenizer manifest: %w", err)
	}

	positive, negative, err := inferLabelOverrides(reference.Labels, *positiveLabel, *negativeLabel)
	if err != nil {
		return fmt.Errorf("export: %w", err)
	}

	if err := copyFile(filepath.Join(legacyBundleDir, nativebundlegen.DefaultArtifactName), filepath.Join(outputDir, nativebundlegen.DefaultArtifactName)); err != nil {
		return fmt.Errorf("export: copy model artifact: %w", err)
	}
	embeddingSrc := filepath.Join(legacyBundleDir, nativebundlegen.DefaultEmbeddingName)
	embeddingDst := filepath.Join(outputDir, nativebundlegen.DefaultEmbeddingName)
	if _, embeddingErr := os.Stat(embeddingSrc); embeddingErr == nil {
		if err := copyFile(embeddingSrc, embeddingDst); err != nil {
			return fmt.Errorf("export: copy embedding artifact: %w", err)
		}
	} else if errors.Is(embeddingErr, os.ErrNotExist) {
		_ = os.Remove(embeddingDst)
	} else {
		return fmt.Errorf("export: inspect embedding artifact: %w", embeddingErr)
	}

	bundleManifest, bundleTokenizerSupported, tokenizerNote := manifestForAlphaBundle(manifest)

	if err := os.RemoveAll(filepath.Join(outputDir, "tokenizer")); err != nil && !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("export: reset tokenizer dir: %w", err)
	}
	if bundleTokenizerSupported {
		if err := copyDir(tokenizerDir, filepath.Join(outputDir, "tokenizer")); err != nil {
			return fmt.Errorf("export: copy tokenizer assets: %w", err)
		}
	}

	if err := writeJSON(filepath.Join(outputDir, "labels.json"), map[string]any{"labels": reference.Labels}); err != nil {
		return fmt.Errorf("export: write labels.json: %w", err)
	}

	metadata := buildAlphaMetadata(modelID, *model, *bundleVersion, *maxLength, legacyMetadata, bundleManifest, positive, negative)
	if err := writeJSON(filepath.Join(outputDir, "metadata.json"), metadata); err != nil {
		return fmt.Errorf("export: write metadata.json: %w", err)
	}

	if strings.TrimSpace(*referenceOutputFlag) != "" {
		referenceOutput := filepath.Clean(*referenceOutputFlag)
		if err := copyFile(sourceReferencePath, referenceOutput); err != nil {
			return fmt.Errorf("export: copy reference output: %w", err)
		}
	}

	printExportSummary(outputDir, manifest, bundleManifest, bundleTokenizerSupported, withPairs, *referenceOutputFlag, tokenizerNote)
	return nil
}

func materializeEmbeddedAsset(assetPath, outputPath string) error {
	raw, err := embeddedAssets.ReadFile(assetPath)
	if err != nil {
		return err
	}
	return os.WriteFile(outputPath, raw, 0o755)
}

func runPythonHelper(runner, pythonExec, helperScriptPath, model, inputPath, referenceOutputPath, tokenizerDir string, maxLength int) error {
	var cmd *exec.Cmd
	switch runner {
	case "uv":
		if _, err := exec.LookPath("uv"); err != nil {
			return errors.New("uv was not found in PATH\nnext step: install uv, or rerun with -python-runner=python and a Python environment that already has torch==2.10.0 and transformers==5.3.0")
		}
		cmd = exec.Command(
			"uv",
			"run",
			"--with", "torch==2.10.0",
			"--with", "transformers==5.3.0",
			"python",
			helperScriptPath,
			"--model-id", model,
			"--input", inputPath,
			"--reference-output", referenceOutputPath,
			"--tokenizer-dir", tokenizerDir,
			"--max-length", fmt.Sprintf("%d", maxLength),
		)
	case "python":
		if _, err := exec.LookPath(pythonExec); err != nil {
			return fmt.Errorf("python executable %q was not found in PATH\nnext step: install Python, change -python-exec, or rerun with the default -python-runner=uv", pythonExec)
		}
		cmd = exec.Command(
			pythonExec,
			helperScriptPath,
			"--model-id", model,
			"--input", inputPath,
			"--reference-output", referenceOutputPath,
			"--tokenizer-dir", tokenizerDir,
			"--max-length", fmt.Sprintf("%d", maxLength),
		)
	default:
		return fmt.Errorf("unsupported python runner %q (expected \"uv\" or \"python\")\nnext step: use -python-runner=uv or -python-runner=python", runner)
	}

	var stdout bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = io.MultiWriter(os.Stdout, &stdout)
	cmd.Stderr = io.MultiWriter(os.Stderr, &stderr)
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("python export step failed: %w\nnext step: confirm the model is a supported Transformers sequence-classification model, local model directories include config/tokenizer/weights files, and export-time Python dependencies are available", err)
	}
	return nil
}

func loadTokenizerManifest(path string) (tokenizerManifest, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return tokenizerManifest{}, err
	}
	var manifest tokenizerManifest
	if err := json.Unmarshal(raw, &manifest); err != nil {
		return tokenizerManifest{}, err
	}
	if manifest.Kind == "" {
		return tokenizerManifest{}, errors.New("tokenizer manifest: missing kind")
	}
	if manifest.Kind == alphaTokenizerRuntimeKind {
		if manifest.Files["tokenizer_json"] == "" {
			return tokenizerManifest{}, errors.New("tokenizer manifest: hf-tokenizer-json requires files.tokenizer_json")
		}
	}
	return manifest, nil
}

func manifestForAlphaBundle(manifest tokenizerManifest) (tokenizerManifest, bool, string) {
	if manifest.Kind != alphaTokenizerRuntimeKind {
		return tokenizerManifest{}, false, fmt.Sprintf(
			"note: detected tokenizer kind %q is outside the current alpha raw-text boundary; this bundle will support tokenized input only (validated raw-text subsets are BERT-style WordPiece and RoBERTa-style ByteLevel BPE tokenizer.json bundles)",
			manifest.Kind,
		)
	}
	if !manifest.RawTextSupported {
		return tokenizerManifest{}, false, "note: staged tokenizer assets did not match the current alpha raw-text boundary; this bundle will support tokenized input only (validated raw-text subsets are BERT-style WordPiece and RoBERTa-style ByteLevel BPE tokenizer.json bundles)"
	}
	return manifest, true, ""
}

func resolveModelID(modelArg, modelIDOverride string) (string, error) {
	if strings.TrimSpace(modelIDOverride) != "" {
		return modelIDOverride, nil
	}
	if pathExists(modelArg) {
		return "", errors.New("-model-id is required when -model points to a local path so the exported bundle records a stable canonical model id\nnext step: rerun with -model-id myorg/my-model")
	}
	return modelArg, nil
}

func inputSetSupportsPairs(path string) (bool, error) {
	inputSet, err := parity.LoadTransformersTextClassificationInputSet(path)
	if err != nil {
		return false, err
	}
	for _, item := range inputSet.Cases {
		if strings.TrimSpace(item.TextPair) != "" {
			return true, nil
		}
	}
	return false, nil
}

func inferLabelOverrides(labels []string, positive, negative string) (string, string, error) {
	if positive != "" && !containsLabel(labels, positive) {
		return "", "", fmt.Errorf("--positive-label %q is not present in labels: %v", positive, labels)
	}
	if negative != "" && !containsLabel(labels, negative) {
		return "", "", fmt.Errorf("--negative-label %q is not present in labels: %v", negative, labels)
	}
	if positive != "" || negative != "" {
		return positive, negative, nil
	}

	normalized := make(map[string]string, len(labels))
	for _, label := range labels {
		normalized[strings.ToLower(label)] = label
	}

	return inferLabelFromSet(normalized, commonPositiveLabels), inferLabelFromSet(normalized, commonNegativeLabels), nil
}

func containsLabel(labels []string, target string) bool {
	for _, label := range labels {
		if label == target {
			return true
		}
	}
	return false
}

func inferLabelFromSet(normalized map[string]string, candidates []string) string {
	for _, candidate := range candidates {
		if value, ok := normalized[candidate]; ok {
			return value
		}
	}
	return ""
}

func buildAlphaMetadata(modelID, modelArg, bundleVersion string, maxLength int, legacy bionet.TextClassificationBundleMetadata, manifest tokenizerManifest, positive, negative string) alphaMetadata {
	source := alphaSource{
		Framework: "pytorch",
		Ecosystem: "transformers",
	}
	if repoURL := detectRepoURL(modelArg); repoURL != "" {
		source.RepoURL = repoURL
	}

	outputs := alphaOutputs{
		Kind:           "label_logits",
		LabelsArtifact: "labels.json",
		PositiveLabel:  positive,
		NegativeLabel:  negative,
	}
	if len(legacy.Labels) == 2 && positive != "" && negative != "" {
		threshold := 0.5
		outputs.Threshold = &threshold
	}

	metadata := alphaMetadata{
		BundleFormat:    "infergo-native",
		BundleVersion:   bundleVersion,
		Family:          "encoder-text-classification",
		Task:            "text-classification",
		Backend:         "bionet",
		BackendArtifact: legacy.Artifact,
		ModelID:         modelID,
		Source:          source,
		Inputs: alphaInputs{
			RawTextSupported:        manifest.RawTextSupported,
			PairTextSupported:       manifest.PairTextSupported,
			TokenizedInputSupported: true,
			MaxSequenceLength:       maxLength,
		},
		Outputs: outputs,
		BackendConfig: alphaBackendConfig{
			FeatureMode:       legacy.FeatureMode,
			FeatureTokenIDs:   append([]int(nil), legacy.FeatureTokenIDs...),
			EmbeddingArtifact: legacy.EmbeddingArtifact,
		},
		CreatedAt: time.Now().UTC().Format(time.RFC3339),
		CreatedBy: alphaCreatedBy{
			Tool:    "infergo-export",
			Version: exportToolVersion,
		},
	}
	if manifest.RawTextSupported {
		metadata.Tokenizer = alphaTokenizer{
			Manifest: "tokenizer/manifest.json",
		}
	}
	return metadata
}

func detectRepoURL(modelArg string) string {
	if pathExists(modelArg) {
		return ""
	}
	if !strings.Contains(modelArg, "/") {
		return ""
	}
	return "https://huggingface.co/" + modelArg
}

func writeJSON(path string, payload any) error {
	raw, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		return err
	}
	raw = append(raw, '\n')
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, raw, 0o644)
}

func copyFile(src, dst string) error {
	input, err := os.ReadFile(src)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}
	return os.WriteFile(dst, input, 0o644)
}

func copyDir(src, dst string) error {
	return filepath.Walk(src, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		relative, err := filepath.Rel(src, path)
		if err != nil {
			return err
		}
		target := filepath.Join(dst, relative)
		if info.IsDir() {
			return os.MkdirAll(target, info.Mode())
		}
		return copyFile(path, target)
	})
}

func pathExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

func printExportSummary(outputDir string, detectedManifest, bundleManifest tokenizerManifest, bundleTokenizerSupported, withPairs bool, referenceOutput, tokenizerNote string) {
	supportedInputs := []string{"input_ids"}
	if bundleManifest.RawTextSupported {
		supportedInputs = append([]string{"text"}, supportedInputs...)
	}
	if bundleManifest.PairTextSupported {
		supportedInputs = append(supportedInputs, "text+text_pair")
	}

	fmt.Printf("wrote family-1 alpha bundle to %s\n", outputDir)
	fmt.Printf("detected tokenizer kind: %s\n", detectedManifest.Kind)
	fmt.Printf("bundle embeds tokenizer metadata: %t\n", bundleTokenizerSupported)
	fmt.Printf("supports raw text: %t\n", bundleManifest.RawTextSupported)
	fmt.Printf("supports pair text: %t\n", bundleManifest.PairTextSupported)
	fmt.Printf("supported inputs: %s\n", strings.Join(supportedInputs, ", "))
	if strings.TrimSpace(referenceOutput) != "" {
		fmt.Printf("saved source reference to %s\n", filepath.Clean(referenceOutput))
	}
	if tokenizerNote != "" {
		fmt.Println(tokenizerNote)
	}
	if withPairs && !bundleManifest.PairTextSupported {
		fmt.Println("note: the input set uses text pairs, but the exported tokenizer assets only support tokenized paired input in the current runtime boundary")
	}
}
