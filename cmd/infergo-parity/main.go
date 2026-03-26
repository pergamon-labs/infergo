package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/pergamon-labs/infergo/internal/parity"
)

func main() {
	fixturePath := flag.String("fixture", "testdata/parity/text-classification/fixture.json", "path to a parity fixture JSON file")
	referencePath := flag.String("reference", "", "path to a Transformers reference JSON file")
	candidatePath := flag.String("candidate", "", "path to a local candidate JSON file")
	torchscriptBundleDir := flag.String("torchscript-bundle-dir", "", "path to a TorchScript bundle directory to run natively from Go")
	infergoBundleDir := flag.String("infergo-bundle-dir", "", "path to an InferGo-native bundle directory to run natively from Go")
	candidateOutputPath := flag.String("candidate-output", "", "optional path to write the generated candidate JSON when using -torchscript-bundle-dir")
	tolerance := flag.Float64("tolerance", 1e-4, "tolerance used for reference/candidate comparisons")
	flag.Parse()

	if *referencePath != "" || *candidatePath != "" || *torchscriptBundleDir != "" || *infergoBundleDir != "" {
		if *referencePath == "" {
			fmt.Fprintln(os.Stderr, "infergo-parity: -reference is required when using external reference comparison flags")
			os.Exit(1)
		}

		backendSources := 0
		if *candidatePath != "" {
			backendSources++
		}
		if *torchscriptBundleDir != "" {
			backendSources++
		}
		if *infergoBundleDir != "" {
			backendSources++
		}

		if backendSources == 0 {
			fmt.Fprintln(os.Stderr, "infergo-parity: provide one of -candidate, -torchscript-bundle-dir, or -infergo-bundle-dir")
			os.Exit(1)
		}

		if backendSources > 1 {
			fmt.Fprintln(os.Stderr, "infergo-parity: use only one of -candidate, -torchscript-bundle-dir, or -infergo-bundle-dir")
			os.Exit(1)
		}

		task, err := parity.DetectTransformersReferenceTask(*referencePath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "infergo-parity: %v\n", err)
			os.Exit(1)
		}

		switch task {
		case "text-classification":
			if *torchscriptBundleDir != "" {
				candidate, err := parity.RunTorchScriptTextClassificationBundle(*referencePath, *torchscriptBundleDir)
				if err != nil {
					fmt.Fprintf(os.Stderr, "infergo-parity: %v\n", err)
					os.Exit(1)
				}

				*candidatePath = persistTextClassificationCandidate(*candidateOutputPath, candidate)
			}

			if *infergoBundleDir != "" {
				candidate, err := parity.RunBionetTextClassificationBundle(*referencePath, *infergoBundleDir)
				if err != nil {
					fmt.Fprintf(os.Stderr, "infergo-parity: %v\n", err)
					os.Exit(1)
				}

				*candidatePath = persistTextClassificationCandidate(*candidateOutputPath, candidate)
			}

			report, err := parity.CompareTransformersTextClassification(*referencePath, *candidatePath, *tolerance)
			if err != nil {
				fmt.Fprintf(os.Stderr, "infergo-parity: %v\n", err)
				os.Exit(1)
			}

			fmt.Print(report.String())
			if !report.Passed() {
				os.Exit(1)
			}
		case "token-classification":
			if *torchscriptBundleDir != "" {
				fmt.Fprintln(os.Stderr, "infergo-parity: TorchScript bundle execution is not implemented for token-classification yet")
				os.Exit(1)
			}

			if *infergoBundleDir != "" {
				candidate, err := parity.RunBionetTokenClassificationBundle(*referencePath, *infergoBundleDir)
				if err != nil {
					fmt.Fprintf(os.Stderr, "infergo-parity: %v\n", err)
					os.Exit(1)
				}

				*candidatePath = persistTokenClassificationCandidate(*candidateOutputPath, candidate)
			}

			report, err := parity.CompareTransformersTokenClassification(*referencePath, *candidatePath, *tolerance)
			if err != nil {
				fmt.Fprintf(os.Stderr, "infergo-parity: %v\n", err)
				os.Exit(1)
			}

			fmt.Print(report.String())
			if !report.Passed() {
				os.Exit(1)
			}
		default:
			fmt.Fprintf(os.Stderr, "infergo-parity: unsupported reference task %q\n", task)
			os.Exit(1)
		}
		return
	}

	report, err := parity.RunTextClassificationFixture(*fixturePath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "infergo-parity: %v\n", err)
		os.Exit(1)
	}

	fmt.Print(report.String())

	if !report.Passed() {
		os.Exit(1)
	}
}

func persistTextClassificationCandidate(outputPath string, candidate parity.TextClassificationCandidate) string {
	if outputPath != "" {
		if err := parity.SaveTextClassificationCandidate(candidate, outputPath); err != nil {
			fmt.Fprintf(os.Stderr, "infergo-parity: %v\n", err)
			os.Exit(1)
		}
		return outputPath
	}

	tempFile, err := os.CreateTemp("", "infergo-text-candidate-*.json")
	if err != nil {
		fmt.Fprintf(os.Stderr, "infergo-parity: create temp candidate file: %v\n", err)
		os.Exit(1)
	}
	tempPath := tempFile.Name()
	tempFile.Close()

	if err := parity.SaveTextClassificationCandidate(candidate, tempPath); err != nil {
		fmt.Fprintf(os.Stderr, "infergo-parity: %v\n", err)
		os.Exit(1)
	}
	return tempPath
}

func persistTokenClassificationCandidate(outputPath string, candidate parity.TokenClassificationCandidate) string {
	if outputPath != "" {
		if err := parity.SaveTokenClassificationCandidate(candidate, outputPath); err != nil {
			fmt.Fprintf(os.Stderr, "infergo-parity: %v\n", err)
			os.Exit(1)
		}
		return outputPath
	}

	tempFile, err := os.CreateTemp("", "infergo-token-candidate-*.json")
	if err != nil {
		fmt.Fprintf(os.Stderr, "infergo-parity: create temp candidate file: %v\n", err)
		os.Exit(1)
	}
	tempPath := tempFile.Name()
	tempFile.Close()

	if err := parity.SaveTokenClassificationCandidate(candidate, tempPath); err != nil {
		fmt.Fprintf(os.Stderr, "infergo-parity: %v\n", err)
		os.Exit(1)
	}
	return tempPath
}
