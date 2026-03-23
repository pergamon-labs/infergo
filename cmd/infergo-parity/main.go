package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/minervaai/infergo/internal/parity"
)

func main() {
	fixturePath := flag.String("fixture", "testdata/parity/text-classification/fixture.json", "path to a parity fixture JSON file")
	referencePath := flag.String("reference", "", "path to a Transformers reference JSON file")
	candidatePath := flag.String("candidate", "", "path to a local candidate JSON file")
	torchscriptBundleDir := flag.String("torchscript-bundle-dir", "", "path to a TorchScript bundle directory to run natively from Go")
	candidateOutputPath := flag.String("candidate-output", "", "optional path to write the generated candidate JSON when using -torchscript-bundle-dir")
	tolerance := flag.Float64("tolerance", 1e-4, "tolerance used for reference/candidate comparisons")
	flag.Parse()

	if *referencePath != "" || *candidatePath != "" || *torchscriptBundleDir != "" {
		if *referencePath == "" {
			fmt.Fprintln(os.Stderr, "infergo-parity: -reference is required when using external reference comparison flags")
			os.Exit(1)
		}

		if *candidatePath == "" && *torchscriptBundleDir == "" {
			fmt.Fprintln(os.Stderr, "infergo-parity: provide either -candidate or -torchscript-bundle-dir")
			os.Exit(1)
		}

		if *candidatePath != "" && *torchscriptBundleDir != "" {
			fmt.Fprintln(os.Stderr, "infergo-parity: use either -candidate or -torchscript-bundle-dir, not both")
			os.Exit(1)
		}

		if *torchscriptBundleDir != "" {
			candidate, err := parity.RunTorchScriptTextClassificationBundle(*referencePath, *torchscriptBundleDir)
			if err != nil {
				fmt.Fprintf(os.Stderr, "infergo-parity: %v\n", err)
				os.Exit(1)
			}

			if *candidateOutputPath != "" {
				if err := parity.SaveTorchScriptTextClassificationCandidate(candidate, *candidateOutputPath); err != nil {
					fmt.Fprintf(os.Stderr, "infergo-parity: %v\n", err)
					os.Exit(1)
				}
				*candidatePath = *candidateOutputPath
			} else {
				tempFile, err := os.CreateTemp("", "infergo-torchscript-candidate-*.json")
				if err != nil {
					fmt.Fprintf(os.Stderr, "infergo-parity: create temp candidate file: %v\n", err)
					os.Exit(1)
				}
				tempPath := tempFile.Name()
				tempFile.Close()
				defer os.Remove(tempPath)

				if err := parity.SaveTorchScriptTextClassificationCandidate(candidate, tempPath); err != nil {
					fmt.Fprintf(os.Stderr, "infergo-parity: %v\n", err)
					os.Exit(1)
				}
				*candidatePath = tempPath
			}
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
