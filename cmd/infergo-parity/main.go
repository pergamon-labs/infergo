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
	tolerance := flag.Float64("tolerance", 1e-4, "tolerance used for reference/candidate comparisons")
	flag.Parse()

	if *referencePath != "" || *candidatePath != "" {
		if *referencePath == "" || *candidatePath == "" {
			fmt.Fprintln(os.Stderr, "infergo-parity: both -reference and -candidate must be provided together")
			os.Exit(1)
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
