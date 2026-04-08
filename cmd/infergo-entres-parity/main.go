package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/pergamon-labs/infergo/internal/parity"
)

func main() {
	fixturePath := flag.String("fixture", "", "path to a family-2 entity-resolution fixture JSON file")
	bundleDir := flag.String("bundle", "", "path to a family-2 entres bridge bundle directory")
	tolerance := flag.Float64("tolerance", 1e-6, "absolute score tolerance for parity comparisons")
	flag.Parse()

	if *fixturePath == "" {
		fmt.Fprintln(os.Stderr, "infergo-entres-parity: fixture path is required")
		os.Exit(1)
	}
	if *bundleDir == "" {
		fmt.Fprintln(os.Stderr, "infergo-entres-parity: bundle path is required")
		os.Exit(1)
	}

	report, err := parity.RunTorchScriptEntityResolutionFixture(*fixturePath, *bundleDir, *tolerance)
	if err != nil {
		fmt.Fprintf(os.Stderr, "infergo-entres-parity: %v\n", err)
		os.Exit(1)
	}

	fmt.Print(report.String())
	if !report.Passed() {
		os.Exit(1)
	}
}
