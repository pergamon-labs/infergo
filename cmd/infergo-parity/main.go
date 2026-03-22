package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/minervaai/infergo/internal/parity"
)

func main() {
	fixturePath := flag.String("fixture", "testdata/parity/text-classification/fixture.json", "path to a parity fixture JSON file")
	flag.Parse()

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
