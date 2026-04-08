package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/pergamon-labs/infergo/backends/bionet"
	"github.com/pergamon-labs/infergo/internal/nativebundlegen"
)

const (
	defaultReferencePath = "testdata/reference/text-classification/distilbert-sst2-reference.json"
	defaultOutputDir     = "testdata/native/text-classification/distilbert-sst2-embedding-masked-avg-pool"
)

func main() {
	referencePath := flag.String("reference", defaultReferencePath, "path to the Transformers reference JSON file")
	outputDir := flag.String("output-dir", defaultOutputDir, "directory to write the InferGo-native bundle into")
	mode := flag.String("mode", bionet.TextClassificationFeatureModeEmbeddingMaskedAvgPool, "native bundle mode: token-id-bag, embedding-avg-pool, or embedding-masked-avg-pool")
	useLayerNorm := flag.Bool("use-layernorm", false, "experimentally add a layer-normalization stage before the masked-pooling classifier head")
	flag.Parse()

	if err := nativebundlegen.GenerateBundleFromReferencePath(*referencePath, *outputDir, *mode, *useLayerNorm); err != nil {
		fatalf("%v", err)
	}

	fmt.Printf("wrote infergo-native bundle to %s\n", *outputDir)
}

func fatalf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "infergo-native-bundlegen: "+format+"\n", args...)
	os.Exit(1)
}
