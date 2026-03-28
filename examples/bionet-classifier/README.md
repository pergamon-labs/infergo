# BIOnet Classifier Example

This example demonstrates the first supported BIOnet-backed inference path using
checked-in public-safe assets.

Run it from the repo root:

```bash
go run ./examples/bionet-classifier
```

By default it loads the checked-in DistilBERT SST-2 native bundle and runs the
`positive-review` reference case.

You can point it at another pack or case:

```bash
go run ./examples/bionet-classifier \
  -bundle ./testdata/native/text-classification/twitter-roberta-sentiment-embedding-masked-avg-pool \
  -reference ./testdata/reference/text-classification/twitter-roberta-sentiment-reference.json \
  -case-id support-praise
```
