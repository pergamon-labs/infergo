# Text Classification Example

This example demonstrates the curated `infer/packs` API for checked-in native
text-classification packs.

Run it from the repo root:

```bash
go run ./examples/bionet-classifier
```

By default it loads the checked-in `infergo-basic-sst2` pack and runs the
`positive-review` reference case through the pack helper.

It also supports the repo's first honest raw-text flow out of the box:

```bash
go run ./examples/bionet-classifier \
  -text "This product is excellent and reliable."
```

You can point it at another pack or case:

```bash
go run ./examples/bionet-classifier \
  -pack twitter-roberta-sentiment \
  -reference ./testdata/reference/text-classification/twitter-roberta-sentiment-reference.json \
  -case-id support-praise
```
