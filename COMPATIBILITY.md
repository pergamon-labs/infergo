# Compatibility

InferGo supports only the explicitly tested artifact types, backends, tasks, and examples documented here and in the README.

InferGo does not make blanket claims such as:

- "supports `.pt` files"
- "supports Hugging Face models"
- "supports PyTorch models"

without a documented export path, backend, and parity test story.

## v1 launch path

- Artifact type: InferGo-native BIOnet bundles backed by `.gob` runtime artifacts plus bundle metadata
- Backend: `bionet`
- Runtime posture: CPU-first
- Primary task shape: small classification-style inference in Go services
- Public APIs:
  - stable bundle API: `infer.LoadTextClassifier` and `infer.LoadTokenClassifier`
  - curated pack API: `infer/packs.LoadTextPack` and `infer/packs.LoadTokenPack`
  - stable HTTP handler API: `infer/httpserver.NewTextPackMux` and `infer/httpserver.NewTokenPackMux`
  - stable server config API: `infer/httpserver.DefaultServerConfig` and `infer/httpserver.NewServer`
  - alpha text-bundle loader validation for:
    - versioned `metadata.json`
    - external `labels.json`
    - tokenizer manifest checks for raw-text-capable bundles
  - Current validated examples:
  - curated pack discovery through `cmd/infergo-packs`
  - stable REST serving through `cmd/infergo-serve`
  - experimental family-1 export through `scripts/export_encoder_text_bundle.py`
    for projection-based single-text and paired-text classification bundles
  - experimental serving of exported family-1 text bundles through
    `cmd/infergo-serve -task text -bundle ...` using tokenized input
  - structured JSON error responses from the stable HTTP surface
  - graceful shutdown and timeout-driven HTTP serving through `cmd/infergo-serve`
  - benchmark suite for current checked-in raw-text text/token paths and the stable HTTP handler surface through `go test ./infer/packs ./infer/httpserver -run '^$' -bench . -benchmem`
  - synthetic text classification on dense feature vectors via `cmd/infergo-parity`
  - native text classification over the manifest-backed public model packs listed in `testdata/reference/text-classification/model-packs.json` via `cmd/infergo-parity -infergo-bundle-dir ...`
  - native token classification over the manifest-backed public model packs listed in `testdata/reference/token-classification/model-packs.json` via `cmd/infergo-parity -infergo-bundle-dir ...`
  - curated text-pack prediction via `examples/bionet-classifier`
  - first truly native raw-text text prediction via the checked-in `infergo-basic-sst2` pack
  - curated token-pack prediction via `examples/token-http-server`
  - first truly native raw-text token prediction via the checked-in `infergo-basic-french-ner` pack
  - text classification served through `infer/httpserver` and `cmd/infergo-serve -task text`
  - token classification served through `infer/httpserver` and `cmd/infergo-serve -task token`
  - the token-classification manifest now includes a first non-English multilingual NER pack through `Davlan/xlm-roberta-base-ner-hrl`
  - the token-classification manifest now also includes a French-specific NER pack through `cmarkea/distilcamembert-base-ner`
  - checked-in native bundle shapes:
    - `token-id-bag`
    - `embedding-avg-pool -> linear` with compact dense token embeddings
    - `embedding-masked-avg-pool -> linear` with compact dense token embeddings
    - `windowed token embedding -> linear` for narrow token classification parity
  - curated pack helpers:
    - checked-in text packs can be loaded and queried by pack key
    - checked-in token packs can be loaded and queried by pack key
    - piece-aware prediction helpers are supported for checked-in packs whose tokenizer-piece to id mapping is validated from the public-safe reference data
    - raw-text prediction is only supported when a pack explicitly validates a checked-in tokenizer helper
  - current raw-text-capable packs: `infergo-basic-sst2`, `infergo-basic-french-ner`
  - first alpha-format exported family-1 bundles can now load through
    `infer.LoadTextClassifier` and compare through `cmd/infergo-parity`
    without requiring Python at runtime

## v1 stretch path

- Artifact type: exported TorchScript artifacts through a documented export flow
- Backend: `torchscript`
- Runtime posture: CPU-first initially
- Support bar: parity-tested on fixed public inputs against a Python reference implementation
- Current external reference paths:
  - the checked-in text-classification model packs listed in `testdata/reference/text-classification/model-packs.json`
  - the checked-in token-classification model packs listed in `testdata/reference/token-classification/model-packs.json` for native parity through the `bionet` backend
- Current local comparison path: TorchScript export plus native Go candidate generation through `cmd/infergo-parity -torchscript-bundle-dir ...`

## Internal dogfood bridge

- Artifact type: `infergo-torchscript-bridge` bundle for numeric-feature
  scoring
- Backend: `torchscript`
- Support posture: experimental and internal-first
- Current experimental surfaces:
  - bundle loading through `backends/torchscript.LoadEntityResolutionBundle`
  - Go API through `infer/experimental/entres`
  - HTTP serving through `cmd/infergo-entres-serve`
  - local parity through `cmd/infergo-entres-parity`
- Intended use: current `entres` entity-resolution model family
- Current local validation bar:
  - individual and organization `core-svc` ER models pass local parity against
    fixtures captured from the current screening runtime
  - bridge metadata explicitly captures the legacy
    `message_projection=legacy_first_value_broadcast` behavior needed to match
    that runtime
- Non-claim: this does not imply broad TorchScript or arbitrary `.pt` support

## Native backend note

- The native `torchscript` backend currently requires `CGO_ENABLED=1`, the `torchscript_native` build tag, and a libtorch install exposed through `CGO_CXXFLAGS` and `CGO_LDFLAGS`.
- On machines without that setup, InferGo builds cleanly but returns a descriptive runtime error if the native backend path is invoked.

## Not supported in v1

- arbitrary `.pt` files without a documented export path
- arbitrary numeric-feature TorchScript scorers beyond the documented family-2 bridge
- direct Hugging Face repository loading
- general transformer execution in the native `bionet` backend
- token classification beyond the explicitly documented local-window NER path
- blanket raw-text tokenization support for checked-in packs that do not validate a native tokenizer helper
- experimental `-use-layernorm` native bundle generation as a public support claim until it meets the same parity bar as the default path
- native attention blocks or full encoder stacks
- ONNX runtime support
- training, autograd, or optimizer APIs
- generative text serving
- multimodal or vision model support

## Backend support rule

A backend should only be called supported when:

1. the artifact/export path is documented
2. at least one public example works end-to-end
3. supported task/model shapes are explicit
4. known unsupported features are documented
5. parity or golden tests exist
6. a normal Go server example exists
7. the stable HTTP surface is documented if REST serving is claimed
