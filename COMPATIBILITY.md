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
- Current validated examples:
  - synthetic text classification on dense feature vectors via `cmd/infergo-parity`
  - native text classification over public DistilBERT reference inputs via `cmd/infergo-parity -infergo-bundle-dir ...`
  - native text classification over public RoBERTa sentiment reference inputs via `cmd/infergo-parity -infergo-bundle-dir ...`
  - native token classification over a public `dslim/distilbert-NER` reference set via `cmd/infergo-parity -infergo-bundle-dir ...`
  - checked-in native bundle shapes:
    - `token-id-bag`
    - `embedding-avg-pool -> linear` with compact dense token embeddings
    - `embedding-masked-avg-pool -> linear` with compact dense token embeddings
    - `token embedding -> linear` for narrow token classification parity

## v1 stretch path

- Artifact type: exported TorchScript artifacts through a documented export flow
- Backend: `torchscript`
- Runtime posture: CPU-first initially
- Support bar: parity-tested on fixed public inputs against a Python reference implementation
- Current external reference paths:
  - `distilbert/distilbert-base-uncased-finetuned-sst-2-english`
  - `cardiffnlp/twitter-roberta-base-sentiment-latest`
  - `dslim/distilbert-NER` for native token-classification parity through the `bionet` backend
- Current local comparison path: TorchScript export plus native Go candidate generation through `cmd/infergo-parity -torchscript-bundle-dir ...`

## Native backend note

- The native `torchscript` backend currently requires `CGO_ENABLED=1`, the `torchscript_native` build tag, and a libtorch install exposed through `CGO_CXXFLAGS` and `CGO_LDFLAGS`.
- On machines without that setup, InferGo builds cleanly but returns a descriptive runtime error if the native backend path is invoked.

## Not supported in v1

- arbitrary `.pt` files without a documented export path
- direct Hugging Face repository loading
- general transformer execution in the native `bionet` backend
- contextual token classification beyond the explicitly documented NER smoke path
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
