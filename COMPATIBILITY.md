# Compatibility

InferGo supports only the explicitly tested artifact types, backends, tasks, and examples documented here and in the README.

InferGo does not make blanket claims such as:

- "supports `.pt` files"
- "supports Hugging Face models"
- "supports PyTorch models"

without a documented export path, backend, and parity test story.

## v1 launch path

- Artifact type: BIOnet `.gob` artifacts with public-safe vocab/assets
- Backend: `bionet`
- Runtime posture: CPU-first
- Primary task shape: small classification-style inference in Go services

## v1 stretch path

- Artifact type: exported TorchScript artifacts through a documented export flow
- Backend: `torchscript`
- Runtime posture: CPU-first initially
- Support bar: parity-tested on fixed public inputs against a Python reference implementation

## Not supported in v1

- arbitrary `.pt` files without a documented export path
- direct Hugging Face repository loading
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
