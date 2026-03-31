# Contributing

This scaffold is intentionally small and extraction-oriented.

InferGo is currently preparing for a first public pre-alpha release. Please
keep contributions aligned with the narrow, explicitly documented support
surface.

Current contribution priorities:

1. keep the public API small and honest
2. avoid broad compatibility claims without parity coverage
3. prefer public-safe fixtures and examples
4. keep backend-specific code behind backend boundaries

## Community norms

By participating in this project, you agree to follow the guidance in
[CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md).

## Pull requests

Before opening a pull request:

- run `go test ./...`
- update `README.md` if the public workflow changed
- update `COMPATIBILITY.md` if support claims changed
- keep examples, bundles, and fixtures public-safe and reproducible

## Before adding support for a backend

- document the artifact/export path
- define the supported task/model shapes
- add a public example
- add parity or golden tests
- document known limitations

## Before promoting an internal runtime detail to public API

- verify that it is useful outside a single backend
- verify naming and docs are stable enough to support
- avoid exposing low-level packages just because they exist internally today
