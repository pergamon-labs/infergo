# AGENTS.md

## Scope

Instructions in this file apply to `/Users/jatto/Documents/workspaces/pergamon-labs/infergo`.

## Inheritance

This file extends:

- `/Users/jatto/Documents/workspaces/AGENTS.md`
- `/Users/jatto/Documents/workspaces/pergamon-labs/AGENTS.md`

If there is a conflict, this file takes precedence inside this repo.

## Purpose

InferGo is an open-source, Go-native inference and model-serving toolkit for backend services.

Current repo reality:

- `backends/bionet/` is the first native backend path and the main implementation focus.
- `backends/torchscript/` is an optional compatibility bridge and must not become the default runtime dependency of the project.
- parity-driven validation is part of the product, not just test scaffolding.
- all checked-in fixtures, bundles, and reference assets must be public-safe and reproducible.

## Local Overlay

If present, also apply:

- `/Users/jatto/Documents/workspaces/pergamon-labs/infergo/AGENTS.local.md`

Use `AGENTS.local.md` for personal notes, machine-specific paths, temporary local workflow instructions, or private preferences that should not be committed.

Rules for the local overlay:

- keep secrets out of repo files even if they are gitignored
- prefer environment variables or external secret stores for credentials
- do not rely on local-only instructions for anything required to build or validate the public repo

## Product Guardrails

- Keep the north star: Go-native inference for backend services.
- Do not market or document InferGo as a blanket `.pt` loader, training framework, or general transformer runtime unless the repo actually supports that path end to end.
- Keep CPU-first and pure-Go-by-default as the baseline posture.
- Treat libtorch and TorchScript as optional backend-specific dependencies, not project-wide requirements.
- Every new support claim should come with:
  - explicit artifact/backend scope
  - updated `README.md`
  - updated `COMPATIBILITY.md`
  - parity tests or golden tests

## Public Safety Rules

- Do not commit proprietary models, private datasets, customer data, or Minerva-internal artifacts.
- Keep examples and fixtures public-safe, small, and reproducible.
- If a file would be awkward to publish publicly, it probably should not be in this repo.

## Repo Map

- `infer/`: stable public-facing API surface
- `backends/bionet/`: native backend path
- `backends/bionet/runtime/`: extracted runtime primitives
- `backends/torchscript/`: optional compatibility backend
- `internal/parity/`: reference loading and parity comparison logic
- `internal/tools/`: reproducible generators for fixtures and bundles
- `cmd/infergo-parity/`: canonical CLI for parity checks
- `testdata/reference/`: public reference outputs
- `testdata/native/`: checked-in native bundles that are safe to publish

## Working Rules

- Run commands from the repo root.
- Prefer `rg` for search and `go test ./...` for broad validation.
- When behavior, support, or architecture changes, update `README.md`, `COMPATIBILITY.md`, and `ARCHITECTURE.md` in the same change.
- Keep generated scratch outputs ignored; checked-in outputs should be reproducible from committed tools and reference inputs.
- Use Conventional Commits for commit messages.
- Avoid destructive git commands unless explicitly asked.

## GitHub Home

- Canonical module path: `github.com/pergamon-labs/infergo`
- Canonical remote org: `https://github.com/pergamon-labs/`
