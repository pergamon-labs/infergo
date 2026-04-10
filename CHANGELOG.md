# Changelog

All notable changes to InferGo will be documented in this file.

This project is still early, so entries are intentionally grouped at the
release level rather than by every internal commit.

## [Unreleased]

## [v0.2.0-alpha.3] - 2026-04-10

Third public alpha prerelease and the latest hardened alpha baseline.

### Added

- validated RoBERTa-style ByteLevel BPE tokenizer-json support for exported
  family-1 raw-text bundles
- stable byte and character offsets for curated raw-text NER flows
- maintainer benchmark snapshot and comparison helpers in `scripts/`

### Changed

- tightened family-1 tokenizer-boundary enforcement so unsupported raw-text
  tokenizers stay tokenized-input-only
- hardened `infergo-export` help and failure guidance for first-time users
- documented the benchmark reporting and regression workflow for maintainers
- pinned the current public alpha native bundle contract to
  `infergo-native` bundle version `1.0`
- made the exporter reject unsupported bundle versions up front instead of
  generating bundles the alpha loader will refuse
- clarified the alpha NER posture around offsets and sample-layer entity
  grouping

### Notes

- token classification remains a curated-pack/sample-service alpha story, not a
  public BYOM family
- family-2 internal validation remains private and separate from the public
  alpha contract
- the remaining TorchScript bridge is still optional and `libtorch`-dependent

## [v0.1.0-prealpha.1] - 2026-03-30

First public pre-alpha release candidate.

### Added

- stable public bundle-loading APIs in `infer` for text and token
  classification
- curated checked-in pack helpers in `infer/packs`
- pack discovery CLI in `cmd/infergo-packs`
- parity tooling in `cmd/infergo-parity`
- public-safe text-classification reference packs and native bundles
- public-safe token-classification reference packs and native bundles
- first raw-text-capable native text pack: `infergo-basic-sst2`
- first raw-text-capable native token pack: `infergo-basic-french-ner`
- tiny HTTP serving examples for text and token classification
- benchmark suite for pack load and prediction paths
- OSS repo scaffolding: CI, contributing guidance, code of conduct, security
  policy, issue templates, and PR template

### Changed

- narrowed public positioning around CPU-first, Go-native inference for backend
  services
- kept TorchScript/libtorch support explicitly optional and backend-specific
- tightened documentation so compatibility claims track actual parity-backed
  support

### Notes

- Recommended first public tag: `v0.1.0-prealpha.1`
- Supported surface is intentionally narrow and centered on checked-in,
  public-safe native bundles
- Checked-in packs are proof fixtures, not a general model zoo
