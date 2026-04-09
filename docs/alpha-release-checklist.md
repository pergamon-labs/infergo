# InferGo Alpha Release Checklist

Maintainer-facing note:

This checklist is for deciding when InferGo is ready for its first alpha
release. If you are evaluating InferGo as a user, start with
[`README.md`](../README.md) and
[`GETTING_STARTED.md`](./GETTING_STARTED.md).

## Alpha release bar

InferGo is ready for alpha when all of the following are true:

- the public family-1 BYOM path works end to end
- the support boundary is explicit and narrow
- the docs are usable without internal context
- the internal family-2 dogfood path has been validated beyond one operator
- the release notes and support posture are clear

## Family 1: public BYOM path

- [x] `infergo-export` installs cleanly without a repo checkout
- [x] a supported model exports into an InferGo-native bundle
- [x] an exported bundle loads in a small Go app
- [x] an exported bundle serves through `infergo-serve`
- [x] `infergo-parity` passes against the exported source reference
- [ ] exporter failure messages are clear enough for a stranger
- [ ] tokenizer boundary is documented clearly enough that users will not
      over-assume support

## Family 2: internal dogfood path

- [ ] the `entres` family-2 runbook has been executed by another engineer or
      service owner
- [ ] confusion points from that handoff have been addressed
- [ ] the internal bridge still remains clearly separate from the public
      family-1 contract

## Token classification alpha posture

- [x] token classification is still documented as curated-pack/sample-service
      support only
- [x] NER sample-service docs still match current behavior
- [ ] remaining NER limitations are explicit

## Documentation and release readiness

- [x] `README.md` still reflects the actual product surface
- [x] `GETTING_STARTED.md`, `USE_CASES.md`, and `PHILOSOPHY.md` are current
- [x] `COMPATIBILITY.md` and `ARCHITECTURE.md` match the current alpha scope
- [x] benchmark docs contain a current snapshot and clear reproduction steps
- [x] pkg.go.dev-facing examples exist for the main public Go API
- [x] release notes draft is ready
- [x] issue backlog reflects alpha hardening rather than broad future scope

## Release mechanics

- [ ] changelog entry is ready
- [ ] release notes file is finalized
- [ ] version tag is chosen
- [ ] final validation commands have been rerun
- [ ] repository working tree is clean

## Suggested final validation run

From the repo root:

```bash
go test ./...
go test ./infer/packs ./infer/httpserver -run '^$' -bench . -benchmem
go run ./cmd/infergo-export help
go run ./cmd/infergo-serve -h
```

Clean-room path:

1. install `infergo-export`, `infergo-serve`, and `infergo-parity`
2. export a supported family-1 model
3. load the bundle from a tiny Go app outside the repo
4. serve the bundle
5. run parity
