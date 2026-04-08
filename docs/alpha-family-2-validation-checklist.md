# InferGo Alpha Family-2 Validation Checklist

This checklist exists so the remaining family-2 work is explicit:

- the bridge mechanics are already working locally
- the remaining risk is repeatability in another engineer's hands

Use this with the internal family-2 runbook, not as a replacement for it.

## Purpose

Validate that another engineer or service owner can:

1. scaffold a family-2 bundle
2. run parity
3. serve the bundle
4. understand what the bridge is doing and where its boundaries are

## What should be provided to the validator

- the family-2 bridge definition:
  [`alpha-family-2-entres-bridge.md`](./alpha-family-2-entres-bridge.md)
- the current runbook used by the team
- the expected local model artifact and fixture locations

## What the validator should try

1. regenerate or locate one known-good parity fixture
2. scaffold a local family-2 bundle
3. run `cmd/infergo-entres-parity`
4. run `cmd/infergo-entres-serve`
5. call `/metadata`
6. call `/predict`

## What feedback to capture

- first unclear prerequisite
- first confusing command or argument
- whether the bridge metadata made sense
- whether parity output was understandable
- whether serving output was trustworthy
- any place where they needed live help

## Success bar

This validation is done when:

- another engineer or service owner completes the checklist
- the flow works without undocumented rescue knowledge
- any resulting doc gaps are patched afterward

## Current status

- local family-2 bundle scaffolding: done
- local family-2 parity: done
- local family-2 serving: done
- second-operator validation: pending
