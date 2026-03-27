# Security Policy

## Supported Versions

InferGo is still pre-alpha. Security fixes are most likely to land on the
latest `main` branch state first.

## Reporting a Vulnerability

Please do not open a public GitHub issue for suspected security problems.

Instead, report vulnerabilities privately to the maintainers through the
repository security contact channel.

When reporting, please include:

- a description of the issue
- the affected files, commands, or workflows
- reproduction steps if available
- impact assessment if known
- whether the issue involves secrets, arbitrary code execution, unsafe model
  loading, or data exposure

We will acknowledge reports as quickly as practical and work with reporters on
responsible disclosure.

## Scope Notes

Because InferGo is an inference toolkit, security-sensitive areas include:

- artifact loading and deserialization
- native backend boundaries
- checked-in reference assets and generated bundles
- CI workflows and release automation
- examples that might encourage unsafe runtime patterns
