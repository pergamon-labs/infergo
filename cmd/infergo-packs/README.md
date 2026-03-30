# Pack Discovery

This command lists the curated, checked-in InferGo packs that ship with the
repository.

Run it from the repo root:

```bash
go run ./cmd/infergo-packs
```

To show only text packs:

```bash
go run ./cmd/infergo-packs -task text
```

To show only raw-text-capable text packs:

```bash
go run ./cmd/infergo-packs -task text -raw-text-only
```

To get machine-readable output:

```bash
go run ./cmd/infergo-packs -json
```
