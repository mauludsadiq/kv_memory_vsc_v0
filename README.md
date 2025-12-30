# KV-Memory v0 (VSC repo)

**Spec ID:** `kv-memory/v0`

This repository is a minimal, deterministic demonstration of the *mathematical* fix for context truncation in attention:

- baseline attention operates on a sliding window 
  \(\{t-L+1,\dots,t\}\), so old keys fall out of the domain.
- KV-memory v0 extends the attention domain to **window + persistent KV slots**
  \(\{t-L+1,\dots,t\}\ \cup\ \{m_0,\dots,m_{M-1}\}\).

No external retrieval. No CAS. This is an in-model domain change.

## What you get

- `src/lib.rs`: `KVMemV0` implementation (f64, deterministic)
- `src/bin/bench_kv_memory.rs`: FACT/FILL/ASK benchmark
- `tests/kv_memory_v0.rs`: reproducibility + truncation-elimination tests
- `vsc/manifest.json`: pinned manifest
- `vsc/manifest.sha256`: sha256 of canonical manifest bytes
- `scripts/verify_vsc_sha256.sh`: verifies the sha256 of `vsc/manifest.json`

## Build / run

```bash
cargo test
cargo run -q --bin bench_kv_memory
```

Expected output:

- baseline: `UNKNOWN`
- kv-mem: `SECRET`

## VSC manifest

Generate/refresh the manifest and sha:

```bash
python3 scripts/make_manifest.py
bash scripts/verify_vsc_sha256.sh
```
