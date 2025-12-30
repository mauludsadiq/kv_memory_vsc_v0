import os
import json
import hashlib

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VSC_DIR = os.path.join(REPO, "vsc")
MANIFEST_JSON = os.path.join(VSC_DIR, "manifest.json")
MANIFEST_SHA = os.path.join(VSC_DIR, "manifest.sha256")

EXCLUDE_DIRS = {".git", "target"}
EXCLUDE_FILES = {
    "vsc/manifest.json",
    "vsc/manifest.sha256",
}

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def list_files(repo_root: str):
    out = []
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for fn in files:
            abs_path = os.path.join(root, fn)
            rel = os.path.relpath(abs_path, repo_root).replace("\\", "/")
            if rel in EXCLUDE_FILES:
                continue
            out.append(rel)
    out.sort()
    return out

def canonical_json_bytes(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

def main():
    os.makedirs(VSC_DIR, exist_ok=True)

    files = []
    for rel in list_files(REPO):
        abs_path = os.path.join(REPO, rel)
        st = os.stat(abs_path)
        files.append({
            "path": rel,
            "bytes": int(st.st_size),
            "sha256": sha256_file(abs_path),
        })

    manifest = {
        "schema": "vsc-manifest/v0.1",
        "spec_id": "kv-memory/v0",
        "version": "0.1.2",
        "repo": "kv_memory_vsc_v0",
        "language": "rust",
        "entrypoints": {
            "lib": "src/lib.rs",
            "bench": "src/bin/bench_kv_memory.rs",
            "bench_capacity": "src/bin/bench_capacity.rs",
            "bench_fidelity_decay": "src/bin/bench_fidelity_decay.rs",
            "tests": [
                "tests/kv_memory_v0.rs",
                "tests/kv_memory_capacity.rs",
                "tests/kv_memory_fidelity_decay.rs",
            ],
        },
        "determinism": {
            "no_rng": True,
            "softmax": "stable max-subtraction; uniform fallback if sumexp==0 or NaN",
            "state_hash": "sha256 over (config, window_KV, memory_KV, ages) in little-endian f64/u64 bytes",
            "tie_break": "argmax ties -> lowest index; LRU ties -> lowest index",
        },
        "pinned_params": {
            # core demo
            "L": 8,
            "M_baseline": 0,
            "M_memory": 1,
            "d": 2,
            "g_write": 1.0,
            "n_fill": 64,
            "tau_reuse": 0.9,

            # capacity benchmark pins
            "cap_L": 8,
            "cap_d": 8,
            "cap_M2": 2,
            "cap_M3": 3,
            "cap_n_fill": 64,
            "cap_thr": 5.0,

            # fidelity-decay benchmark pins
            "fid_L": 1,
            "fid_d": 8,
            "fid_M": 1,
            "fid_A": 60.0,
            "fid_thr": 5.0,
            "fid_tau_reuse": 0.9,
            "fid_tau_novel": 0.5,
            "fid_g_write": 0.25,
            "fid_n_max": 6,
        },
        "expected": {
            "baseline": "UNKNOWN",
            "kv_memory": "SECRET",

            "capacity": {
                "baseline": {"A": "MISS", "B": "MISS", "C": "MISS"},
                "m2": {"A": "MISS", "B": "HIT", "C": "HIT"},
                "m3": {"A": "HIT", "B": "HIT", "C": "HIT"},
            },

            "fidelity_decay": {
                "baseline": {"n0": "MISS", "n1": "MISS", "n2": "MISS", "n3": "MISS", "n4": "MISS", "n5": "MISS", "n6": "MISS"},
                "g025":     {"n0": "HIT",  "n1": "HIT",  "n2": "HIT",  "n3": "HIT",  "n4": "MISS", "n5": "MISS", "n6": "MISS"},
                "params":   {"L": 1, "M": 1, "d": 8, "A": 60.0, "thr": 5.0, "tau_reuse": 0.9, "tau_novel": 0.5, "g_write": 0.25, "n_max": 6},
            },
        },
        "files": files,
    }

    with open(MANIFEST_JSON, "wb") as f:
        f.write(canonical_json_bytes(manifest))

    digest = hashlib.sha256(open(MANIFEST_JSON, "rb").read()).hexdigest()
    with open(MANIFEST_SHA, "w", encoding="utf-8") as f:
        f.write(digest + "\n")

    print(f"wrote {MANIFEST_JSON}")
    print(f"wrote {MANIFEST_SHA}")
    print(f"sha256 {digest}")
    print(f"n_files {len(files)}")

if __name__ == "__main__":
    main()
