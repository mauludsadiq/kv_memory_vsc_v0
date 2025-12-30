use kv_memory_vsc_v0::KVMemV0;

fn max_abs(x: &[f64]) -> f64 {
    x.iter().map(|v| v.abs()).fold(0.0, f64::max)
}
fn argmax_abs(x: &[f64]) -> usize {
    let mut bi = 0usize;
    let mut bv = -1.0f64;
    for (i, v) in x.iter().enumerate() {
        let a = v.abs();
        if a > bv {
            bv = a;
            bi = i;
        }
    }
    bi
}
fn is_hit(out: &[f64], idx: usize, thr: f64) -> bool {
    argmax_abs(out) == idx && max_abs(out) > thr
}
fn e(d: usize, idx: usize, s: f64) -> Vec<f64> {
    let mut v = vec![0.0f64; d];
    v[idx] = s;
    v
}

#[test]
fn baseline_misses_after_truncation_even_without_drift() {
    // Baseline: M=0, so only window KV exists. With L=1, one eviction makes earlier fact unreachable.
    let (l, d) = (1usize, 8usize);
    let (tau_reuse, tau_novel, g_write) = (0.9f64, 0.5f64, 0.25f64);
    let thr = 5.0f64;

    let mut base = KVMemV0::new(l, 0, d, tau_reuse, tau_novel, g_write);

    // Write FACT A into the *window* (baseline has no memory).
    let k_a = e(d, 0, 1.0);
    let v_a = e(d, 0, 60.0); // large amplitude for deterministic detection in kvmem (baseline should still miss after eviction)
    let _ = base.step(vec![0.0; d], k_a.clone(), v_a, true);

    // Evict from window (L=1) with a key anti-aligned to the query to keep attention weight off the window token.
    let k_evict = e(d, 0, -1.0);
    let _ = base.step(vec![0.0; d], k_evict, vec![0.0; d], false);

    // Query for A (query is e0). In baseline, A is out of window => MISS.
    let out = base.step(k_a, vec![0.0; d], vec![0.0; d], false);
    assert!(!is_hit(&out, 0, thr));
}

#[test]
fn fidelity_decay_g025_hits_until_n3_then_miss_at_n4() {
    // KV-memory: M=1. Store FACT A once, then apply n decay-writes (reuse updates with v=0),
    // measuring when the readback drops below the HIT threshold.
    //
    // Model: vm_0 = g*A, vm_n = (1-g)^n * vm_0.
    // Query-time window contains an eviction token with cosine = -1, so memory attention weight is:
    //   α = exp(1) / (exp(1) + exp(-1)) = e^2/(e^2+1) ≈ 0.881.
    // Output magnitude ≈ α * vm_n.
    // With A=60, g=0.25 => vm_0=15, α*vm_0≈13.2.
    // Threshold thr=5 => HIT for n=0..3, MISS for n>=4.
    let (l, d, m) = (1usize, 8usize, 1usize);
    let (tau_reuse, tau_novel, g_write) = (0.9f64, 0.5f64, 0.25f64);
    let (a_amp, thr) = (60.0f64, 5.0f64);

    let mut mem = KVMemV0::new(l, m, d, tau_reuse, tau_novel, g_write);

    let q_a = e(d, 0, 1.0);
    let k_a = e(d, 0, 1.0);
    let v_a = e(d, 0, a_amp);

    // Store A (novel write into empty memory).
    let _ = mem.step(vec![0.0; d], k_a.clone(), v_a, true);

    // Helper tokens:
    let k_evict = e(d, 0, -1.0);          // keeps window similarity at -1 w.r.t q_a
    let k_decay = e(d, 0,  1.0);          // same key => reuse path triggers EMA update
    let v_zero  = vec![0.0f64; d];

    // n = 0..6
    for n in 0..=6usize {
        // Ensure FACT A is NOT in the window at query time.
        // (a) evict once before first query
        // (b) after each decay-write, evict again (since decay-write inserts k_decay into window)
        if n == 0 {
            let _ = mem.step(vec![0.0; d], k_evict.clone(), v_zero.clone(), false);
        } else {
            let _ = mem.step(vec![0.0; d], k_decay.clone(), v_zero.clone(), true);   // decay write
            let _ = mem.step(vec![0.0; d], k_evict.clone(), v_zero.clone(), false);  // evict
        }

        // Query for A
        let out = mem.step(q_a.clone(), v_zero.clone(), v_zero.clone(), false);

        let should_hit = n <= 3;
        assert_eq!(is_hit(&out, 0, thr), should_hit, "unexpected HIT/MISS at n={}", n);
    }
}

#[test]
fn memory_kv_sha256_changes_on_writes_not_on_reads() {
    let (l, d, m) = (1usize, 8usize, 1usize);
    let (tau_reuse, tau_novel, g_write) = (0.9f64, 0.5f64, 0.25f64);

    let mut mem = KVMemV0::new(l, m, d, tau_reuse, tau_novel, g_write);

    let k_a = e(d, 0, 1.0);
    let v_a = e(d, 0, 60.0);
    let k_evict = e(d, 0, -1.0);
    let v_zero  = vec![0.0f64; d];

    // Write A => memory changes
    let h0 = mem.memory_kv_sha256();
    let _ = mem.step(vec![0.0; d], k_a.clone(), v_a, true);
    let h1 = mem.memory_kv_sha256();
    assert_ne!(h0, h1);

    // Evict + read-only query => memory should not change
    let _ = mem.step(vec![0.0; d], k_evict.clone(), v_zero.clone(), false);
    let _ = mem.step(k_a.clone(), v_zero.clone(), v_zero.clone(), false);
    let h2 = mem.memory_kv_sha256();
    assert_eq!(h1, h2);

    // Decay write (reuse) => memory changes
    let _ = mem.step(vec![0.0; d], k_a.clone(), v_zero.clone(), true);
    let h3 = mem.memory_kv_sha256();
    assert_ne!(h2, h3);

    // Read-only again => stable
    let _ = mem.step(vec![0.0; d], k_evict, v_zero.clone(), false);
    let _ = mem.step(k_a, v_zero.clone(), v_zero, false);
    let h4 = mem.memory_kv_sha256();
    assert_eq!(h3, h4);
}
