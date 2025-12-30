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
fn status(out: &[f64], idx: usize, thr: f64) -> &'static str {
    if is_hit(out, idx, thr) { "HIT" } else { "MISS" }
}
fn e(d: usize, idx: usize, s: f64) -> Vec<f64> {
    let mut v = vec![0.0f64; d];
    v[idx] = s;
    v
}

fn main() {
    let (l, d) = (1usize, 8usize);
    let (tau_reuse, tau_novel) = (0.9f64, 0.5f64);
    let g_write = 0.25f64;
    let (a_amp, thr) = (60.0f64, 5.0f64);
    let n_max = 6usize;

    println!(
        "fidelity_params: L={} d={} A={} thr={} tau_reuse={} tau_novel={} g_write={} n_max={}",
        l, d, a_amp, thr, tau_reuse, tau_novel, g_write, n_max
    );

    let q_a = e(d, 0, 1.0);
    let k_a = e(d, 0, 1.0);
    let v_a = e(d, 0, a_amp);

    let k_evict = e(d, 0, -1.0);
    let k_decay = e(d, 0,  1.0);
    let v_zero  = vec![0.0f64; d];

    // Baseline (M=0)
    let mut base = KVMemV0::new(l, 0, d, tau_reuse, tau_novel, g_write);
    let _ = base.step(vec![0.0; d], k_a.clone(), v_a.clone(), true);

    print!("baseline:");
    for n in 0..=n_max {
        if n == 0 {
            let _ = base.step(vec![0.0; d], k_evict.clone(), v_zero.clone(), false);
        } else {
            let _ = base.step(vec![0.0; d], k_decay.clone(), v_zero.clone(), false);
            let _ = base.step(vec![0.0; d], k_evict.clone(), v_zero.clone(), false);
        }
        let out = base.step(q_a.clone(), v_zero.clone(), v_zero.clone(), false);
        print!(" n{}={}", n, status(&out, 0, thr));
    }
    println!();
    println!("baseline_state_sha256: {}", base.state_sha256());

    // KV-mem (M=1)
    let mut mem = KVMemV0::new(l, 1, d, tau_reuse, tau_novel, g_write);
    let _ = mem.step(vec![0.0; d], k_a.clone(), v_a, true);

    print!("kvmem_g025:");
    for n in 0..=n_max {
        if n == 0 {
            let _ = mem.step(vec![0.0; d], k_evict.clone(), v_zero.clone(), false);
        } else {
            let _ = mem.step(vec![0.0; d], k_decay.clone(), v_zero.clone(), true);
            let _ = mem.step(vec![0.0; d], k_evict.clone(), v_zero.clone(), false);
        }
        let out = mem.step(q_a.clone(), v_zero.clone(), v_zero.clone(), false);
        print!(" n{}={}", n, status(&out, 0, thr));
    }
    println!();
    println!("kvmem_state_sha256: {}", mem.state_sha256());
    println!("kvmem_memory_kv_sha256: {}", mem.memory_kv_sha256());
}
