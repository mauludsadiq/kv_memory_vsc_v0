use kv_memory_vsc_v0::KVMemV0;

fn e(d: usize, i: usize, s: f64) -> Vec<f64> {
    let mut v = vec![0.0; d];
    v[i] = s;
    v
}

fn max_abs(x: &[f64]) -> f64 {
    x.iter().fold(0.0, |a, &b| a.max(b.abs()))
}

fn argmax_abs(x: &[f64]) -> usize {
    let mut bi = 0usize;
    let mut bv = f64::NEG_INFINITY;
    for (i, &v) in x.iter().enumerate() {
        let a = v.abs();
        if a > bv {
            bv = a;
            bi = i;
        }
    }
    bi
}

fn write_fact(m: &mut KVMemV0, d: usize, idx: usize) {
    let k = e(d, idx, 10.0);
    let v = e(d, idx, 100.0);
    let q = k.clone();
    let _ = m.step(q, k, v, true);
}

fn fill(m: &mut KVMemV0, d: usize, n: usize) {
    let z = vec![0.0; d];
    for _ in 0..n {
        let _ = m.step(z.clone(), z.clone(), z.clone(), false);
    }
}

fn ask(m: &mut KVMemV0, d: usize, idx: usize) -> Vec<f64> {
    let q = e(d, idx, 10.0);
    let z = vec![0.0; d];
    m.step(q, z.clone(), z, false)
}

#[test]
fn baseline_fails_all_facts_under_saturation() {
    let d = 8usize;
    let l = 8usize;

    let tau_reuse = 0.85f64;
    let tau_novel = 0.50f64;
    let g_write = 1.0f64;

    let mut base = KVMemV0::new(l, 0, d, tau_reuse, tau_novel, g_write);

    write_fact(&mut base, d, 0);
    write_fact(&mut base, d, 1);
    write_fact(&mut base, d, 2);

    fill(&mut base, d, 64);

    for i in 0..3 {
        let out = ask(&mut base, d, i);
        assert!(max_abs(&out) < 1.0);
    }
}

#[test]
fn memory_m2_only_slot_pressure_remains() {
    let d = 8usize;
    let l = 8usize;

    let tau_reuse = 0.85f64;
    let tau_novel = 0.50f64;
    let g_write = 1.0f64;

    let mut m2 = KVMemV0::new(l, 2, d, tau_reuse, tau_novel, g_write);

    write_fact(&mut m2, d, 0);
    fill(&mut m2, d, 1);
    write_fact(&mut m2, d, 1);
    fill(&mut m2, d, 1);
    write_fact(&mut m2, d, 2);

    fill(&mut m2, d, 64);

    let out0 = ask(&mut m2, d, 0);
    let out1 = ask(&mut m2, d, 1);
    let out2 = ask(&mut m2, d, 2);

    assert!(max_abs(&out0) < 1.0);

    assert!(max_abs(&out1) > 5.0);
    assert_eq!(argmax_abs(&out1), 1);

    assert!(max_abs(&out2) > 5.0);
    assert_eq!(argmax_abs(&out2), 2);
}

#[test]
fn memory_m3_keeps_three_facts_under_saturation() {
    let d = 8usize;
    let l = 8usize;

    let tau_reuse = 0.85f64;
    let tau_novel = 0.50f64;
    let g_write = 1.0f64;

    let mut m3 = KVMemV0::new(l, 3, d, tau_reuse, tau_novel, g_write);

    write_fact(&mut m3, d, 0);
    write_fact(&mut m3, d, 1);
    write_fact(&mut m3, d, 2);

    fill(&mut m3, d, 64);

    for i in 0..3 {
        let out = ask(&mut m3, d, i);
        assert!(max_abs(&out) > 5.0);
        assert_eq!(argmax_abs(&out), i);
    }
}
