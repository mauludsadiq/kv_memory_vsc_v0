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

fn report(name: &str, out: &[f64]) {
    let a = argmax_abs(out);
    let m = max_abs(out);
    println!("{name}: argmax_abs={a} max_abs={m:.3}");
}

fn main() {
    let d = 8usize;
    let l = 8usize;

    let tau_reuse = 0.99f64;
    let tau_novel = 0.00f64;
    let g_write = 1.0f64;

    let mut base = KVMemV0::new(l, 0, d, tau_reuse, tau_novel, g_write);
    write_fact(&mut base, d, 0);
    write_fact(&mut base, d, 1);
    write_fact(&mut base, d, 2);
    fill(&mut base, d, 64);
    let b0 = ask(&mut base, d, 0);
    let b1 = ask(&mut base, d, 1);
    let b2 = ask(&mut base, d, 2);

    let mut m2 = KVMemV0::new(l, 2, d, tau_reuse, tau_novel, g_write);
    write_fact(&mut m2, d, 0);
    fill(&mut m2, d, 1);
    write_fact(&mut m2, d, 1);
    fill(&mut m2, d, 1);
    write_fact(&mut m2, d, 2);
    fill(&mut m2, d, 64);
    let m20 = ask(&mut m2, d, 0);
    let m21 = ask(&mut m2, d, 1);
    let m22 = ask(&mut m2, d, 2);

    let mut m3 = KVMemV0::new(l, 3, d, tau_reuse, tau_novel, g_write);
    write_fact(&mut m3, d, 0);
    write_fact(&mut m3, d, 1);
    write_fact(&mut m3, d, 2);
    fill(&mut m3, d, 64);
    let m30 = ask(&mut m3, d, 0);
    let m31 = ask(&mut m3, d, 1);
    let m32 = ask(&mut m3, d, 2);

    println!("baseline");
    report("A", &b0);
    report("B", &b1);
    report("C", &b2);
    println!("baseline_state_sha256: {}", base.state_sha256());

    println!("m2");
    report("A", &m20);
    report("B", &m21);
    report("C", &m22);
    println!("m2_state_sha256: {}", m2.state_sha256());
    println!("m2_memory_kv_sha256: {}", m2.memory_kv_sha256());

    println!("m3");
    report("A", &m30);
    report("B", &m31);
    report("C", &m32);
    println!("m3_state_sha256: {}", m3.state_sha256());
    println!("m3_memory_kv_sha256: {}", m3.memory_kv_sha256());
}
