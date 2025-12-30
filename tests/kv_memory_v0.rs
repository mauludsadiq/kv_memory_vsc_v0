use kv_memory_vsc_v0::KVMemV0;

const TOK_FACT: usize = 2;
const TOK_FILL: usize = 1;
const TOK_ASK: usize  = 3;

fn qkv(token: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, bool) {
    match token {
        TOK_FACT => (vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0], true),
        TOK_ASK  => (vec![10.0, 0.0], vec![0.0, 0.0], vec![0.0, 0.0], false),
        _        => (vec![0.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0], false),
    }
}

fn decode_secret(o: &[f64]) -> bool {
    o[1] > o[0]
}

fn run(mut m: KVMemV0, l_fill: usize) -> (bool, String) {
    let (q,k,v,w) = qkv(TOK_FACT);
    let _ = m.step(q,k,v,w);

    for _ in 0..l_fill {
        let (q,k,v,w) = qkv(TOK_FILL);
        let _ = m.step(q,k,v,w);
    }

    let (q,k,v,w) = qkv(TOK_ASK);
    let o = m.step(q,k,v,w);

    (decode_secret(&o), m.state_sha256())
}

#[test]
fn baseline_fails_under_saturation() {
    let tau_reuse = 0.85f64;
    let tau_novel = 0.50f64;
    let g_write = 1.0f64;

    let m = KVMemV0::new(8, 0, 2, tau_reuse, tau_novel, g_write);
    let (ok, _) = run(m, 64);
    assert!(!ok);
}

#[test]
fn kv_memory_succeeds_under_saturation() {
    let tau_reuse = 0.85f64;
    let tau_novel = 0.50f64;
    let g_write = 1.0f64;

    let m = KVMemV0::new(8, 1, 2, tau_reuse, tau_novel, g_write);
    let (ok, _) = run(m, 64);
    assert!(ok);
}

#[test]
fn determinism_state_hash_is_stable() {
    let tau_reuse = 0.85f64;
    let tau_novel = 0.50f64;
    let g_write = 1.0f64;

    let m1 = KVMemV0::new(8, 1, 2, tau_reuse, tau_novel, g_write);
    let m2 = KVMemV0::new(8, 1, 2, tau_reuse, tau_novel, g_write);

    let (ok1, h1) = run(m1, 64);
    let (ok2, h2) = run(m2, 64);

    assert!(ok1 && ok2);
    assert_eq!(h1, h2);
}

#[test]
fn novelty_gate_blocks_duplicate_write() {
    let tau_reuse = 0.85f64;
    let tau_novel = 0.50f64;
    let g_write = 1.0f64;

    let mut m = KVMemV0::new(8, 1, 2, tau_reuse, tau_novel, g_write);

    let (q,k,v,w) = qkv(TOK_FACT);
    let _ = m.step(q,k,v,w);

    let h1 = m.memory_kv_sha256();

    let (q2,k2,v2,w2) = qkv(TOK_FACT);
    let _ = m.step(q2,k2,v2,w2);

    let h2 = m.memory_kv_sha256();

    assert_eq!(h1, h2);
}
