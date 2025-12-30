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

fn decode(o: &[f64]) -> &'static str {
    if o[1] > o[0] { "SECRET" } else { "UNKNOWN" }
}

fn run(mut m: KVMemV0, l_fill: usize) -> (String, String) {
    let (q,k,v,w) = qkv(TOK_FACT);
    let _ = m.step(q,k,v,w);

    for _ in 0..l_fill {
        let (q,k,v,w) = qkv(TOK_FILL);
        let _ = m.step(q,k,v,w);
    }

    let (q,k,v,w) = qkv(TOK_ASK);
    let o = m.step(q,k,v,w);

    (decode(&o).to_string(), m.state_sha256())
}

fn main() {
    let l_window = 8usize;
    let d = 2usize;
    let l_fill = 64usize;

    let tau_reuse = 0.85f64;
    let tau_novel = 0.50f64;
    let g_write = 1.0f64;

    let base = KVMemV0::new(l_window, 0, d, tau_reuse, tau_novel, g_write);
    let (ans_b, h_b) = run(base, l_fill);

    let mem  = KVMemV0::new(l_window, 1, d, tau_reuse, tau_novel, g_write);
    let (ans_m, h_m) = run(mem, l_fill);

    println!("baseline: {}", ans_b);
    println!("kv-mem:   {}", ans_m);
    println!("baseline_state_sha256: {}", h_b);
    println!("kvmem_state_sha256:    {}", h_m);
}
