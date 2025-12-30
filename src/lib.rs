use sha2::{Digest, Sha256};

#[derive(Clone)]
pub struct KVMemV0 {
    pub l_window: usize,
    pub m_slots: usize,
    pub d: usize,
    pub tau_reuse: f64,
    pub tau_novel: f64,
    pub g_write: f64,

    kw: Vec<Vec<f64>>,
    vw: Vec<Vec<f64>>,

    km: Vec<Vec<f64>>,
    vm: Vec<Vec<f64>>,
    age: Vec<u64>,
}

impl KVMemV0 {
    pub fn new(l_window: usize, m_slots: usize, d: usize, tau_reuse: f64, tau_novel: f64, g_write: f64) -> Self {
        assert!(d > 0);
        assert!(tau_reuse >= -1.0 && tau_reuse <= 1.0);
        assert!(tau_novel >= -1.0 && tau_novel <= 1.0);
        assert!(g_write > 0.0 && g_write <= 1.0);

        Self {
            l_window,
            m_slots,
            d,
            tau_reuse,
            tau_novel,
            g_write,
            kw: vec![],
            vw: vec![],
            km: vec![vec![0.0; d]; m_slots],
            vm: vec![vec![0.0; d]; m_slots],
            age: vec![0; m_slots],
        }
    }

    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    fn norm(a: &[f64]) -> f64 {
        Self::dot(a, a).sqrt()
    }

    fn cosine(a: &[f64], b: &[f64]) -> f64 {
        if a.iter().any(|x| x.is_nan()) || b.iter().any(|x| x.is_nan()) {
            return f64::NEG_INFINITY;
        }
        let na = Self::norm(a);
        let nb = Self::norm(b);
        if na == 0.0 || nb == 0.0 {
            0.0
        } else {
            Self::dot(a, b) / (na * nb)
        }
    }

    fn softmax(scores: &[f64]) -> Vec<f64> {
        let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
        let z: f64 = exps.iter().sum();
        if z == 0.0 || z.is_nan() {
            let n = scores.len().max(1) as f64;
            return vec![1.0 / n; scores.len()];
        }
        exps.into_iter().map(|e| e / z).collect()
    }

    fn weighted_sum(w: &[f64], vecs: &[Vec<f64>]) -> Vec<f64> {
        let d = vecs[0].len();
        let mut out = vec![0.0; d];
        for (wi, v) in w.iter().zip(vecs.iter()) {
            for j in 0..d {
                out[j] += wi * v[j];
            }
        }
        out
    }

    fn push_window(&mut self, k: Vec<f64>, v: Vec<f64>) {
        self.kw.push(k);
        self.vw.push(v);
        if self.kw.len() > self.l_window {
            self.kw.remove(0);
            self.vw.remove(0);
        }
    }

    fn max_sim(&self, k: &[f64]) -> (usize, f64) {
        if self.m_slots == 0 {
            return (0, f64::NEG_INFINITY);
        }

        let mut best_i = 0usize;
        let mut best_s = f64::NEG_INFINITY;
        for i in 0..self.m_slots {
            let s = Self::cosine(k, &self.km[i]);
            if s > best_s {
                best_s = s;
                best_i = i;
            }
        }
        (best_i, best_s)
    }

    fn choose_slot(&self, k: &[f64]) -> usize {
        if self.m_slots == 0 {
            return 0;
        }

        let (best_i, best_s) = self.max_sim(k);

        if best_s >= self.tau_reuse {
            return best_i;
        }

        let mut j = 0usize;
        let mut best_age = self.age[0];
        for i in 1..self.m_slots {
            if self.age[i] > best_age {
                best_age = self.age[i];
                j = i;
            }
        }
        j
    }

    fn write_memory_novelty_gated(&mut self, k: Vec<f64>, v: Vec<f64>, write_event: bool) {
        if self.m_slots == 0 {
            return;
        }

        for i in 0..self.m_slots {
            self.age[i] += 1;
        }

        if !write_event {
            return;
        }

        let (_, best_s) = self.max_sim(&k);

        if best_s >= self.tau_novel {
            return;
        }

        let j = self.choose_slot(&k);

        let g = self.g_write;
        for t in 0..self.d {
            self.km[j][t] = (1.0 - g) * self.km[j][t] + g * k[t];
            self.vm[j][t] = (1.0 - g) * self.vm[j][t] + g * v[t];
        }
        self.age[j] = 0;
    }

    pub fn step(&mut self, q: Vec<f64>, k: Vec<f64>, v: Vec<f64>, write_event: bool) -> Vec<f64> {
        self.write_memory_novelty_gated(k.clone(), v.clone(), write_event);
        self.push_window(k, v);

        let mut keys = self.kw.clone();
        let mut vals = self.vw.clone();

        if self.m_slots > 0 {
            keys.extend(self.km.clone());
            vals.extend(self.vm.clone());
        }

        let scale = (self.d as f64).sqrt();
        let scores: Vec<f64> = keys.iter().map(|kk| Self::dot(&q, kk) / scale).collect();
        let w = Self::softmax(&scores);
        Self::weighted_sum(&w, &vals)
    }

    pub fn state_sha256(&self) -> String {
        let mut h = Sha256::new();

        h.update((self.l_window as u64).to_le_bytes());
        h.update((self.m_slots as u64).to_le_bytes());
        h.update((self.d as u64).to_le_bytes());
        h.update(self.tau_reuse.to_le_bytes());
        h.update(self.tau_novel.to_le_bytes());
        h.update(self.g_write.to_le_bytes());

        for row in &self.kw {
            for x in row {
                h.update(x.to_le_bytes());
            }
        }
        for row in &self.vw {
            for x in row {
                h.update(x.to_le_bytes());
            }
        }
        for row in &self.km {
            for x in row {
                h.update(x.to_le_bytes());
            }
        }
        for row in &self.vm {
            for x in row {
                h.update(x.to_le_bytes());
            }
        }
        for a in &self.age {
            h.update(a.to_le_bytes());
        }

        hex::encode(h.finalize())
    }
    pub fn memory_kv_sha256(&self) -> String {
        let mut h = Sha256::new();
        h.update((self.m_slots as u64).to_le_bytes());
        h.update((self.d as u64).to_le_bytes());
        h.update(self.tau_reuse.to_le_bytes());
        h.update(self.tau_novel.to_le_bytes());
        h.update(self.g_write.to_le_bytes());
        for row in &self.km { for x in row { h.update(x.to_le_bytes()); } }
        for row in &self.vm { for x in row { h.update(x.to_le_bytes()); } }
        hex::encode(h.finalize())
    }
}
