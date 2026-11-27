//! Chunking of a long signal for a network that runs on fixed chunks:
//! chunks of `chunk` samples, `overlap` samples shared by neighbours and
//! cross-faded linearly (the two weights of a shared sample sum to 1).

pub struct Plan {
    pub starts: Vec<usize>,
    pub chunk: usize,
    pub overlap: usize,
}

impl Plan {
    pub fn new(len: usize, chunk: usize, overlap: usize) -> Self {
        assert!(overlap < chunk, "overlap must be shorter than the chunk");
        let step = chunk - overlap;
        let mut starts = vec![0];
        while starts.last().unwrap() + chunk < len {
            starts.push(starts.last().unwrap() + step);
        }
        Self { starts, chunk, overlap }
    }

    /// Weight of sample `j` of chunk `i`.
    pub fn weight(&self, i: usize, j: usize) -> f32 {
        let o = self.overlap;
        if o == 0 {
            return 1.0;
        }
        let mut w = 1.0f32;
        if i > 0 && j < o {
            w = (j as f32 + 0.5) / o as f32;
        }
        if i + 1 < self.starts.len() && j >= self.chunk - o {
            w *= 1.0 - (j - (self.chunk - o)) as f32 / o as f32 - 0.5 / o as f32;
        }
        w
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weights_sum_to_one() {
        let plan = Plan::new(10_000, 1024, 256);
        let mut sum = vec![0.0f32; 10_000];
        for (i, &s) in plan.starts.iter().enumerate() {
            for j in 0..1024 {
                if s + j < sum.len() {
                    sum[s + j] += plan.weight(i, j);
                }
            }
        }
        assert!(sum.iter().all(|&v| (v - 1.0).abs() < 1e-5));
        assert!(plan.starts.last().unwrap() + 1024 >= 10_000);
    }
}
