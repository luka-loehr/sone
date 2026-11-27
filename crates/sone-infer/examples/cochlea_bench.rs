//! Speed and memory of the Stems network on one core, with random weights:
//! `cargo run -p sone-infer --release --example cochlea_bench [-- sizes... --seconds 8 --reps 3]`.
//! Prints, per size, parameters, multiply-adds per second of audio, CPU
//! seconds per second of audio (this is single-threaded: wall time on one
//! core), the heap peak while separating one chunk and the F16 file size.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use sone_infer::{Cochlea, CochleaConfig, CochleaWorkspace, WeightFile, random};

struct Counting;
static NOW: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            let now = NOW.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(now, Ordering::Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        NOW.fetch_sub(layout.size(), Ordering::Relaxed);
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut sizes = Vec::new();
    let (mut seconds, mut reps) = (8.0f64, 3usize);
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--seconds" => {
                seconds = args[i + 1].parse()?;
                i += 1;
            }
            "--reps" => {
                reps = args[i + 1].parse()?;
                i += 1;
            }
            s => sizes.push(s.to_string()),
        }
        i += 1;
    }
    if sizes.is_empty() {
        sizes = vec!["small".into(), "base".into(), "large".into()];
    }
    println!("| size | params | MMAC/frame | GMAC per s audio | CPU s per s audio | heap peak for {seconds} s chunk (MB) | weights f16 (MB) |");
    println!("|---|---|---|---|---|---|---|");
    for name in sizes {
        let c = CochleaConfig::size(&name).ok_or_else(|| anyhow::anyhow!("unknown size {name}"))?;
        let bytes = random::file("cochlea", &c, &c.tensor_shapes(), 1, true)?;
        let file_mb = bytes.len() as f64 / 1e6;
        let net = Cochlea::load(WeightFile::from_vec(bytes)?)?;
        let len = ((seconds * c.sample_rate as f64) as usize / c.hop) * c.hop;
        let mut rng = random::Rng::new(9);
        let l: Vec<f32> = (0..len).map(|_| rng.sym(0.3)).collect();
        let r: Vec<f32> = (0..len).map(|_| rng.sym(0.3)).collect();
        let mut out = vec![[Vec::new(), Vec::new()]; 4];
        let base = NOW.load(Ordering::Relaxed);
        PEAK.store(base, Ordering::Relaxed);
        let mut ws = CochleaWorkspace::default();
        net.separate_chunk(&mut ws, &l, &r, &mut out);
        let peak = PEAK.load(Ordering::Relaxed) - base;
        let mut best = f64::MAX;
        for _ in 0..reps {
            let t0 = Instant::now();
            net.separate_chunk(&mut ws, &l, &r, &mut out);
            best = best.min(t0.elapsed().as_secs_f64());
        }
        let audio_s = len as f64 / c.sample_rate as f64;
        let fps = c.sample_rate as f64 / c.hop as f64;
        let macs = c.macs_per_frame() as f64;
        println!(
            "| {name} | {:.2} M | {:.1} | {:.2} | {:.4} | {:.1} (+{:.1} weights) | {:.1} |",
            c.num_params() as f64 / 1e6,
            macs / 1e6,
            (macs + c.transform_macs_per_frame() as f64) * fps / 1e9,
            best / audio_s,
            peak as f64 / 1e6,
            file_mb,
            file_mb
        );
    }
    Ok(())
}
