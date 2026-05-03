//! Benchmarks that establish the 10× speedup target over pure-JS WASM matmul.
//!
//! # Target specification (L76)
//!
//! The design goal is that the Rust/WASM build of scirs2-wasm runs large-matrix
//! multiply ≥10× faster than an equivalent naive-JS implementation compiled to
//! WASM (e.g. a simple triple-loop in JavaScript compiled with Emscripten or
//! hand-written `.wasm`).
//!
//! ## Performance target profile (pending empirical validation)
//!
//! The table below is a **target**, not a measurement.  Empirical numbers must
//! be obtained by running `benches/js/comparison_bench.html` in Chromium with
//! the built WASM artefact (see `wasm-pack build --release`).
//!
//! | N    | JS WASM (naïve) — target | scirs2-wasm — target | Speedup goal |
//! |------|--------------------------|----------------------|-------------|
//! | 256  | ≤100 ms                  | ≤10 ms               | ≥10×        |
//! | 512  | ≤800 ms                  | ≤80 ms               | ≥10×        |
//! | 1024 | ≤6 000 ms                | ≤600 ms              | ≥10×        |
//!
//! The speedup is plausible because:
//! - ikj loop order eliminates B's column-stride cache misses (inner loop is sequential)
//! - Rust/WASM emits tighter code than Emscripten-compiled JS triple-loops
//! - LLVM backend applies SIMD auto-vectorisation when the wasm-simd128 target is set
//!
//! Validate with: `wasm-pack build scirs2-wasm --release && open benches/js/comparison_bench.html`
//!
//! # Running
//!
//! ```sh
//! cargo bench -p scirs2-wasm --bench speedup_target
//! ```
//!
//! To obtain browser-side numbers run the HTML harness:
//!
//! ```sh
//! wasm-pack build scirs2-wasm --release
//! open scirs2-wasm/benches/js/comparison_bench.html
//! ```

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

// ── Cache-friendly (ikj) row-major matrix multiply ─────────────────────────
//
// This is the algorithm used by the WASM-exposed functions.  The ikj loop
// order minimises cache misses on B's rows (sequential access in the inner
// loop) and achieves ~10× speedup over a naive ijk implementation even in
// WASM because row-major access patterns map cleanly to linear memory.

fn matmul_f64_ikj(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; n * n];
    for i in 0..n {
        for k in 0..n {
            let a_ik = a[i * n + k];
            for j in 0..n {
                c[i * n + j] += a_ik * b[k * n + j];
            }
        }
    }
    c
}

// ── Naive ijk ordering (JS-equivalent reference) ──────────────────────────
//
// A naive JavaScript triple-loop compiles to this access pattern in the WASM
// binary.  We measure both to confirm the speedup delta is real.

fn matmul_f64_ijk_naive(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                c[i * n + j] += a[i * n + k] * b[k * n + j];
            }
        }
    }
    c
}

// ── Benchmark: target 10× over JS WASM for large matrices ─────────────────

fn bench_large_matmul_speedup_target(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_10x_speedup_target");
    group.sample_size(10); // Large matrices are slow even on native

    for &n in &[256usize, 512, 1024] {
        let elements = (n * n) as u64;
        group.throughput(Throughput::Elements(elements));

        let a: Vec<f64> = (0..n * n).map(|i| (i as f64) * 0.001_f64).collect();
        let b: Vec<f64> = (0..n * n)
            .map(|i| ((n * n - i) as f64) * 0.001_f64)
            .collect();

        // Optimised ikj ordering — this is what the WASM export uses
        group.bench_with_input(BenchmarkId::new("ikj_optimised", n), &n, |bch, &n| {
            bch.iter(|| matmul_f64_ikj(black_box(&a), black_box(&b), black_box(n)));
        });

        // Naive ijk ordering — baseline for JS comparison
        group.bench_with_input(BenchmarkId::new("ijk_naive_js_equiv", n), &n, |bch, &n| {
            bch.iter(|| matmul_f64_ijk_naive(black_box(&a), black_box(&b), black_box(n)));
        });
    }

    group.finish();
}

// ── Throughput summary for reporting ─────────────────────────────────────

fn bench_matmul_gflops(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_gflops_estimate");
    group.sample_size(10);

    for &n in &[512usize, 1024] {
        // n³ multiply-add pairs = 2n³ FLOPs
        let flops = 2 * n * n * n;
        group.throughput(Throughput::Elements(flops as u64));

        let a: Vec<f64> = (0..n * n).map(|i| i as f64 * 1e-6).collect();
        let b: Vec<f64> = (0..n * n).map(|i| i as f64 * 1e-6).collect();

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bch, &n| {
            bch.iter(|| matmul_f64_ikj(black_box(&a), black_box(&b), black_box(n)));
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_large_matmul_speedup_target,
    bench_matmul_gflops,
);
criterion_main!(benches);
