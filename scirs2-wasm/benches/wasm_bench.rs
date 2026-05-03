//! Criterion benchmarks for scirs2-wasm operations.
//!
//! These benchmarks measure native-Rust performance of the operations that are
//! compiled to WASM.  Run with:
//!
//! ```sh
//! cargo bench -p scirs2-wasm --bench wasm_bench
//! ```
//!
//! For browser-side benchmarks comparing against ml5.js and tfjs-wasm, see
//! `benches/js/comparison_bench.html`.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

// ---------------------------------------------------------------------------
// Matrix multiply
// ---------------------------------------------------------------------------

/// Row-major square matrix multiply: C = A × B  (O(n³))
fn matmul_f32(a: &[f32], b: &[f32], n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; n * n];
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

fn bench_matrix_multiply(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_f32");

    for &n in &[32usize, 64, 128, 256] {
        let elements = (n * n) as u64;
        group.throughput(Throughput::Elements(elements));

        let a: Vec<f32> = (0..n * n).map(|i| i as f32 * 0.001).collect();
        let b: Vec<f32> = (0..n * n).map(|i| (n * n - i) as f32 * 0.001).collect();

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bch, &n| {
            bch.iter(|| matmul_f32(black_box(&a), black_box(&b), black_box(n)));
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Dot product
// ---------------------------------------------------------------------------

fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn bench_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_product_f32");

    for &len in &[256usize, 1024, 4096, 16384] {
        group.throughput(Throughput::Elements(len as u64));

        let a: Vec<f32> = (0..len).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..len).map(|i| (len - i) as f32).collect();

        group.bench_with_input(BenchmarkId::from_parameter(len), &len, |bch, _| {
            bch.iter(|| dot_f32(black_box(&a), black_box(&b)));
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Naive DFT (reference for FFT comparison)
// ---------------------------------------------------------------------------

/// Discrete Fourier Transform (O(n²)) — reference implementation.
///
/// Returns interleaved `[re0, im0, re1, im1, …]` output.
fn naive_dft(input: &[f32]) -> Vec<f32> {
    let n = input.len();
    let mut out = vec![0.0f32; n * 2];
    for k in 0..n {
        let mut re = 0.0f32;
        let mut im = 0.0f32;
        for (t, &x) in input.iter().enumerate() {
            let angle = -2.0 * std::f32::consts::PI * (k * t) as f32 / n as f32;
            re += x * angle.cos();
            im += x * angle.sin();
        }
        out[k * 2] = re;
        out[k * 2 + 1] = im;
    }
    out
}

fn bench_dft(c: &mut Criterion) {
    let mut group = c.benchmark_group("naive_dft_f32");

    for &n in &[32usize, 64, 128, 256] {
        group.throughput(Throughput::Elements(n as u64));

        let input: Vec<f32> = (0..n)
            .map(|i| (i as f32 * 2.0 * std::f32::consts::PI / n as f32).sin())
            .collect();

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bch, _| {
            bch.iter(|| naive_dft(black_box(&input)));
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Element-wise activation (sigmoid)
// ---------------------------------------------------------------------------

fn sigmoid_f32(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect()
}

fn bench_sigmoid(c: &mut Criterion) {
    let mut group = c.benchmark_group("sigmoid_f32");

    for &len in &[1024usize, 4096, 16384, 65536] {
        group.throughput(Throughput::Elements(len as u64));

        let input: Vec<f32> = (0..len)
            .map(|i| (i as f32 - len as f32 / 2.0) * 0.01)
            .collect();

        group.bench_with_input(BenchmarkId::from_parameter(len), &len, |bch, _| {
            bch.iter(|| sigmoid_f32(black_box(&input)));
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Descriptive statistics (mean + variance in one pass)
// ---------------------------------------------------------------------------

fn mean_variance_f32(data: &[f32]) -> (f32, f32) {
    let n = data.len() as f32;
    let mean = data.iter().copied().sum::<f32>() / n;
    let var = data.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n;
    (mean, var)
}

fn bench_statistics(c: &mut Criterion) {
    let mut group = c.benchmark_group("mean_variance_f32");

    for &len in &[1024usize, 16384, 262144] {
        group.throughput(Throughput::Elements(len as u64));

        let data: Vec<f32> = (0..len).map(|i| i as f32).collect();

        group.bench_with_input(BenchmarkId::from_parameter(len), &len, |bch, _| {
            bch.iter(|| mean_variance_f32(black_box(&data)));
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Softmax
// ---------------------------------------------------------------------------

fn softmax_f32(x: &[f32]) -> Vec<f32> {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = x.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

fn bench_softmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax_f32");

    for &len in &[128usize, 1024, 4096] {
        group.throughput(Throughput::Elements(len as u64));

        let input: Vec<f32> = (0..len).map(|i| i as f32 * 0.1).collect();

        group.bench_with_input(BenchmarkId::from_parameter(len), &len, |bch, _| {
            bch.iter(|| softmax_f32(black_box(&input)));
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// criterion_group / criterion_main
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_matrix_multiply,
    bench_dot_product,
    bench_dft,
    bench_sigmoid,
    bench_statistics,
    bench_softmax,
);
criterion_main!(benches);
