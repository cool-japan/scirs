use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use scirs2_signal::dwt::{dwt_decompose, dwt_reconstruct, wavedec, waverec, Wavelet};
use scirs2_signal::swt::{iswt, swt, swt_decompose, swt_reconstruct};
use scirs2_signal::waveforms::chirp;
use scirs2_signal::wpt::{reconstruct_from_nodes, wp_decompose};
use std::time::Duration;

fn generate_signal(size: usize) -> Vec<f64> {
    // Generate a chirp signal with increasing frequency
    let fs = 1000.0; // Sample rate in Hz
    let t = (0..size).map(|i| i as f64 / fs).collect::<Vec<f64>>();
    // Return the signal
    chirp(&t, 0.0, 1.0, 100.0, "linear", 0.5).unwrap()
}

fn bench_wavelets_single_level(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_level_dwt");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(50);

    // Signal sizes to test - use smaller sizes to speed up
    let sizes = [1024];

    // Wavelet families to test - reduced for faster execution
    let wavelets = [
        (Wavelet::Haar, "Haar"),
        (Wavelet::DB(4), "DB4"),
        (Wavelet::Meyer, "Meyer"),
        (Wavelet::DMeyer, "DMeyer"),
    ];

    for &size in &sizes {
        let signal = generate_signal(size);

        for (wavelet, name) in &wavelets {
            // Benchmark decomposition
            group.bench_with_input(
                BenchmarkId::new(format!("{}_decompose", name), size),
                &size,
                |b, _| b.iter(|| black_box(dwt_decompose(&signal, *wavelet, None).unwrap())),
            );

            // Also benchmark reconstruction
            // First get coefficients
            let (approx, detail) = dwt_decompose(&signal, *wavelet, None).unwrap();

            // Benchmark reconstruction
            group.bench_with_input(
                BenchmarkId::new(format!("{}_reconstruct", name), size),
                &size,
                |b, _| b.iter(|| black_box(dwt_reconstruct(&approx, &detail, *wavelet).unwrap())),
            );
        }
    }

    group.finish();
}

#[allow(dead_code)]
fn bench_wavelets_multi_level(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_level_dwt");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(100);

    // Signal sizes to test
    let sizes = [1024, 4096, 16384];

    // Wavelet families to test
    let wavelets = [
        (Wavelet::Haar, "Haar"),
        (Wavelet::DB(4), "DB4"),
        (Wavelet::Sym(4), "Sym4"),
        (Wavelet::Coif(3), "Coif3"),
        (Wavelet::Meyer, "Meyer"),
        (Wavelet::DMeyer, "DMeyer"),
    ];

    // Decomposition levels
    let levels = [3, 5];

    for &size in &sizes {
        let signal = generate_signal(size);

        for (wavelet, name) in &wavelets {
            for &level in &levels {
                let level_name = format!("{}_level{}", name, level);

                // Benchmark multi-level decomposition
                group.bench_with_input(
                    BenchmarkId::new(format!("{}_decompose", level_name), size),
                    &size,
                    |b, _| {
                        b.iter(|| black_box(wavedec(&signal, *wavelet, Some(level), None).unwrap()))
                    },
                );

                // Get coefficients for reconstruction benchmark
                let coeffs = wavedec(&signal, *wavelet, Some(level), None).unwrap();

                // Benchmark multi-level reconstruction
                group.bench_with_input(
                    BenchmarkId::new(format!("{}_reconstruct", level_name), size),
                    &size,
                    |b, _| b.iter(|| black_box(waverec(&coeffs, *wavelet).unwrap())),
                );
            }
        }
    }

    group.finish();
}

#[allow(dead_code)]
fn bench_stationary_wavelet_transform(c: &mut Criterion) {
    let mut group = c.benchmark_group("swt");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);

    // Signal sizes to test
    let sizes = [1024, 4096];

    // Wavelet families to test
    let wavelets = [
        (Wavelet::Haar, "Haar"),
        (Wavelet::DB(4), "DB4"),
        (Wavelet::Sym(4), "Sym4"),
        (Wavelet::Meyer, "Meyer"),
        (Wavelet::DMeyer, "DMeyer"),
    ];

    // Decomposition levels
    let levels = [1, 3];

    for &size in &sizes {
        let signal = generate_signal(size);

        for (wavelet, name) in &wavelets {
            for &level in &levels {
                let level_name = format!("{}_level{}", name, level);

                // Benchmark single-level SWT decomposition
                group.bench_with_input(
                    BenchmarkId::new(format!("{}_swt_decompose", level_name), size),
                    &size,
                    |b, _| {
                        b.iter(|| black_box(swt_decompose(&signal, *wavelet, level, None).unwrap()))
                    },
                );

                // Get coefficients for reconstruction benchmark
                let (approx, detail) = swt_decompose(&signal, *wavelet, level, None).unwrap();

                // Benchmark single-level SWT reconstruction
                group.bench_with_input(
                    BenchmarkId::new(format!("{}_swt_reconstruct", level_name), size),
                    &size,
                    |b, _| {
                        b.iter(|| {
                            black_box(swt_reconstruct(&approx, &detail, *wavelet, level).unwrap())
                        })
                    },
                );

                // For level > 1, also benchmark multi-level SWT
                if level > 1 {
                    // Benchmark multi-level SWT decomposition
                    group.bench_with_input(
                        BenchmarkId::new(format!("{}_full_swt", level_name), size),
                        &size,
                        |b, _| b.iter(|| black_box(swt(&signal, *wavelet, level, None).unwrap())),
                    );

                    // Get coefficients for multi-level reconstruction benchmark
                    let (details, approx) = swt(&signal, *wavelet, level, None).unwrap();

                    // Benchmark multi-level SWT reconstruction
                    group.bench_with_input(
                        BenchmarkId::new(format!("{}_full_iswt", level_name), size),
                        &size,
                        |b, _| b.iter(|| black_box(iswt(&details, &approx, *wavelet).unwrap())),
                    );
                }
            }
        }
    }

    group.finish();
}

#[allow(dead_code)]
fn bench_wavelet_packet_transform(c: &mut Criterion) {
    let mut group = c.benchmark_group("wpt");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);

    // Signal sizes to test
    let sizes = [1024, 4096];

    // Wavelet families to test
    let wavelets = [
        (Wavelet::Haar, "Haar"),
        (Wavelet::DB(4), "DB4"),
        (Wavelet::Sym(4), "Sym4"),
        (Wavelet::Meyer, "Meyer"),
        (Wavelet::DMeyer, "DMeyer"),
    ];

    // Decomposition levels
    let levels = [2, 3];

    for &size in &sizes {
        let signal = generate_signal(size);

        for (wavelet, name) in &wavelets {
            for &level in &levels {
                let level_name = format!("{}_level{}", name, level);

                // Benchmark WPT decomposition
                group.bench_with_input(
                    BenchmarkId::new(format!("{}_wp_decompose", level_name), size),
                    &size,
                    |b, _| {
                        b.iter(|| black_box(wp_decompose(&signal, *wavelet, level, None).unwrap()))
                    },
                );

                // Get tree for reconstruction benchmark
                let tree = wp_decompose(&signal, *wavelet, level, None).unwrap();

                // Create nodes list for level-based reconstruction
                let mut nodes = Vec::new();
                for i in 0..(1 << level) {
                    nodes.push((level, i));
                }

                // Benchmark WPT reconstruction
                group.bench_with_input(
                    BenchmarkId::new(format!("{}_wp_reconstruct", level_name), size),
                    &size,
                    |b, _| b.iter(|| black_box(reconstruct_from_nodes(&tree, &nodes).unwrap())),
                );
            }
        }
    }

    group.finish();
}

criterion_group!(
    wavelet_benches,
    bench_wavelets_single_level // Limiting to just single level for faster execution
                                // bench_wavelets_multi_level,
                                // bench_stationary_wavelet_transform,
                                // bench_wavelet_packet_transform
);
criterion_main!(wavelet_benches);
