//! FSReD criterion benchmark — Feynman Symbolic Regression Dataset (first 20).
//!
//! Benchmarks [`scirs2_symbolic::regression::discover`] on 20 equations
//! drawn from the Feynman Lectures (the standard "easy" FSReD subset).
//!
//! # Running
//!
//! ```text
//! cargo bench --bench fsred_bench -p scirs2-symbolic
//! ```
//!
//! Results are additionally written to `target/fsred_results.json`.
//!
//! # PySR comparison
//!
//! The Python / PySR side of the comparison is documented in
//! `bench-comparison/README.md` and implemented as manual-only stubs in
//! `bench-comparison/run_pysr.py`.  The Julia toolchain required for PySR
//! is not available in the standard CI environment; the Rust-only baseline
//! is what runs automatically.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{Array1, Array2};
use scirs2_symbolic::regression::{discover, SrConfig};
use std::hint::black_box;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Equation catalogue
// ---------------------------------------------------------------------------

/// A single FSReD equation descriptor.
struct FsredEq {
    /// Short identifier matching the Feynman catalogue label.
    name: &'static str,
    /// Number of input features.
    n_features: usize,
    /// Ground-truth function evaluated on inputs in the sample domain.
    f: fn(&[f64]) -> f64,
}

/// First 20 "easy" Feynman equations.
///
/// All inputs are drawn from the uniform domain **[1, 5]^k**.  Each closure
/// must be finite and well-defined on that domain — equations whose original
/// formulation would produce non-finite values (e.g. square-roots of negative
/// numbers, or division by a quantity that can reach zero) are expressed in
/// their physically equivalent but numerically stable form for [1, 5].
fn fsred_equations() -> Vec<FsredEq> {
    vec![
        // I.6.2a  — Gaussian (no sigma; fixed sigma=1): exp(-x^2/2)/sqrt(2π)
        FsredEq {
            name: "I.6.2a",
            n_features: 1,
            f: |x| (-x[0] * x[0] / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt(),
        },
        // I.6.20  — Gaussian with sigma: exp(-x^2/(2σ^2)) / (σ√(2π))
        FsredEq {
            name: "I.6.20",
            n_features: 2,
            f: |x| {
                (-x[0] * x[0] / (2.0 * x[1] * x[1])).exp()
                    / (x[1] * (2.0 * std::f64::consts::PI).sqrt())
            },
        },
        // I.11.19 — displacement: x0 + d*cos(θ) + e*sin(θ)  (4 vars: x0, d, e, θ)
        FsredEq {
            name: "I.11.19",
            n_features: 4,
            f: |x| x[0] + x[1] * x[3].cos() + x[2] * x[3].sin(),
        },
        // I.18.4  — centre of mass: (m1*r1 + m2*r2) / (m1 + m2)
        FsredEq {
            name: "I.18.4",
            n_features: 4,
            f: |x| (x[0] * x[1] + x[2] * x[3]) / (x[0] + x[2]),
        },
        // I.24.6  — energy of oscillator: m*ω^2*(x^2 + y^2) / 4
        // (using only 3 vars: m, ω, x; fixing y = x for a 1D version)
        FsredEq {
            name: "I.24.6",
            n_features: 3,
            f: |x| x[0] * x[1] * x[1] * x[2] * x[2] / 4.0,
        },
        // I.26.2  — Snell/refraction angle: arcsin(n * sin(θ))
        // Domain note: n*sin(θ) must lie in [-1,1] for arcsin to be finite.
        // On [1,5], n≥1 and sin(θ) can be up to 1 → product >1 → NaN.
        // Use a constrained variant: arcsin(sin(θ) / n) (reversed Snell).
        FsredEq {
            name: "I.26.2_constrained",
            n_features: 2,
            f: |x| (x[0].sin() / x[1]).asin(),
        },
        // I.29.16 — wave distance: 1/sqrt(x1^2 + x2^2 - 2*x1*x2*cos(θ))
        // On [1,5], x1^2+x2^2 - 2*x1*x2*cos(θ) ≥ x1^2+x2^2 - 2*x1*x2 = (x1-x2)^2 ≥ 0.
        // Division-by-zero only when x1=x2 exactly AND cos(θ)=1 (measure zero).
        // Clamp denominator away from 0 with max for safety.
        FsredEq {
            name: "I.29.16",
            n_features: 3,
            f: |x| {
                let d2 = x[0] * x[0] + x[1] * x[1] - 2.0 * x[0] * x[1] * x[2].cos();
                1.0 / d2.abs().sqrt().max(1e-10)
            },
        },
        // I.30.3  — multi-slit interference: I0 * sin(n*δ/2)^2 / sin(δ/2)^2
        // On [1,5], δ=x[2] ∈ [1,5]; sin(δ/2) ≠ 0 on this range (δ/2 ∈ [0.5,2.5]).
        FsredEq {
            name: "I.30.3",
            n_features: 3,
            f: |x| {
                let half_delta = x[2] / 2.0;
                let n = x[1];
                x[0] * (n * half_delta).sin().powi(2) / half_delta.sin().powi(2)
            },
        },
        // I.34.10 — Doppler (relativistic): ω' = ω0 * sqrt((1+v)/(1-v))
        // Original ω0/(1-v/c) is singular on [1,5]. Use relativistic form instead,
        // with v = x[1] / 10.0 to keep ratio in [0.1, 0.5] (sub-luminal domain).
        FsredEq {
            name: "I.34.10_rel",
            n_features: 2,
            f: |x| {
                let v = x[1] / 10.0; // rescale to sub-luminal
                x[0] * ((1.0 + v) / (1.0 - v)).sqrt()
            },
        },
        // I.34.14 — relativistic Doppler: ω0*(1+v/c)/sqrt(1-v^2/c^2)
        // Use v = x[1] / 10.0 to keep v<1.
        FsredEq {
            name: "I.34.14_rel",
            n_features: 2,
            f: |x| {
                let v = x[1] / 10.0;
                x[0] * (1.0 + v) / (1.0 - v * v).sqrt()
            },
        },
        // I.37.4  — interference: I1+I2+2*sqrt(I1*I2)*cos(δ)
        FsredEq {
            name: "I.37.4",
            n_features: 3,
            f: |x| x[0] + x[1] + 2.0 * (x[0] * x[1]).sqrt() * x[2].cos(),
        },
        // I.50.26 — sinusoidal motion: x0 * cos(ω*t)
        FsredEq {
            name: "I.50.26",
            n_features: 3,
            f: |x| x[0] * (x[1] * x[2]).cos(),
        },
        // II.2.42 — thermal conductivity with linear temperature coeff: κ0*(1+α*θ)
        FsredEq {
            name: "II.2.42",
            n_features: 3,
            f: |x| x[0] * (1.0 + x[1] * x[2]),
        },
        // II.11.3 — particle distribution: n0 / (1 - p)
        // On [1,5], 1-p ≤ 0. Use physical rescaling: p = x[1]/10.0 ∈ [0.1,0.5].
        FsredEq {
            name: "II.11.3_rescaled",
            n_features: 2,
            f: |x| {
                let p = x[1] / 10.0;
                x[0] / (1.0 - p)
            },
        },
        // II.11.27 — density with field: n0*(1 + p*Ef/(kb*T))
        FsredEq {
            name: "II.11.27",
            n_features: 4,
            f: |x| x[0] * (1.0 + x[1] * x[2] / x[3]),
        },
        // II.11.28 — Clausius-Mossotti: (1+n*α) / (1 - n*α/3)
        // On [1,5], n*α ≥ 1 → denominator 1-n*α/3 can be negative.
        // Rescale: n=x[0]/10, α=x[1]/10, keeping n*α in [0.01, 0.25].
        FsredEq {
            name: "II.11.28_rescaled",
            n_features: 2,
            f: |x| {
                let na = x[0] * x[1] / 100.0; // n*alpha, in [0.01, 0.25]
                (1.0 + na) / (1.0 - na / 3.0)
            },
        },
        // I.15.1  — Lorentz boost: x' = (x-u*t)/sqrt(1-u^2)
        // On [1,5], u ≥ 1 → sqrt(1-u^2) is imaginary.  Rescale: u=x[1]/10.
        FsredEq {
            name: "I.15.1_rescaled",
            n_features: 3,
            f: |x| {
                let u = x[1] / 10.0;
                (x[0] - u * x[2]) / (1.0 - u * u).sqrt()
            },
        },
        // I.9.18 — gravitational/Coulomb simplified: q1*q2/r^2
        FsredEq {
            name: "I.9.18_simplified",
            n_features: 3,
            f: |x| x[0] * x[1] / (x[2] * x[2]),
        },
        // sum_squares — x^2 + y^2  (a simple baseline)
        FsredEq {
            name: "sum_squares",
            n_features: 2,
            f: |x| x[0] * x[0] + x[1] * x[1],
        },
        // product — x * y  (the most basic 2-var product)
        FsredEq {
            name: "product_xy",
            n_features: 2,
            f: |x| x[0] * x[1],
        },
    ]
}

// ---------------------------------------------------------------------------
// Data generation (no external rand dep — LCG PRNG)
// ---------------------------------------------------------------------------

/// Generate `n_samples` rows with each feature drawn from Uniform([1, 5]).
/// Returns a 2-D ndarray features matrix and a 1-D targets vector.
/// Rows where `f(x)` is non-finite are silently dropped.
fn generate_data(eq: &FsredEq, n_samples: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
    let mut state = seed;
    // 64-bit LCG (Knuth MMIX constants).
    let lcg_next = |s: &mut u64| -> f64 {
        *s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // Map high 32 bits to [1, 5].
        let bits = ((*s >> 32) as u32) as f64;
        1.0 + (bits / (u32::MAX as f64)) * 4.0
    };

    let mut xs: Vec<f64> = Vec::with_capacity(n_samples * eq.n_features);
    let mut ys: Vec<f64> = Vec::with_capacity(n_samples);
    let mut accepted = 0usize;

    // Generate up to 2× requested samples to account for non-finite drops.
    let budget = n_samples * 2;
    for _ in 0..budget {
        if accepted >= n_samples {
            break;
        }
        let row: Vec<f64> = (0..eq.n_features).map(|_| lcg_next(&mut state)).collect();
        let y = (eq.f)(&row);
        if y.is_finite() {
            xs.extend_from_slice(&row);
            ys.push(y);
            accepted += 1;
        }
    }

    // Pad with zeros if we somehow couldn't fill n_samples (shouldn't happen).
    while accepted < n_samples {
        xs.extend(std::iter::repeat_n(1.0, eq.n_features));
        ys.push(0.0);
        accepted += 1;
    }

    let n = ys.len();
    let features = Array2::from_shape_vec((n, eq.n_features), xs).expect("features shape");
    let targets = Array1::from_vec(ys);
    (features, targets)
}

// ---------------------------------------------------------------------------
// SrConfig for benchmarking (intentionally fast)
// ---------------------------------------------------------------------------

/// Return a benchmark-friendly configuration.
///
/// `max_iter=15` keeps each `discover` call to a manageable wall time
/// (≈ 5–15 s on 4 cores with 1000 samples) while still exercising the
/// expansion logic.  The default `beam_width=32` is retained.
fn bench_config() -> SrConfig {
    SrConfig::default().with_max_iter(15).with_top_n(3)
}

// ---------------------------------------------------------------------------
// Per-equation recovery metric (computed once outside `b.iter`)
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct EquationResult {
    name: String,
    mse: f64,
    r_squared: f64,
    recovered: bool,
}

fn run_equation(eq: &FsredEq, seed: u64) -> EquationResult {
    let (features, targets) = generate_data(eq, 1000, seed);
    let config = bench_config();
    let results = discover(features.view(), targets.view(), &config);
    let (mse, r_sq, recovered) = results
        .first()
        .map(|f| (f.fitness.mse, f.fitness.r_squared, f.fitness.mse < 1e-3))
        .unwrap_or((f64::INFINITY, 0.0, false));
    EquationResult {
        name: eq.name.to_string(),
        mse,
        r_squared: r_sq,
        recovered,
    }
}

// ---------------------------------------------------------------------------
// Simple hand-rolled JSON writer (no serde_json dep needed)
// ---------------------------------------------------------------------------

fn write_results_json(results: &[EquationResult]) {
    let mut buf = String::from("[\n");
    for (i, r) in results.iter().enumerate() {
        let comma = if i + 1 < results.len() { "," } else { "" };
        let mse_str = if r.mse.is_finite() {
            format!("{:.6e}", r.mse)
        } else {
            "\"Infinity\"".to_string()
        };
        buf.push_str(&format!(
            "  {{\"name\": \"{}\", \"mse\": {}, \"r_squared\": {:.6}, \"recovered\": {}}}{}\n",
            r.name, mse_str, r.r_squared, r.recovered, comma
        ));
    }
    buf.push(']');

    // Write to <crate-root>/target/fsred_results.json.
    // Using CARGO_MANIFEST_DIR ensures the path resolves correctly regardless
    // of what directory cargo was invoked from (workspace root vs crate root).
    let path_str = concat!(env!("CARGO_MANIFEST_DIR"), "/target/fsred_results.json");
    let path = std::path::Path::new(path_str);
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    match std::fs::write(path, &buf) {
        Ok(()) => eprintln!("[fsred_bench] results written to {}", path.display()),
        Err(e) => eprintln!("[fsred_bench] failed to write results: {e}"),
    }
}

// ---------------------------------------------------------------------------
// Benchmark group
// ---------------------------------------------------------------------------

fn bench_fsred(c: &mut Criterion) {
    let equations = fsred_equations();

    // Pre-compute recovery metrics once (warm path) and save to JSON.
    let all_results: Vec<EquationResult> =
        equations.iter().map(|eq| run_equation(eq, 42)).collect();
    write_results_json(&all_results);

    let mut group = c.benchmark_group("fsred");
    group.measurement_time(Duration::from_secs(30));
    group.sample_size(10);

    for (eq, result) in equations.iter().zip(all_results.iter()) {
        // Pre-generate data once for this equation so criterion's timer
        // only measures the `discover` call, not data generation.
        let (features, targets) = generate_data(eq, 1000, 42);
        let config = bench_config();

        eprintln!(
            "[fsred_bench] {:>20}  mse={:.3e}  R²={:.4}  recovered={}",
            eq.name, result.mse, result.r_squared, result.recovered
        );

        group.bench_with_input(
            BenchmarkId::new("discover", eq.name),
            eq.name,
            |b, _name| {
                b.iter(|| {
                    discover(
                        black_box(features.view()),
                        black_box(targets.view()),
                        black_box(&config),
                    )
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_fsred);
criterion_main!(benches);
