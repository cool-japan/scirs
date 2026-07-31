//! Performance validation
//!
//! This module validates performance characteristics of the Lomb-Scargle
//! implementation.
//!
//! NOTE ON "SciPy comparison": this is a pure-Rust crate with no SciPy/Python
//! installation available to benchmark against (and adding a Python/SciPy
//! FFI dependency would violate the crate's Pure Rust policy). The
//! `PerformanceValidationResult` fields are therefore computed as genuine,
//! reproducible measurements of *this crate's* optimized [`lombscargle`]
//! implementation against its own direct/naive reference implementation
//! ([`compute_reference_lombscargle`], the same O(N*F) ground truth
//! `accuracy.rs` uses for numerical-accuracy comparisons) -- not literal
//! SciPy timings. This is clearly documented rather than silently presented
//! as a genuine SciPy comparison.

use super::accuracy::compute_reference_lombscargle;
use super::types::*;
use crate::error::SignalResult;
use crate::lombscargle::{lombscargle, AutoFreqMethod};
use scirs2_core::ndarray::Array1;
use std::time::Instant;

/// Generate a deterministic, non-trivial (irregularly sampled, multi-tone)
/// benchmark signal of length `n` for performance testing.
fn generate_benchmark_signal(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut t = Vec::with_capacity(n);
    let mut signal = Vec::with_capacity(n);
    for i in 0..n {
        // Irregular sampling: base spacing plus a deterministic jitter term.
        let base = i as f64 * 0.37;
        let jitter = 0.1 * ((i as f64 * 0.913).sin());
        let time = base + jitter;
        t.push(time);
        signal.push((2.0 * std::f64::consts::PI * 0.2 * time).sin() + 0.3 * (time * 0.05).cos());
    }
    (t, signal)
}

/// Validate performance characteristics.
///
/// Measures the wall-clock speed and reports a formula-based memory
/// footprint estimate of this crate's optimized Lomb-Scargle implementation
/// relative to its own direct/naive reference implementation, and a
/// scalability score derived from how runtime actually grows with problem
/// size -- replacing a previous stand-in that returned fixed constants
/// (`speed_ratio: 1.2`, `memory_ratio: 0.9`, `scalability_score: 95.0`)
/// regardless of the configuration or the host machine.
#[allow(dead_code)]
pub fn validate_performance_characteristics(
    config: &ScipyValidationConfig,
) -> SignalResult<PerformanceValidationResult> {
    let mut sizes: Vec<usize> = config
        .test_lengths
        .iter()
        .copied()
        .filter(|&n| n >= 8)
        .collect();
    if sizes.is_empty() {
        sizes = vec![64, 256, 1024];
    }
    sizes.sort_unstable();
    sizes.dedup();

    let mut our_times = Vec::with_capacity(sizes.len());
    let mut reference_times = Vec::with_capacity(sizes.len());
    let mut total_n = 0usize;
    let mut total_freqs = 0usize;

    for &n in &sizes {
        let (t, signal) = generate_benchmark_signal(n);
        let n_freqs = (n / 2).max(8);
        let freqs: Vec<f64> = Array1::linspace(0.05, 5.0, n_freqs).to_vec();

        let start = Instant::now();
        let _ = lombscargle(
            &t,
            &signal,
            Some(&freqs),
            Some("standard"),
            Some(true),
            Some(true),
            Some(1.0),
            Some(AutoFreqMethod::Fft),
        )?;
        our_times.push(start.elapsed().as_secs_f64().max(1e-9));

        let start = Instant::now();
        let _ = compute_reference_lombscargle(&t, &signal, &freqs)?;
        reference_times.push(start.elapsed().as_secs_f64().max(1e-9));

        total_n += n;
        total_freqs += n_freqs;
    }

    let total_our: f64 = our_times.iter().sum();
    let total_reference: f64 = reference_times.iter().sum();
    // >1.0 means our implementation is faster than the internal
    // direct/naive reference across the benchmarked sizes.
    let speed_ratio = total_reference / total_our.max(1e-12);

    // Formula-based (not empirically heap-profiled -- that would require
    // either an external memory-profiling dependency or a process-wide
    // custom allocator, both out of scope here) memory-footprint estimate,
    // grounded in the implementations' actual, verified data-structure
    // usage: `lombscargle` converts its inputs into two extra owned f64
    // buffers (`x_f64`, `y_f64`) before computing, on top of the
    // O(n_freqs) output shared by both implementations, whereas
    // `compute_reference_lombscargle` operates directly on the (already
    // f64) input slices and only allocates the O(n_freqs) output.
    let our_words = 2 * total_n + total_freqs;
    let reference_words = total_freqs.max(1);
    let memory_ratio = our_words as f64 / reference_words as f64;

    // Scalability: the empirical growth exponent of our own timings across
    // problem sizes (time ~ n^exponent), scored 100 for linear-or-better
    // scaling (exponent <= 1) degrading to 0 at quadratic-or-worse
    // (exponent >= 2), computed from real measurements rather than assumed.
    let scalability_score = if sizes.len() >= 2 {
        let first_n = sizes[0] as f64;
        let last_n = *sizes.last().expect("sizes is non-empty") as f64;
        let first_t = our_times[0];
        let last_t = *our_times.last().expect("our_times is non-empty");

        let size_ratio = (last_n / first_n).max(1.0 + 1e-9);
        let time_ratio = (last_t / first_t).max(1e-9);
        let exponent = time_ratio.ln() / size_ratio.ln();

        (100.0 * (2.0 - exponent.clamp(1.0, 2.0))).clamp(0.0, 100.0)
    } else {
        // Not enough distinct sizes in the configuration to measure a
        // genuine trend; report the (honest) neutral midpoint rather than
        // fabricating a specific score.
        50.0
    };

    Ok(PerformanceValidationResult {
        speed_ratio,
        memory_ratio,
        scalability_score,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_performance_characteristics_reacts_to_config() {
        // n=10 hits the `n_freqs = (n/2).max(8)` floor (n_freqs=8, not 5),
        // giving a genuinely different n:n_freqs ratio than n=2000 (where
        // n/2 dominates); a fixed-constant stub could not distinguish them.
        let small_config = ScipyValidationConfig {
            test_lengths: vec![10],
            ..ScipyValidationConfig::default()
        };
        let large_config = ScipyValidationConfig {
            test_lengths: vec![512, 2048],
            ..ScipyValidationConfig::default()
        };

        let small_result = validate_performance_characteristics(&small_config)
            .expect("performance validation should succeed");
        let large_result = validate_performance_characteristics(&large_config)
            .expect("performance validation should succeed");

        // All metrics must be finite, and the fabricated implementation
        // always returned the exact same constants (1.2, 0.9, 95.0)
        // regardless of configuration; a genuine implementation reacts to
        // problem size.
        for result in [&small_result, &large_result] {
            assert!(result.speed_ratio.is_finite() && result.speed_ratio > 0.0);
            assert!(result.memory_ratio.is_finite() && result.memory_ratio > 0.0);
            assert!((0.0..=100.0).contains(&result.scalability_score));
        }

        assert_ne!(small_result.memory_ratio, large_result.memory_ratio);
    }

    #[test]
    fn test_memory_ratio_reflects_problem_size() {
        // The memory ratio formula is grounded in actual per-call buffer
        // sizes (2n extra input-conversion words vs n_freqs output words);
        // a configuration with a much larger n relative to n_freqs should
        // show a correspondingly larger ratio.
        let config = ScipyValidationConfig {
            test_lengths: vec![2000],
            ..ScipyValidationConfig::default()
        };
        let result = validate_performance_characteristics(&config).expect("should succeed");
        // n=2000 => n_freqs=1000, our_words=2*2000+1000=5000, ratio=5.0
        assert!((result.memory_ratio - 5.0).abs() < 1e-9);
    }
}
