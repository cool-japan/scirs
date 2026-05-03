//! Automatic CPU/GPU dispatch layer for FFT computation.
//!
//! This module exposes `fft_auto_dispatch`, `fft_batch_gpu`, and
//! `overlap_save_gpu` as the high-level public API.  The dispatch logic
//! routes computations to the GPU (via the `wgpu_fft` feature) for large
//! inputs and falls back to the CPU-based [`GpuFftPipeline`] otherwise.
//!
//! # Design
//!
//! The naming deliberately avoids clashing with the existing
//! `GpuFftConfig` / `GpuFftResult` types already present in `types.rs`.
//! New, orthogonal type names (`AutoDispatchConfig`, `DispatchFftOutput`)
//! are used throughout this module.
//!
//! # Feature flags
//!
//! * `wgpu_fft` — enables the wgpu GPU back-end.  When absent (or when
//!   no adapter is available at runtime) every call transparently falls
//!   back to the CPU pipeline.

use scirs2_core::numeric::Complex64;

use super::pipeline::GpuFftPipeline;
use super::types::{FftDirection, GpuFftConfig, GpuFftError, NormalizationMode};
use crate::error::FFTError;

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the auto-dispatch FFT layer.
///
/// Distinct from [`GpuFftConfig`] which configures the underlying pipeline.
#[derive(Debug, Clone)]
pub struct AutoDispatchConfig {
    /// Minimum input length (in complex samples) before the dispatch layer
    /// considers routing to a GPU back-end.  Inputs shorter than this are
    /// always executed on the CPU regardless of available hardware.
    ///
    /// Default: **4096**.
    pub gpu_threshold: usize,

    /// Perform an inverse FFT instead of the default forward FFT.
    ///
    /// Default: **false**.
    pub inverse: bool,
}

impl Default for AutoDispatchConfig {
    fn default() -> Self {
        Self {
            gpu_threshold: 4096,
            inverse: false,
        }
    }
}

/// Output produced by [`fft_auto_dispatch`].
#[derive(Debug)]
pub struct DispatchFftOutput {
    /// Complex-valued FFT result (length equals the padded power-of-two
    /// input size when zero-padding was applied).
    pub data: Vec<Complex64>,

    /// `true` if the computation was offloaded to a GPU back-end;
    /// `false` means the CPU pipeline was used.
    pub used_gpu: bool,

    /// Number of Cooley-Tukey butterfly stages (= log₂(n)).
    pub n_stages: u32,
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Return the smallest power of two that is ≥ `n`.
fn next_power_of_two(n: usize) -> usize {
    if n.is_power_of_two() {
        n
    } else {
        1usize << (usize::BITS - n.leading_zeros()) as usize
    }
}

/// Map a [`GpuFftError`] to an [`FFTError`].
fn gpu_err_to_fft(e: GpuFftError) -> FFTError {
    FFTError::BackendError(e.to_string())
}

/// Build a default [`GpuFftPipeline`] without extra normalisation.
///
/// Note: [`cooley_tukey_gpu`] already applies `1/N` scaling for the inverse
/// direction internally, so no additional normalisation mode should be set
/// here — using `NormalizationMode::Backward` would double-scale.
fn build_pipeline() -> GpuFftPipeline {
    GpuFftPipeline::new(GpuFftConfig {
        normalization: NormalizationMode::None,
        ..GpuFftConfig::default()
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// fft_auto_dispatch
// ─────────────────────────────────────────────────────────────────────────────

/// Compute an FFT with automatic CPU/GPU dispatch.
///
/// # Behaviour
///
/// 1. **Zero-pad** `input` to the next power of two when its length is not
///    already a power of two.  The `data` field in the returned
///    [`DispatchFftOutput`] has this padded length.
/// 2. **Route to GPU** when the `wgpu_fft` feature is enabled, the padded
///    length ≥ `config.gpu_threshold`, and a wgpu adapter is available.
///    If the adapter is unavailable the call falls through to the CPU path.
/// 3. **CPU path**: the existing [`GpuFftPipeline`] is used (pure Rust,
///    always available).
///
/// # Errors
///
/// Returns [`FFTError`] if the pipeline or any kernel call fails.
pub fn fft_auto_dispatch(
    input: &[Complex64],
    config: &AutoDispatchConfig,
) -> Result<DispatchFftOutput, FFTError> {
    let n_padded = next_power_of_two(input.len().max(2));
    let n_stages = n_padded.trailing_zeros();
    let direction = if config.inverse {
        FftDirection::Inverse
    } else {
        FftDirection::Forward
    };

    // Zero-pad into the working buffer.
    let mut buf = Vec::with_capacity(n_padded);
    buf.extend_from_slice(input);
    buf.resize(n_padded, Complex64::new(0.0, 0.0));

    // Try the wgpu path first (compile-time + runtime guarded).
    #[cfg(feature = "wgpu_fft")]
    {
        if n_padded >= config.gpu_threshold {
            match super::wgpu_backend::fft_wgpu(&buf, config.inverse) {
                Ok(result) => {
                    return Ok(DispatchFftOutput {
                        data: result,
                        used_gpu: true,
                        n_stages,
                    });
                }
                Err(_) => {
                    // GPU unavailable at runtime — fall through to CPU.
                }
            }
        }
    }

    // CPU path — always available.
    let pipeline = build_pipeline();
    pipeline
        .execute(&mut buf, n_padded, direction)
        .map_err(gpu_err_to_fft)?;

    Ok(DispatchFftOutput {
        data: buf,
        used_gpu: false,
        n_stages,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// fft_batch_gpu
// ─────────────────────────────────────────────────────────────────────────────

/// GPU batch FFT: compute many same-size transforms efficiently.
///
/// All input slices **must have the same length**; if they differ the
/// shortest common power-of-two is used as the padded size (each input is
/// zero-padded individually).
///
/// Returns one output spectrum per input slice.  The spectra have length
/// equal to the padded size (next power of two of the longest input).
///
/// # Errors
///
/// * [`FFTError::ValueError`] – if `inputs` is empty.
/// * [`FFTError::BackendError`] – if any pipeline call fails.
pub fn fft_batch_gpu(inputs: &[Vec<Complex64>]) -> Result<Vec<Vec<Complex64>>, FFTError> {
    if inputs.is_empty() {
        return Err(FFTError::ValueError(
            "batch input must contain at least one signal".into(),
        ));
    }

    let max_len = inputs.iter().map(|v| v.len()).max().unwrap_or(0);
    let n_padded = next_power_of_two(max_len.max(2));

    // Build complex batches, padding as needed.
    let mut batch: Vec<Vec<Complex64>> = inputs
        .iter()
        .map(|signal| {
            let mut buf = Vec::with_capacity(n_padded);
            buf.extend_from_slice(signal);
            buf.resize(n_padded, Complex64::new(0.0, 0.0));
            buf
        })
        .collect();

    let pipeline = build_pipeline();
    let result = pipeline
        .execute_batch(&mut batch, FftDirection::Forward)
        .map_err(gpu_err_to_fft)?;

    Ok(result.outputs)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const EPS: f64 = 1e-7;

    // ── Default config has threshold 4096 ────────────────────────────────────

    #[test]
    fn gpu_fft_config_default_threshold_4096() {
        let cfg = AutoDispatchConfig::default();
        assert_eq!(cfg.gpu_threshold, 4096);
        assert!(!cfg.inverse);
    }

    // ── CPU path is taken for small inputs ───────────────────────────────────

    #[test]
    fn gpu_fft_auto_dispatch_cpu_path_correct() {
        // 8-point impulse: FFT should be all-ones.
        let input: Vec<Complex64> = {
            let mut v = vec![Complex64::new(0.0, 0.0); 8];
            v[0] = Complex64::new(1.0, 0.0);
            v
        };

        let config = AutoDispatchConfig {
            gpu_threshold: 4096, // 8 << 4096 → CPU path
            inverse: false,
        };

        let out = fft_auto_dispatch(&input, &config).expect("dispatch failed");
        assert!(!out.used_gpu, "small input must use CPU");
        assert_eq!(out.n_stages, 3); // log2(8) = 3

        for (k, c) in out.data.iter().enumerate() {
            assert!(
                (c.re - 1.0).abs() < EPS,
                "bin {k} re = {} (expected 1.0)",
                c.re
            );
            assert!(c.im.abs() < EPS, "bin {k} im = {} (expected 0.0)", c.im);
        }
    }

    // ── Non-power-of-two gets padded to next power of two ────────────────────

    #[test]
    fn fft_power_of_two_padding_correct() {
        // 6-element input → padded to 8.
        let input: Vec<Complex64> = (0..6).map(|i| Complex64::new(i as f64, 0.0)).collect();
        let config = AutoDispatchConfig::default();
        let out = fft_auto_dispatch(&input, &config).expect("dispatch failed");
        assert_eq!(out.data.len(), 8, "padded length must be 8");
    }

    // ── Forward then inverse gives back the original ─────────────────────────

    #[test]
    fn gpu_fft_auto_dispatch_roundtrip() {
        let n = 16;
        let original: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((i as f64 * PI / 8.0).sin(), 0.0))
            .collect();

        let config_fwd = AutoDispatchConfig {
            gpu_threshold: 4096,
            inverse: false,
        };
        let config_inv = AutoDispatchConfig {
            gpu_threshold: 4096,
            inverse: true,
        };

        let forward = fft_auto_dispatch(&original, &config_fwd).expect("forward");
        let recovered = fft_auto_dispatch(&forward.data, &config_inv).expect("inverse");

        // After IFFT the pipeline applies 1/N normalisation (NormalizationMode::Backward).
        for (i, (orig, rec)) in original.iter().zip(recovered.data.iter()).enumerate() {
            assert!(
                (orig.re - rec.re).abs() < 1e-6,
                "index {i}: {:.6} vs {:.6}",
                orig.re,
                rec.re
            );
        }
    }

    // ── Batch results match individual transforms ────────────────────────────

    #[test]
    fn gpu_fft_batch_results_match_individual() {
        let n = 16;
        let signals: Vec<Vec<Complex64>> = (0..8_u64)
            .map(|k| {
                (0..n)
                    .map(|i| Complex64::new(i as f64 + k as f64, 0.0))
                    .collect()
            })
            .collect();

        let config = AutoDispatchConfig::default();

        // Individual
        let individual: Vec<Vec<Complex64>> = signals
            .iter()
            .map(|s| fft_auto_dispatch(s, &config).expect("individual").data)
            .collect();

        // Batch
        let batch = fft_batch_gpu(&signals).expect("batch");

        assert_eq!(batch.len(), signals.len());
        for (sig_idx, (ind, bat)) in individual.iter().zip(batch.iter()).enumerate() {
            assert_eq!(ind.len(), bat.len(), "signal {sig_idx} length mismatch");
            for (bin, (a, b)) in ind.iter().zip(bat.iter()).enumerate() {
                assert!(
                    (a.re - b.re).abs() < 1e-6,
                    "signal {sig_idx} bin {bin} re: {:.8} vs {:.8}",
                    a.re,
                    b.re
                );
                assert!(
                    (a.im - b.im).abs() < 1e-6,
                    "signal {sig_idx} bin {bin} im: {:.8} vs {:.8}",
                    a.im,
                    b.im
                );
            }
        }
    }

    // ── Batch rejects empty input ────────────────────────────────────────────

    #[test]
    fn gpu_fft_batch_rejects_empty() {
        let result = fft_batch_gpu(&[]);
        assert!(result.is_err());
    }
}
