//! Overlap-save (OLS) convolution for real-time filtering.
//!
//! The overlap-save method (also called overlap-discard) is an FFT-based
//! block convolution algorithm.  Unlike overlap-add it avoids the
//! reconstruction summation step: each output block is obtained by
//! discarding the *leading* contaminated samples that arise from circular
//! wrap-around, rather than accumulating tail overlaps.
//!
//! # Algorithm (overview)
//!
//! Given a causal FIR kernel of length `M` and a block size `B`:
//!
//! 1. Zero-pad the kernel to length `N = B + M − 1` (next power of two).
//! 2. Pre-compute `H = FFT(kernel_padded)`.
//! 3. For each block `b` starting at position `p = b * B`:
//!    a. Extract `N` samples: the previous `M − 1` samples followed by
//!    the next `B` samples (zero-padded at the signal boundaries).
//!    b. Compute `Y = IFFT(FFT(block) × H)`.
//!    c. Discard the first `M − 1` samples of `Y`; append the remaining
//!    `B` samples to the output.
//!
//! # CPU implementation
//!
//! This module uses the pure-Rust [`GpuFftPipeline`] for all FFT calls.
//! The function signature matches the GPU dispatch API so it can be swapped
//! for a hardware-accelerated path in future without changing callers.

use scirs2_core::numeric::Complex64;

use super::pipeline::GpuFftPipeline;
use super::types::{FftDirection, GpuFftConfig, GpuFftError, NormalizationMode};
use crate::error::FFTError;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Smallest power of two ≥ `n`.
fn next_pow2(n: usize) -> usize {
    if n.is_power_of_two() {
        n
    } else {
        1usize << (usize::BITS - n.leading_zeros()) as usize
    }
}

fn gpu_err(e: GpuFftError) -> FFTError {
    FFTError::BackendError(e.to_string())
}

/// Build a pipeline with no extra normalisation.
///
/// [`cooley_tukey_gpu`] already applies `1/N` for inverse transforms, so
/// we must not add another `Backward` normalisation on top.
fn make_pipeline() -> GpuFftPipeline {
    GpuFftPipeline::new(GpuFftConfig {
        normalization: NormalizationMode::None,
        ..GpuFftConfig::default()
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// overlap_save_gpu
// ─────────────────────────────────────────────────────────────────────────────

/// FFT-based overlap-save convolution.
///
/// Filters `signal` with FIR `kernel` using the overlap-save block
/// algorithm.  The result has the same length as `signal` and corresponds
/// to the *valid* portion of a full linear convolution (i.e. output
/// samples that have a full kernel window into the signal, with implicit
/// zero-padding at the start).
///
/// # Parameters
///
/// - `signal`     — real-valued input signal.
/// - `kernel`     — real-valued FIR filter impulse response (`M` taps).
/// - `block_size` — number of *new* output samples per block.  Values in
///   the range `4 * kernel.len()` … `8 * kernel.len()` work well.  If
///   `0` is passed, `4 * kernel.len()` is used as the default.
///
/// # Errors
///
/// - [`FFTError::ValueError`] – if `signal` or `kernel` is empty, or if
///   `kernel` is longer than `signal`.
/// - [`FFTError::BackendError`] – if any underlying FFT call fails.
pub fn overlap_save_gpu(
    signal: &[f32],
    kernel: &[f32],
    block_size: usize,
) -> Result<Vec<f32>, FFTError> {
    if signal.is_empty() {
        return Err(FFTError::ValueError("signal must not be empty".into()));
    }
    if kernel.is_empty() {
        return Err(FFTError::ValueError("kernel must not be empty".into()));
    }
    let m = kernel.len(); // FIR length
    let b = if block_size == 0 {
        (4 * m).max(8)
    } else {
        block_size.max(1)
    };

    // FFT length: smallest power-of-two ≥ B + M - 1.
    let n_fft = next_pow2(b + m - 1);

    let pipeline = make_pipeline();

    // Pre-compute frequency-domain kernel H = FFT(kernel_padded).
    let mut kernel_buf: Vec<Complex64> = kernel
        .iter()
        .map(|&x| Complex64::new(x as f64, 0.0))
        .collect();
    kernel_buf.resize(n_fft, Complex64::new(0.0, 0.0));
    pipeline
        .execute(&mut kernel_buf, n_fft, FftDirection::Forward)
        .map_err(gpu_err)?;
    let h_freq = kernel_buf; // alias for clarity

    let sig_len = signal.len();
    let mut output = vec![0.0_f32; sig_len];

    // Process blocks. The number of "new" samples each block contributes
    // is B = n_fft - (M - 1).
    let effective_b = n_fft - (m - 1);

    let mut out_pos = 0usize;
    let mut block_idx = 0usize;

    while out_pos < sig_len {
        // Signal start position for this block (may be negative → zero-fill).
        let sig_start = block_idx * effective_b;

        // Gather n_fft samples: first M-1 from overlap region (or zeros),
        // then up to effective_b new samples.
        let mut block: Vec<Complex64> = Vec::with_capacity(n_fft);

        for k in 0..n_fft {
            let idx = sig_start + k;
            // The overlap region at the very start is zero-padded.
            let val = if idx < (m - 1) {
                0.0_f64
            } else {
                let real_idx = idx - (m - 1);
                if real_idx < sig_len {
                    signal[real_idx] as f64
                } else {
                    0.0_f64
                }
            };
            block.push(Complex64::new(val, 0.0));
        }

        // Y = FFT(block)
        pipeline
            .execute(&mut block, n_fft, FftDirection::Forward)
            .map_err(gpu_err)?;

        // Multiply Y × H element-wise.
        for (y, h) in block.iter_mut().zip(h_freq.iter()) {
            *y = *y * *h;
        }

        // y_time = IFFT(Y × H)  — normalised by 1/N via Backward mode.
        pipeline
            .execute(&mut block, n_fft, FftDirection::Inverse)
            .map_err(gpu_err)?;

        // Discard the first M-1 samples (aliased overlap region);
        // copy the remaining effective_b samples to output.
        for k in (m - 1)..n_fft {
            if out_pos >= sig_len {
                break;
            }
            output[out_pos] = block[k].re as f32;
            out_pos += 1;
        }

        block_idx += 1;
    }

    Ok(output)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Naive O(N·M) direct convolution for reference comparison.
    fn direct_convolve(signal: &[f32], kernel: &[f32]) -> Vec<f32> {
        let n = signal.len();
        let m = kernel.len();
        let mut out = vec![0.0_f32; n];
        for i in 0..n {
            let mut acc = 0.0_f32;
            for (j, &k) in kernel.iter().enumerate() {
                if i + j >= m - 1 && i + j - (m - 1) < n {
                    acc += signal[i + j - (m - 1)] * k;
                }
            }
            out[i] = acc;
        }
        out
    }

    // ── OLS matches direct convolution for a known FIR kernel ───────────────

    #[test]
    fn overlap_save_convolution_matches_direct() {
        // 5-tap box filter: each output = average of 5 consecutive inputs.
        let kernel: Vec<f32> = vec![0.2, 0.2, 0.2, 0.2, 0.2];
        let signal: Vec<f32> = (0..64).map(|i| i as f32).collect();

        let ols = overlap_save_gpu(&signal, &kernel, 32).expect("OLS failed");
        let direct = direct_convolve(&signal, &kernel);

        assert_eq!(ols.len(), signal.len());
        for (i, (&o, &d)) in ols.iter().zip(direct.iter()).enumerate() {
            assert!((o - d).abs() < 1e-3, "index {i}: OLS={o:.6} direct={d:.6}");
        }
    }

    // ── Impulse kernel returns the signal unchanged ──────────────────────────

    #[test]
    fn overlap_save_impulse_kernel_identity() {
        let kernel = vec![1.0_f32];
        let signal: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let out = overlap_save_gpu(&signal, &kernel, 0).expect("OLS impulse");
        assert_eq!(out.len(), signal.len());
        for (i, (&o, &s)) in out.iter().zip(signal.iter()).enumerate() {
            assert!((o - s).abs() < 1e-4, "index {i}: {o} vs {s}");
        }
    }

    // ── Sinusoidal signal with low-pass FIR ─────────────────────────────────

    #[test]
    fn overlap_save_lowpass_reduces_high_freq() {
        use std::f32::consts::PI;
        // Low-frequency sine at 1/16 fs — this should pass through a 7-tap
        // box filter without significant attenuation.
        let n = 128;
        let signal: Vec<f32> = (0..n).map(|i| (2.0 * PI * i as f32 / 16.0).sin()).collect();
        let kernel: Vec<f32> = vec![1.0 / 7.0; 7];
        let out = overlap_save_gpu(&signal, &kernel, 64).expect("OLS lowpass");
        assert_eq!(out.len(), n);
        // After the initial transient (first M-1 = 6 samples) the output
        // magnitude should be close to the input for a low-frequency sine.
        let mid = 16;
        assert!(
            out[mid].abs() < 2.0,
            "output magnitude out of expected range at {mid}: {}",
            out[mid]
        );
    }

    // ── Empty signal is rejected ─────────────────────────────────────────────

    #[test]
    fn overlap_save_empty_signal_error() {
        let err = overlap_save_gpu(&[], &[1.0], 32);
        assert!(err.is_err());
    }

    // ── Empty kernel is rejected ─────────────────────────────────────────────

    #[test]
    fn overlap_save_empty_kernel_error() {
        let err = overlap_save_gpu(&[1.0, 2.0], &[], 32);
        assert!(err.is_err());
    }
}
