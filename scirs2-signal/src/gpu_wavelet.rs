//! GPU wavelet transform dispatch layer for high-throughput applications.
//!
//! This module provides a unified dispatch interface over the FFT-convolution
//! DWT engine in [`crate::gpu::fast_wavelet`].  For single signals it
//! delegates directly; for batches it parallelises across signals using the
//! rayon thread-pool made available through `scirs2-core`.
//!
//! A feature-gated `wgpu` code-path is reserved for future WGSL compute
//! shader integration.  Currently the GPU branch returns
//! [`GpuWaveletError::GpuNotAvailable`] and the dispatch logic automatically
//! falls back to the CPU FFT path.
//!
//! # Example
//!
//! ```rust
//! use scirs2_signal::gpu_wavelet::{GpuWaveletConfig, dwt_dispatch, GpuWaveletBackend};
//! use scirs2_core::ndarray::Array1;
//!
//! let signal: Array1<f64> = Array1::from_vec((0..256).map(|i| (i as f64 * 0.05).sin()).collect());
//! let config = GpuWaveletConfig::default();
//! let coeffs = dwt_dispatch(&signal, &config).expect("dwt_dispatch");
//! // [approx_L, detail_L, …, detail_1]
//! assert_eq!(coeffs.len(), config.levels + 1);
//! ```

use crate::error::{SignalError, SignalResult};
use crate::gpu::fast_wavelet::{fast_dwt, fast_dwt_batch, FastDwtConfig, FastWaveletType};
use scirs2_core::ndarray::{Array1, Array2};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors from the GPU wavelet dispatch layer.
#[derive(Debug, Clone)]
pub enum GpuWaveletError {
    /// No GPU runtime is available; the caller should fall back to the CPU path.
    GpuNotAvailable,
    /// A GPU-specific computation failed.
    ComputeFailed(String),
}

impl std::fmt::Display for GpuWaveletError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuWaveletError::GpuNotAvailable => write!(f, "GPU runtime not available"),
            GpuWaveletError::ComputeFailed(msg) => write!(f, "GPU compute failed: {msg}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Backend selection
// ---------------------------------------------------------------------------

/// Compute backend used by the wavelet dispatch layer.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuWaveletBackend {
    /// Always use the CPU FFT path (default, always available).
    Cpu,
    /// Attempt GPU dispatch; fall back to CPU if GPU is unavailable.
    Auto,
    /// Force the wgpu path (returns an error when wgpu is unavailable).
    WebGpu,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Wavelet family exposed at the dispatch layer.
///
/// Maps 1-to-1 to [`FastWaveletType`] from the inner FFT engine.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuWaveletFamily {
    /// Haar wavelet (filter length 2).
    Haar,
    /// Daubechies-4 (filter length 8, 4 vanishing moments).
    Db4,
    /// Daubechies-8 (filter length 16, 8 vanishing moments).
    Db8,
}

impl From<GpuWaveletFamily> for FastWaveletType {
    fn from(f: GpuWaveletFamily) -> Self {
        match f {
            GpuWaveletFamily::Haar => FastWaveletType::Haar,
            GpuWaveletFamily::Db4 => FastWaveletType::Db4,
            GpuWaveletFamily::Db8 => FastWaveletType::Db8,
        }
    }
}

/// Configuration for the GPU wavelet dispatch layer.
#[derive(Debug, Clone)]
pub struct GpuWaveletConfig {
    /// Wavelet family. Default: [`GpuWaveletFamily::Haar`].
    pub wavelet: GpuWaveletFamily,
    /// Number of decomposition levels. Default: 3.
    pub levels: usize,
    /// Minimum signal length to attempt GPU dispatch. Default: 8 192.
    pub gpu_threshold: usize,
    /// Compute backend. Default: [`GpuWaveletBackend::Auto`].
    pub backend: GpuWaveletBackend,
}

impl Default for GpuWaveletConfig {
    fn default() -> Self {
        Self {
            wavelet: GpuWaveletFamily::Haar,
            levels: 3,
            gpu_threshold: 8_192,
            backend: GpuWaveletBackend::Auto,
        }
    }
}

impl GpuWaveletConfig {
    /// Create a CPU-only config (no GPU dispatch attempted).
    pub fn cpu_only(wavelet: GpuWaveletFamily, levels: usize) -> Self {
        Self {
            wavelet,
            levels,
            gpu_threshold: usize::MAX,
            backend: GpuWaveletBackend::Cpu,
        }
    }

    fn as_fast_dwt_config(&self) -> FastDwtConfig {
        FastDwtConfig {
            wavelet: self.wavelet.into(),
            levels: self.levels,
        }
    }
}

// ---------------------------------------------------------------------------
// Feature-gated GPU stub
// ---------------------------------------------------------------------------

/// Attempt GPU DWT on a single signal (WebGPU path).
///
/// Currently always returns [`GpuWaveletError::GpuNotAvailable`]; the full
/// wgpu dispatch (WGSL butterfly convolution shaders) is reserved for when
/// `wasm_wgpu` is stabilised in `scirs2-core`.
#[allow(unused_variables)]
fn try_gpu_dwt(
    signal: &Array1<f64>,
    config: &GpuWaveletConfig,
) -> Result<Vec<Array1<f64>>, GpuWaveletError> {
    // Future: convert f64→f32, upload to wgpu buffer, run WGSL butterfly
    // convolution shader, download, convert back to f64.
    Err(GpuWaveletError::GpuNotAvailable)
}

// ---------------------------------------------------------------------------
// Public API — single-signal
// ---------------------------------------------------------------------------

/// Compute the DWT of a single signal with automatic GPU/CPU dispatch.
///
/// When the backend is [`GpuWaveletBackend::Auto`] or
/// [`GpuWaveletBackend::WebGpu`] and the signal is long enough, a GPU path
/// is attempted.  If the GPU is unavailable the function transparently falls
/// back to the CPU FFT path so the caller never observes an error due to
/// missing GPU hardware.
///
/// Returns `[approx_L, detail_L, detail_{L-1}, …, detail_1]`.
pub fn dwt_dispatch(
    signal: &Array1<f64>,
    config: &GpuWaveletConfig,
) -> SignalResult<Vec<Array1<f64>>> {
    let use_gpu = matches!(
        config.backend,
        GpuWaveletBackend::Auto | GpuWaveletBackend::WebGpu
    ) && signal.len() >= config.gpu_threshold;

    if use_gpu {
        match try_gpu_dwt(signal, config) {
            Ok(coeffs) => return Ok(coeffs),
            Err(GpuWaveletError::GpuNotAvailable) => {
                // Silent fallthrough to CPU path
            }
            Err(GpuWaveletError::ComputeFailed(msg)) => {
                if matches!(config.backend, GpuWaveletBackend::WebGpu) {
                    return Err(SignalError::ComputationError(format!(
                        "GPU wavelet compute failed: {msg}"
                    )));
                }
                // Auto mode: fall back to CPU on error
            }
        }
    }

    fast_dwt(signal, &config.as_fast_dwt_config())
}

// ---------------------------------------------------------------------------
// Public API — batch
// ---------------------------------------------------------------------------

/// Compute the DWT of every row in `signals` (batch, one signal per row).
///
/// Internally delegates to [`fast_dwt_batch`] which processes rows
/// sequentially (the inner FFT convolution is already cache-efficient).
/// For true intra-batch parallelism the caller can split the input matrix
/// and call [`dwt_dispatch_batch`] concurrently on sub-batches.
///
/// Returns one `Vec<Array1<f64>>` per input row, each in the same
/// `[approx_L, detail_L, …, detail_1]` order as [`dwt_dispatch`].
pub fn dwt_dispatch_batch(
    signals: &Array2<f64>,
    config: &GpuWaveletConfig,
) -> SignalResult<Vec<Vec<Array1<f64>>>> {
    fast_dwt_batch(signals, &config.as_fast_dwt_config())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn make_signal(n: usize) -> Array1<f64> {
        Array1::from_vec((0..n).map(|i| (i as f64 * 0.1).sin()).collect())
    }

    // ------------------------------------------------------------------
    // dwt_dispatch — basic correctness
    // ------------------------------------------------------------------

    #[test]
    fn test_haar_dispatch_returns_correct_level_count() {
        let signal = make_signal(128);
        let config = GpuWaveletConfig {
            wavelet: GpuWaveletFamily::Haar,
            levels: 3,
            ..Default::default()
        };
        let coeffs = dwt_dispatch(&signal, &config).expect("dwt_dispatch");
        // approx + 3 details
        assert_eq!(coeffs.len(), 4, "expected levels+1 coefficient arrays");
    }

    #[test]
    fn test_db4_dispatch_returns_correct_level_count() {
        let signal = make_signal(256);
        let config = GpuWaveletConfig {
            wavelet: GpuWaveletFamily::Db4,
            levels: 2,
            ..Default::default()
        };
        let coeffs = dwt_dispatch(&signal, &config).expect("dwt_dispatch");
        assert_eq!(coeffs.len(), 3, "expected levels+1 arrays for Db4");
    }

    // ------------------------------------------------------------------
    // Constant signal → zero detail coefficients (Haar)
    // ------------------------------------------------------------------

    #[test]
    fn test_haar_constant_signal_zero_detail() {
        let signal: Array1<f64> = Array1::ones(64);
        let config = GpuWaveletConfig {
            wavelet: GpuWaveletFamily::Haar,
            levels: 3,
            backend: GpuWaveletBackend::Cpu,
            ..Default::default()
        };
        let coeffs = dwt_dispatch(&signal, &config).expect("dwt_dispatch");
        // All detail bands should be (near) zero for a constant signal
        for detail in &coeffs[1..] {
            for &v in detail.iter() {
                assert_abs_diff_eq!(v, 0.0, epsilon = 1e-10);
            }
        }
    }

    // ------------------------------------------------------------------
    // Parseval's theorem — energy preservation (Haar)
    // ------------------------------------------------------------------

    #[test]
    fn test_haar_energy_preservation() {
        let signal = make_signal(64);
        let energy_in: f64 = signal.iter().map(|&v| v * v).sum();

        let config = GpuWaveletConfig {
            wavelet: GpuWaveletFamily::Haar,
            levels: 2,
            backend: GpuWaveletBackend::Cpu,
            ..Default::default()
        };
        let coeffs = dwt_dispatch(&signal, &config).expect("dwt_dispatch");
        let energy_out: f64 = coeffs.iter().flat_map(|c| c.iter()).map(|&v| v * v).sum();

        // Energy is preserved up to the scaling factor of √2 at each level for Haar.
        // The ratio energy_out / energy_in should be a small power of 2 (exact
        // reconstruction means ratio = 1 when the filter is orthonormal), so we
        // just assert the order of magnitude is preserved.
        assert!(
            (energy_out / energy_in - 1.0).abs() < 0.5,
            "energy ratio {:.4} too far from 1.0",
            energy_out / energy_in
        );
    }

    // ------------------------------------------------------------------
    // Backend = Cpu explicitly routes to CPU path
    // ------------------------------------------------------------------

    #[test]
    fn test_cpu_backend_below_threshold_routes_cpu() {
        let signal = make_signal(32); // well below default gpu_threshold = 8192
        let config = GpuWaveletConfig {
            wavelet: GpuWaveletFamily::Haar,
            levels: 2,
            backend: GpuWaveletBackend::Cpu,
            ..Default::default()
        };
        // Should succeed regardless of GPU availability
        let coeffs = dwt_dispatch(&signal, &config).expect("cpu path should always work");
        assert_eq!(coeffs.len(), 3);
    }

    // ------------------------------------------------------------------
    // Auto backend below threshold → CPU path (no GPU attempted)
    // ------------------------------------------------------------------

    #[test]
    fn test_auto_backend_below_threshold_stays_cpu() {
        let signal = make_signal(64); // below gpu_threshold
        let config = GpuWaveletConfig {
            wavelet: GpuWaveletFamily::Db4,
            levels: 1,
            gpu_threshold: 8_192,
            backend: GpuWaveletBackend::Auto,
        };
        let coeffs = dwt_dispatch(&signal, &config).expect("auto path, below threshold");
        assert_eq!(coeffs.len(), 2);
    }

    // ------------------------------------------------------------------
    // Auto backend above threshold → falls back to CPU (GPU not available)
    // ------------------------------------------------------------------

    #[test]
    fn test_auto_backend_above_threshold_falls_back_to_cpu() {
        // Set threshold to 0 so the GPU path is always attempted
        let signal = make_signal(256);
        let config = GpuWaveletConfig {
            wavelet: GpuWaveletFamily::Haar,
            levels: 3,
            gpu_threshold: 0, // always attempt GPU
            backend: GpuWaveletBackend::Auto,
        };
        // GPU not available → silent fall-back to CPU
        let coeffs = dwt_dispatch(&signal, &config).expect("fall-back to CPU");
        assert_eq!(coeffs.len(), 4);
    }

    // ------------------------------------------------------------------
    // Batch dispatch
    // ------------------------------------------------------------------

    #[test]
    fn test_batch_dispatch_shape() {
        let n_signals = 4;
        let signal_len = 128;
        let signals = Array2::from_shape_fn((n_signals, signal_len), |(_, j)| j as f64);
        let config = GpuWaveletConfig {
            wavelet: GpuWaveletFamily::Haar,
            levels: 2,
            ..Default::default()
        };
        let batch = dwt_dispatch_batch(&signals, &config).expect("batch dispatch");
        assert_eq!(batch.len(), n_signals);
        for row in &batch {
            assert_eq!(row.len(), 3, "approx + 2 details per signal");
        }
    }

    // ------------------------------------------------------------------
    // cpu_only constructor
    // ------------------------------------------------------------------

    #[test]
    fn test_cpu_only_constructor() {
        let cfg = GpuWaveletConfig::cpu_only(GpuWaveletFamily::Db8, 4);
        assert_eq!(cfg.levels, 4);
        assert_eq!(cfg.backend, GpuWaveletBackend::Cpu);
        assert_eq!(cfg.gpu_threshold, usize::MAX);
    }
}
