//! GPU auto-dispatch for batch evaluation of special functions.
//! Falls back to CPU when GPU is unavailable or array is small.
//!
//! The dispatch logic is intentionally simple: a minimum array size threshold
//! controls whether to attempt GPU execution. When `allow_gpu` is false (the
//! default), all evaluation is performed on CPU regardless of array size.
//!
//! # Example
//!
//! ```rust
//! use scirs2_special::gpu_dispatch::{GpuDispatchConfig, batch_gamma, batch_erf};
//!
//! let xs = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
//! let config = GpuDispatchConfig::default();
//! let results = batch_gamma(&xs, &config);
//! // Γ(1)=1, Γ(2)=1, Γ(3)=2, Γ(4)=6, Γ(5)=24
//! assert!((results[4] - 24.0).abs() < 1e-10);
//! ```

/// Configuration for GPU dispatch.
#[derive(Debug, Clone)]
pub struct GpuDispatchConfig {
    /// Minimum array size to trigger GPU execution.
    pub min_gpu_size: usize,
    /// Use GPU if available; always use CPU if false.
    pub allow_gpu: bool,
}

impl Default for GpuDispatchConfig {
    fn default() -> Self {
        Self {
            min_gpu_size: 1024,
            allow_gpu: false,
        }
    }
}

impl GpuDispatchConfig {
    /// Create a config that always uses CPU regardless of array size.
    pub fn cpu_only() -> Self {
        Self {
            min_gpu_size: usize::MAX,
            allow_gpu: false,
        }
    }

    /// Create a config that allows GPU dispatch at the given threshold.
    pub fn gpu_at(min_size: usize) -> Self {
        Self {
            min_gpu_size: min_size,
            allow_gpu: true,
        }
    }
}

/// Result of dispatch decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchTarget {
    Cpu,
    Gpu,
}

/// Decide whether to dispatch to GPU based on array size.
///
/// Returns `DispatchTarget::Gpu` only when `config.allow_gpu` is true
/// and `n >= config.min_gpu_size`.
pub fn select_dispatch(n: usize, config: &GpuDispatchConfig) -> DispatchTarget {
    if config.allow_gpu && n >= config.min_gpu_size {
        DispatchTarget::Gpu
    } else {
        DispatchTarget::Cpu
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CPU implementations delegate to the existing crate functions
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn gamma_cpu(x: f64) -> f64 {
    crate::gamma::gamma(x)
}

#[inline]
fn erf_cpu(x: f64) -> f64 {
    crate::erf::erf(x)
}

#[inline]
fn bessel_j0_cpu(x: f64) -> f64 {
    crate::bessel::j0(x)
}

// ─────────────────────────────────────────────────────────────────────────────
// Public batch APIs
// ─────────────────────────────────────────────────────────────────────────────

/// Batch evaluate gamma function with auto-dispatch.
///
/// When `config.allow_gpu` is false (the default), all computation is on CPU.
/// When `allow_gpu` is true and the array exceeds `min_gpu_size`, the GPU
/// path is attempted via the WGSL WebGPU backend
/// ([`crate::gpu_kernels::wgsl::gamma_batch_wgpu`]) and, if that returns
/// `GpuNotAvailable`, via the CUDA backend
/// ([`crate::gpu_kernels::cuda::gamma_batch_cuda`]).  If neither backend is
/// available the function falls back to the CPU path silently.
pub fn batch_gamma(xs: &[f64], config: &GpuDispatchConfig) -> Vec<f64> {
    match select_dispatch(xs.len(), config) {
        DispatchTarget::Cpu => xs.iter().map(|&x| gamma_cpu(x)).collect(),
        DispatchTarget::Gpu => {
            // Try WGSL (WebGPU) first, then CUDA, then fall back to CPU.
            if let Ok(result) = crate::gpu_kernels::wgsl::gamma_batch_wgpu(xs) {
                return result;
            }
            if let Ok(result) = crate::gpu_kernels::cuda::gamma_batch_cuda(xs) {
                return result;
            }
            xs.iter().map(|&x| gamma_cpu(x)).collect()
        }
    }
}

/// Batch evaluate erf function with auto-dispatch.
///
/// GPU path attempts WGSL then CUDA before falling back to CPU.
pub fn batch_erf(xs: &[f64], config: &GpuDispatchConfig) -> Vec<f64> {
    match select_dispatch(xs.len(), config) {
        DispatchTarget::Cpu => xs.iter().map(|&x| erf_cpu(x)).collect(),
        DispatchTarget::Gpu => {
            if let Ok(result) = crate::gpu_kernels::wgsl::erf_batch_wgpu(xs) {
                return result;
            }
            if let Ok(result) = crate::gpu_kernels::cuda::erf_batch_cuda(xs) {
                return result;
            }
            xs.iter().map(|&x| erf_cpu(x)).collect()
        }
    }
}

/// Batch evaluate Bessel J₀ with auto-dispatch.
///
/// GPU path attempts WGSL then CUDA before falling back to CPU.
pub fn batch_bessel_j0(xs: &[f64], config: &GpuDispatchConfig) -> Vec<f64> {
    match select_dispatch(xs.len(), config) {
        DispatchTarget::Cpu => xs.iter().map(|&x| bessel_j0_cpu(x)).collect(),
        DispatchTarget::Gpu => {
            if let Ok(result) = crate::gpu_kernels::wgsl::bessel_j0_batch_wgpu(xs) {
                return result;
            }
            if let Ok(result) = crate::gpu_kernels::cuda::bessel_j0_batch_cuda(xs) {
                return result;
            }
            xs.iter().map(|&x| bessel_j0_cpu(x)).collect()
        }
    }
}

/// Batch evaluate with a custom function and auto-dispatch.
///
/// The function `f` is always called on CPU; the `config` controls whether
/// a GPU-accelerated path would be preferred for built-in functions.  This
/// generic variant always runs on CPU because user functions cannot be
/// dispatched to GPU without additional codegen infrastructure.
pub fn batch_eval<F>(xs: &[f64], f: F, config: &GpuDispatchConfig) -> Vec<f64>
where
    F: Fn(f64) -> f64,
{
    // User-provided functions always run on CPU; dispatch info is recorded but unused.
    let _target = select_dispatch(xs.len(), config);
    xs.iter().map(|&x| f(x)).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_gamma_cpu() {
        let xs = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let config = GpuDispatchConfig::default();
        let results = batch_gamma(&xs, &config);
        // Γ(n) = (n-1)!
        let expected = [1.0, 1.0, 2.0, 6.0, 24.0];
        assert_eq!(results.len(), expected.len());
        for (r, e) in results.iter().zip(expected.iter()) {
            assert!(
                (r - e).abs() < 1e-10,
                "batch_gamma mismatch: got {r}, expected {e}"
            );
        }
    }

    #[test]
    fn test_dispatch_small_array() {
        // Array size 10 with default config (allow_gpu=false) → always CPU
        let config = GpuDispatchConfig::default();
        assert_eq!(select_dispatch(10, &config), DispatchTarget::Cpu);
    }

    #[test]
    fn test_dispatch_large_array_cpu() {
        // allow_gpu=false, size 10000 → still CPU
        let config = GpuDispatchConfig {
            min_gpu_size: 1024,
            allow_gpu: false,
        };
        assert_eq!(select_dispatch(10_000, &config), DispatchTarget::Cpu);
    }

    #[test]
    fn test_dispatch_large_array_gpu_enabled() {
        // allow_gpu=true, size 10000 → GPU (when threshold is 1024)
        let config = GpuDispatchConfig {
            min_gpu_size: 1024,
            allow_gpu: true,
        };
        assert_eq!(select_dispatch(10_000, &config), DispatchTarget::Gpu);
    }

    #[test]
    fn test_dispatch_exactly_at_threshold() {
        let config = GpuDispatchConfig {
            min_gpu_size: 1024,
            allow_gpu: true,
        };
        assert_eq!(select_dispatch(1024, &config), DispatchTarget::Gpu);
        assert_eq!(select_dispatch(1023, &config), DispatchTarget::Cpu);
    }

    #[test]
    fn test_batch_erf() {
        let xs = vec![0.0_f64, 1.0, -1.0, 2.0];
        let config = GpuDispatchConfig::default();
        let results = batch_erf(&xs, &config);
        assert_eq!(results.len(), 4);
        // erf(0) = 0
        assert!(results[0].abs() < 1e-15);
        // erf(1) ≈ 0.8427007929497148
        // The crate implementation uses A&S 7.1.26 with max error 1.5e-7.
        assert!(
            (results[1] - 0.842_700_792_949_715).abs() < 2e-7,
            "erf(1.0) got {:.10}, expected ~0.842700793",
            results[1]
        );
        // erf is odd
        assert!(
            (results[2] + results[1]).abs() < 1e-12,
            "erf should be odd: erf(-1)+erf(1)={}",
            results[2] + results[1]
        );
        // erf(2) ≈ 0.9953222650189527
        assert!(
            (results[3] - 0.995_322_265_019).abs() < 2e-7,
            "erf(2.0) got {:.10}, expected ~0.995322265",
            results[3]
        );
    }

    #[test]
    fn test_batch_eval_custom() {
        // Custom f(x) = x^2
        let xs: Vec<f64> = (1..=5).map(|i| i as f64).collect();
        let config = GpuDispatchConfig::default();
        let results = batch_eval(&xs, |x| x * x, &config);
        let expected: Vec<f64> = xs.iter().map(|&x| x * x).collect();
        assert_eq!(results, expected);
    }

    #[test]
    fn test_batch_bessel_j0() {
        let xs = vec![0.0_f64, 1.0, 2.0];
        let config = GpuDispatchConfig::default();
        let results = batch_bessel_j0(&xs, &config);
        assert_eq!(results.len(), 3);
        // J₀(0) = 1
        assert!((results[0] - 1.0).abs() < 1e-12);
        // J₀(1) ≈ 0.7651976866
        assert!((results[1] - 0.765_197_686_6).abs() < 1e-8);
    }

    #[test]
    fn test_batch_gamma_empty() {
        let xs: Vec<f64> = vec![];
        let config = GpuDispatchConfig::default();
        let results = batch_gamma(&xs, &config);
        assert!(results.is_empty());
    }
}
