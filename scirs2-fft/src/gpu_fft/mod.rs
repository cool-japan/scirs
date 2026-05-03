//! GPU-accelerated FFT pipeline for large-scale signal processing.
//!
//! This module provides a pure-Rust simulation of a tile-based GPU FFT
//! pipeline.  The API surface mirrors what a real CUDA/ROCm back-end would
//! expose, so upgrading to hardware acceleration requires only a back-end
//! swap — no caller changes are needed.
//!
//! # Quick start
//!
//! ```rust
//! use scirs2_fft::gpu_fft::{GpuFftPipeline, GpuFftConfig, FftDirection};
//!
//! let pipeline = GpuFftPipeline::new(GpuFftConfig::default());
//!
//! // Real-to-complex forward FFT.
//! let signal: Vec<f64> = (0..8).map(|i| i as f64).collect();
//! let spectrum = pipeline.execute_r2c(&signal).expect("r2c FFT");
//! assert_eq!(spectrum.len(), 8);
//!
//! // Complex-to-real inverse FFT (roundtrip).
//! let recovered = pipeline.execute_c2r(&spectrum, signal.len()).expect("c2r FFT");
//! for (orig, rec) in signal.iter().zip(recovered.iter()) {
//!     assert!((orig - rec).abs() < 1e-7);
//! }
//! ```
//!
//! # Modules
//!
//! | Sub-module | Contents |
//! |------------|----------|
//! | `types`       | Configuration (`GpuFftConfig`), error (`GpuFftError`), plan (`GpuFftPlan`), result structs |
//! | `kernels`     | Simulated GPU kernels: twiddles, bit-reversal, butterfly, Cooley-Tukey, Bluestein, tiling, normalisation |
//! | `pipeline`    | `GpuFftPipeline` — plan cache, single/batch/R2C/C2R/signal execution |
//! | `dispatch`    | `fft_auto_dispatch`, `fft_batch_gpu` — CPU/GPU auto-dispatch layer |
//! | `overlap_save`| `overlap_save_gpu` — FFT-based overlap-save (OLS) convolution |
//! | `wgpu_backend`| wgpu compute shader back-end (feature `wgpu_fft`) |

pub mod dispatch;
pub mod kernels;
pub mod overlap_save;
pub mod pipeline;
pub mod types;
pub mod wgpu_backend;

// Flat re-exports for ergonomic usage.
pub use dispatch::{fft_auto_dispatch, fft_batch_gpu, AutoDispatchConfig, DispatchFftOutput};
pub use kernels::{
    apply_normalization, bit_reverse_permute_gpu, bluestein_gpu, butterfly_pass_gpu,
    compute_inverse_twiddles_gpu, compute_twiddles_gpu, cooley_tukey_gpu, tiled_fft_1d,
};
pub use overlap_save::overlap_save_gpu;
pub use pipeline::GpuFftPipeline;
pub use types::{
    BatchFftResult, FftDirection, GpuFftConfig, GpuFftError, GpuFftPlan, GpuFftResult,
    NormalizationMode,
};
