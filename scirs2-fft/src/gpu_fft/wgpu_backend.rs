//! wgpu GPU FFT backend.
//!
//! This module is compiled only when the `wgpu_fft` feature is enabled.
//! It exposes `fft_wgpu`, which attempts to:
//!
//! 1. Acquire a wgpu adapter and device (GPU).
//! 2. Upload the input buffer to the GPU.
//! 3. Execute the Cooley-Tukey radix-2 DIT FFT via a WGSL compute shader
//!    (`fft_shader.wgsl`) for `log2(n)` passes.
//! 4. Read the result back to the CPU.
//!
//! If no GPU adapter is found at runtime (CI, headless server, etc.) the
//! function returns `Err(FftBackendError::NoAdapter)`.  The dispatch layer
//! in [`super::dispatch`] catches that error and falls back to the CPU path,
//! so callers never need to handle the GPU-unavailable case explicitly.
//!
//! # Status
//!
//! The API surface, error type, and runtime-availability check are fully
//! implemented.  The actual wgpu device initialisation, shader compilation,
//! and buffer upload/readback are marked with `TODO` comments where the
//! hardware interaction code must be inserted.  The CPU fallback pipeline
//! used below is fully functional.
//!
//! # Feature gate
//!
//! This entire module is behind `#[cfg(feature = "wgpu_fft")]`.

#[cfg(feature = "wgpu_fft")]
mod inner {
    use super::super::pipeline::GpuFftPipeline;
    use super::super::types::{FftDirection, GpuFftConfig, NormalizationMode};
    use crate::error::FFTError;
    use scirs2_core::numeric::Complex64;
    use wgpu::{Backends, Instance, InstanceDescriptor, PowerPreference, RequestAdapterOptions};

    // ─────────────────────────────────────────────────────────────────────────
    // Error type
    // ─────────────────────────────────────────────────────────────────────────

    /// Errors specific to the wgpu FFT back-end.
    #[derive(Debug, thiserror::Error)]
    pub enum FftBackendError {
        /// No compatible GPU adapter was found on this system.
        #[error("no wgpu adapter available (GPU unavailable or unsupported)")]
        NoAdapter,

        /// The adapter was found but the device could not be created.
        #[error("wgpu device creation failed: {0}")]
        DeviceCreation(String),

        /// A shader compilation error occurred.
        #[error("WGSL shader compilation failed: {0}")]
        ShaderCompilation(String),

        /// A buffer operation (upload/readback) failed.
        #[error("GPU buffer operation failed: {0}")]
        Buffer(String),

        /// The input length is not a power of two (required by the shader).
        #[error("wgpu FFT requires a power-of-two input length; got {0}")]
        NonPowerOfTwo(usize),
    }

    impl From<FftBackendError> for FFTError {
        fn from(e: FftBackendError) -> Self {
            FFTError::BackendError(e.to_string())
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Runtime availability check
    // ─────────────────────────────────────────────────────────────────────────

    /// Returns `true` when a wgpu adapter appears to be available on this
    /// system.  This is a best-effort, synchronous check — it should not be
    /// relied upon in production code without a subsequent `fft_wgpu` call.
    ///
    /// # Implementation note
    ///
    /// Performs a real wgpu adapter enumeration using `pollster::block_on` to
    /// drive the async adapter request synchronously.  Returns `false` on any
    /// headless / CI environment where no GPU adapter is found, so the
    /// dispatch layer can fall back to the CPU path transparently.
    pub fn gpu_available() -> bool {
        let instance_desc = InstanceDescriptor {
            backends: Backends::all(),
            flags: wgpu::InstanceFlags::default(),
            memory_budget_thresholds: Default::default(),
            backend_options: Default::default(),
            display: None,
        };
        let instance = Instance::new(instance_desc);
        pollster::block_on(async {
            instance
                .request_adapter(&RequestAdapterOptions {
                    power_preference: PowerPreference::default(),
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await
                .is_ok()
        })
    }

    // ─────────────────────────────────────────────────────────────────────────
    // fft_wgpu
    // ─────────────────────────────────────────────────────────────────────────

    /// Compute an FFT using the wgpu compute shader pipeline.
    ///
    /// `input` must have a **power-of-two length**.  Use
    /// [`super::dispatch::fft_auto_dispatch`] for automatic padding.
    ///
    /// Returns `Err(FftBackendError::NoAdapter.into())` when no GPU is
    /// available; the dispatch layer uses this to select the CPU path.
    ///
    /// # GPU execution pipeline (TODO)
    ///
    /// Once the wgpu initialisation is filled in, the steps are:
    ///
    /// 1. `wgpu::Instance::default()` → `request_adapter` → `request_device`.
    /// 2. Create a storage buffer for the complex data (`array<vec2<f32>>`).
    /// 3. Create a uniform buffer for `FFTParams { n, stage, inverse, _pad }`.
    /// 4. Load `fft_shader.wgsl` (include via `include_str!`), compile the
    ///    compute pipeline.
    /// 5. Bit-reverse permute the input on the CPU, upload to the GPU buffer.
    /// 6. For each `stage` in `0..log2(n)`:
    ///    a. Update the uniform buffer with the current stage index.
    ///    b. Dispatch `n/2 / 64` workgroups (`@workgroup_size(64)`).
    ///    c. Insert a pipeline barrier (`queue.submit`).
    /// 7. Map the output buffer back to the CPU, read the `vec2<f32>` pairs
    ///    as `Complex32`, convert to `Complex64`, return.
    /// 8. If `inverse`, scale each sample by `1/n`.
    pub fn fft_wgpu(input: &[Complex64], _inverse: bool) -> Result<Vec<Complex64>, FFTError> {
        let n = input.len();
        if !n.is_power_of_two() {
            return Err(FftBackendError::NonPowerOfTwo(n).into());
        }

        // ── TODO: Real wgpu initialisation ──────────────────────────────────
        // Replace the block below with an actual wgpu device + pipeline setup
        // following the GPU execution pipeline steps documented above.
        // ────────────────────────────────────────────────────────────────────

        // For now always signal "no adapter" → dispatch falls back to CPU.
        let adapter_available = gpu_available();
        if !adapter_available {
            return Err(FftBackendError::NoAdapter.into());
        }

        // ── Placeholder GPU path ────────────────────────────────────────────
        // This code is unreachable until `gpu_available()` returns `true`.
        // When the wgpu initialisation is filled in above, remove the
        // `NoAdapter` early return and implement the buffer operations here.
        let direction = if _inverse {
            FftDirection::Inverse
        } else {
            FftDirection::Forward
        };
        let norm = if _inverse {
            NormalizationMode::Backward
        } else {
            NormalizationMode::None
        };
        let pipeline = GpuFftPipeline::new(GpuFftConfig {
            normalization: norm,
            ..GpuFftConfig::default()
        });
        let mut buf = input.to_vec();
        pipeline
            .execute(&mut buf, n, direction)
            .map_err(|e| FFTError::BackendError(e.to_string()))?;
        Ok(buf)
    }
}

// Re-export the public items when the feature is active.
#[cfg(feature = "wgpu_fft")]
pub use inner::{fft_wgpu, gpu_available, FftBackendError};

#[cfg(all(test, feature = "wgpu_fft"))]
mod tests {
    use super::gpu_available;

    /// Verify that `gpu_available()` completes without panicking and returns a
    /// valid boolean.  The actual value (`true` or `false`) is environment-
    /// dependent: CI / headless machines will return `false`, real GPU hosts
    /// may return `true`.  We only assert that the call completes.
    #[test]
    fn test_gpu_available_returns_bool() {
        let result: bool = gpu_available();
        // Log the result for diagnostic purposes; never assert the specific value.
        println!("gpu_available() = {result}");
    }
}
