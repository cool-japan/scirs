//! wgpu GPU FFT backend.
//!
//! This module is compiled only when the `wgpu` feature is enabled.
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
//! # Feature gate
//!
//! This entire module is behind `#[cfg(feature = "wgpu")]`.

#[cfg(feature = "wgpu")]
mod inner {
    use crate::error::FFTError;
    use scirs2_core::numeric::Complex64;
    use wgpu::{Backends, Instance, InstanceDescriptor, PowerPreference, RequestAdapterOptions};

    use super::super::kernels::bit_reverse_permute_gpu;

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
    // Internal helpers
    // ─────────────────────────────────────────────────────────────────────────

    /// Encode FFT uniform-buffer params as raw little-endian bytes.
    ///
    /// Layout: `{ n: u32, stage: u32, inverse: u32, _pad: u32 }` (16 bytes).
    fn encode_params(n: u32, stage: u32, inverse: u32) -> [u8; 16] {
        let mut out = [0u8; 16];
        out[0..4].copy_from_slice(&n.to_le_bytes());
        out[4..8].copy_from_slice(&stage.to_le_bytes());
        out[8..12].copy_from_slice(&inverse.to_le_bytes());
        // _pad = 0 (already zero)
        out
    }

    /// Serialise a slice of `Complex64` as `array<vec2<f32>>` bytes.
    ///
    /// Each complex sample becomes two contiguous `f32` values (real then
    /// imaginary), each encoded as 4 little-endian bytes, for a total of 8
    /// bytes per sample.
    fn complex64_to_bytes(data: &[Complex64]) -> Vec<u8> {
        let mut out = Vec::with_capacity(data.len() * 8);
        for c in data {
            out.extend_from_slice(&(c.re as f32).to_le_bytes());
            out.extend_from_slice(&(c.im as f32).to_le_bytes());
        }
        out
    }

    /// Deserialise `array<vec2<f32>>` bytes back to `Vec<Complex64>`.
    fn bytes_to_complex64(bytes: &[u8]) -> Vec<Complex64> {
        bytes
            .chunks_exact(8)
            .map(|chunk| {
                let re = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f64;
                let im = f32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]) as f64;
                Complex64::new(re, im)
            })
            .collect()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // fft_wgpu
    // ─────────────────────────────────────────────────────────────────────────

    /// Compute an FFT (or IFFT) using the wgpu compute shader pipeline.
    ///
    /// `input` must have a **power-of-two length**.  Use
    /// `super::dispatch::fft_auto_dispatch` for automatic padding.
    ///
    /// Returns `Err(FftBackendError::NoAdapter.into())` when no GPU is
    /// available; the dispatch layer uses this to select the CPU path.
    ///
    /// # GPU execution pipeline
    ///
    /// 1. `wgpu::Instance::new` → `request_adapter` → `request_device`.
    /// 2. Bit-reverse permute the input on the CPU.
    /// 3. Upload the complex data to a storage buffer as `array<vec2<f32>>`.
    /// 4. Create a uniform buffer for `FFTParams { n, stage, inverse, _pad }`.
    /// 5. Load `fft_shader.wgsl` via `include_str!`, compile the compute pipeline.
    /// 6. For each `stage` in `0..log2(n)`: update the uniform buffer with
    ///    the current stage index via `queue.write_buffer`, encode one compute
    ///    pass dispatching `ceil(n/2 / 64)` workgroups, submit and poll until
    ///    the GPU is idle before the next stage.
    /// 7. Copy the result buffer to a CPU-mappable staging buffer, map and
    ///    read back the `vec2<f32>` pairs as `Complex64`.
    /// 8. If `inverse`, scale each sample by `1.0 / n`.
    pub fn fft_wgpu(input: &[Complex64], inverse: bool) -> Result<Vec<Complex64>, FFTError> {
        use wgpu::{
            util::{BufferInitDescriptor, DeviceExt as _},
            BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
            BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
            CommandEncoderDescriptor, ComputePassDescriptor, DeviceDescriptor, Features, Limits,
            MapMode, ShaderModuleDescriptor, ShaderSource, ShaderStages,
        };

        let n = input.len();
        if !n.is_power_of_two() {
            return Err(FftBackendError::NonPowerOfTwo(n).into());
        }
        // n == 0 or n == 1 are degenerate: return as-is (trivial FFT).
        if n <= 1 {
            return Ok(input.to_vec());
        }

        let log2_n = n.trailing_zeros();
        let inverse_flag: u32 = if inverse { 1 } else { 0 };
        let byte_len = (n * 8) as u64; // 8 bytes per complex sample (2 × f32)

        // ── Adapter / device acquisition ──────────────────────────────────────
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::all(),
            flags: wgpu::InstanceFlags::default(),
            memory_budget_thresholds: Default::default(),
            backend_options: Default::default(),
            display: None,
        });

        let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .map_err(|_| FFTError::from(FftBackendError::NoAdapter))?;

        let (device, queue) = pollster::block_on(adapter.request_device(&DeviceDescriptor {
            label: Some("scirs2-fft"),
            required_features: Features::empty(),
            required_limits: Limits::default(),
            ..Default::default()
        }))
        .map_err(|e| FFTError::from(FftBackendError::DeviceCreation(e.to_string())))?;

        // ── Bit-reverse permutation on the CPU ────────────────────────────────
        let mut buf = input.to_vec();
        bit_reverse_permute_gpu(&mut buf);

        // ── Data buffer (storage read_write + COPY_SRC for readback) ──────────
        let data_bytes = complex64_to_bytes(&buf);

        let buf_data = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("scirs2-fft-data"),
            contents: &data_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        });

        // ── Uniform buffer for FFTParams (starts at stage 0) ──────────────────
        let initial_params = encode_params(n as u32, 0, inverse_flag);

        let buf_params = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("scirs2-fft-params"),
            contents: &initial_params,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        // ── Staging buffer (CPU readable) ─────────────────────────────────────
        let buf_staging = device.create_buffer(&BufferDescriptor {
            label: Some("scirs2-fft-staging"),
            size: byte_len,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Bind group layout matching the shader bindings ────────────────────
        //    @group(0) @binding(0) var<storage, read_write> data: array<vec2<f32>>;
        //    @group(0) @binding(1) var<uniform> params: FFTParams;
        let bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("scirs2-fft-bgl"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // ── Pipeline layout ───────────────────────────────────────────────────
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scirs2-fft-layout"),
            bind_group_layouts: &[Some(&bgl)],
            ..Default::default()
        });

        // ── Shader module ─────────────────────────────────────────────────────
        let shader_src = include_str!("fft_shader.wgsl");
        let shader_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("scirs2-fft-shader"),
            source: ShaderSource::Wgsl(shader_src.into()),
        });

        // ── Compute pipeline ──────────────────────────────────────────────────
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("scirs2-fft-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // ── Bind group (static — data buffer and params buffer are fixed) ─────
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("scirs2-fft-bg"),
            layout: &bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: buf_data.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: buf_params.as_entire_binding(),
                },
            ],
        });

        // ── Per-stage dispatch loop ───────────────────────────────────────────
        // Dispatch ceil(n/2 / 64) workgroups; the shader uses @workgroup_size(64)
        // and each thread handles exactly one butterfly pair.
        let workgroups = (n / 2).div_ceil(64) as u32;

        for stage in 0..log2_n {
            // Update the uniform buffer with the current stage index.
            let params_bytes = encode_params(n as u32, stage, inverse_flag);
            queue.write_buffer(&buf_params, 0, &params_bytes);

            let mut encoder =
                device.create_command_encoder(&CommandEncoderDescriptor { label: None });
            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
            queue.submit([encoder.finish()]);

            // Wait for the GPU to finish before updating the stage for the next pass.
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|e| {
                    FFTError::from(FftBackendError::Buffer(format!("GPU poll error: {e:?}")))
                })?;
        }

        // ── Copy result from data buffer to staging buffer ────────────────────
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&buf_data, 0, &buf_staging, 0, byte_len);
        queue.submit([encoder.finish()]);

        // ── Map staging buffer and read back ──────────────────────────────────
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| {
                FFTError::from(FftBackendError::Buffer(format!(
                    "GPU poll before map: {e:?}"
                )))
            })?;

        let slice = buf_staging.slice(0..byte_len);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(MapMode::Read, move |r| {
            let _ = tx.send(r);
        });

        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| {
                FFTError::from(FftBackendError::Buffer(format!(
                    "GPU poll during map: {e:?}"
                )))
            })?;

        rx.recv()
            .map_err(|_| {
                FFTError::from(FftBackendError::Buffer(
                    "channel closed during map_async".into(),
                ))
            })?
            .map_err(|e| {
                FFTError::from(FftBackendError::Buffer(format!("map_async failed: {e:?}")))
            })?;

        let mapped = slice.get_mapped_range();
        let mut result = bytes_to_complex64(&mapped);
        drop(mapped);
        buf_staging.unmap();

        // ── Inverse FFT scaling ───────────────────────────────────────────────
        if inverse {
            let scale = 1.0 / n as f64;
            for c in &mut result {
                c.re *= scale;
                c.im *= scale;
            }
        }

        Ok(result)
    }
}

// Re-export the public items when the feature is active.
#[cfg(feature = "wgpu")]
pub use inner::{fft_wgpu, gpu_available, FftBackendError};

#[cfg(all(test, feature = "wgpu"))]
mod tests {
    use super::{fft_wgpu, gpu_available};
    use scirs2_core::numeric::Complex64;

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

    /// An 8-point FFT then IFFT must recover the original input within f32
    /// floating-point tolerance (~0.01).  On headless / CI machines without a
    /// GPU the test is silently skipped.
    #[test]
    fn test_fft_wgpu_roundtrip_or_skip() {
        let input: Vec<Complex64> = (0..8).map(|i| Complex64::new(i as f64, 0.0)).collect();

        match fft_wgpu(&input, false) {
            Err(e)
                if e.to_string().contains("adapter")
                    || e.to_string().contains("NoAdapter")
                    || e.to_string().contains("no wgpu") =>
            {
                println!("test_fft_wgpu_roundtrip_or_skip: skipping — no GPU adapter");
            }
            Err(e) => panic!("unexpected fft_wgpu error: {e}"),
            Ok(spectrum) => {
                assert_eq!(spectrum.len(), input.len());
                // IFFT to recover
                match fft_wgpu(&spectrum, true) {
                    Err(e) => panic!("unexpected ifft_wgpu error: {e}"),
                    Ok(recovered) => {
                        for (orig, rec) in input.iter().zip(recovered.iter()) {
                            assert!(
                                (orig.re - rec.re).abs() < 0.01,
                                "re mismatch: {} vs {}",
                                orig.re,
                                rec.re
                            );
                            assert!(
                                (orig.im - rec.im).abs() < 0.01,
                                "im mismatch: {} vs {}",
                                orig.im,
                                rec.im
                            );
                        }
                    }
                }
            }
        }
    }

    /// Non-power-of-two input must always be rejected immediately, regardless
    /// of whether a GPU adapter is available.
    #[test]
    fn test_fft_wgpu_non_power_of_two_rejected() {
        let input: Vec<Complex64> = vec![Complex64::new(1.0, 0.0); 7];
        let result = fft_wgpu(&input, false);
        assert!(
            result.is_err(),
            "non-power-of-two input must return an error"
        );
    }
}
