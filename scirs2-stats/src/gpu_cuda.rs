#![cfg(feature = "cuda")]
//! Optional, off-by-default, NVIDIA-only pure-Rust CUDA path for batched
//! standard-normal PDF and CDF, built directly on the `oxicuda` ecosystem.
//!
//! This module is an opt-in CUDA acceleration path for the standard-normal
//! probability density and cumulative distribution functions over a batch of
//! `f64` inputs. It is built entirely on the pure-Rust `oxicuda` crates
//! (`oxicuda-ptx`, `oxicuda-launch`, `oxicuda-driver`, `oxicuda-memory`) — and
//! specifically on `oxicuda-ptx`'s `exp_f64` and `erf_f64` transcendentals —
//! compiled only when the crate-local `cuda` feature is enabled, and it does
//! **not** route through `scirs2-core`. The existing CPU and `wgpu` paths are
//! left completely untouched — this is purely additive.
//!
//! ## Availability (runtime-probed, NVIDIA-only)
//!
//! `oxicuda-driver` loads `libcuda` at runtime. On a machine with no NVIDIA
//! driver (for example this development Mac), initialization fails and the
//! public functions return a [`StatsError`] variant rather than panicking.
//! Call [`cuda_is_available`] first to probe for a usable device; it never
//! panics and returns `false` when CUDA is unavailable (for example on
//! non-NVIDIA hosts such as this development Mac). Importantly, the crate still
//! *compiles* on macOS with `--features cuda` — only the runtime probe returns
//! `false`.
//!
//! ## The CUSTOM-KERNEL pattern
//!
//! This module **generates bespoke `normal_pdf_batch` and `normal_cdf_batch`
//! PTX kernels** using `oxicuda-ptx`'s instruction-level DSL — specifically the
//! [`exp_f64`](oxicuda_ptx::builder::BodyBuilder::exp_f64) and
//! [`erf_f64`](oxicuda_ptx::builder::BodyBuilder::erf_f64) transcendentals
//! (the latter from Abramowitz & Stegun 7.1.26, ~1.5e-7 absolute accuracy) —
//! then dispatches them with `oxicuda-launch`. The PTX is produced by
//! [`generate_normal_pdf_ptx`] and [`generate_normal_cdf_ptx`], which are fully
//! **GPU-free** code generators: they can be inspected and unit-tested on any
//! platform without a CUDA device present. Only the actual device launch
//! (inside [`cuda_normal_pdf_batch`] / [`cuda_normal_cdf_batch`]) requires
//! NVIDIA hardware.
//!
//! ## f64-native — a genuine precision advantage over the wgpu path
//!
//! The crate's `wgpu` paths downcast host `f64` inputs to `f32`, because WGSL's
//! portable storage type is `f32`. This CUDA path is **`f64`-native end to
//! end**: inputs are uploaded as `f64`, every arithmetic instruction is the
//! double-precision PTX op, and the result is read back as `f64`. The CDF's A&S
//! polynomial caps mathematical accuracy at ~1.5e-7, but the arithmetic carries
//! full `f64` precision (no `f32` bridge) and no `.approx.f64` instruction is
//! emitted.
//!
//! ## Kernel design — one thread per element
//!
//! The generated `normal_pdf_batch(out_ptr, n, in_ptr)` and
//! `normal_cdf_batch(out_ptr, n, in_ptr)` kernels each have a thread `tid` load
//! `in_ptr[tid]` (an `f64` at byte offset `tid*8`), evaluate the closed-form
//! distribution value, and store the result to `out_ptr[tid]`. A `tid < n`
//! guard makes the launch safe for any grid size.

use oxicuda_driver::Module;
use oxicuda_launch::{grid_size_for, Kernel, LaunchParams};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::builder::KernelBuilder;
use oxicuda_ptx::ir::PtxType;
use std::sync::Arc;

use crate::error::{StatsError, StatsResult};

/// Probe whether a usable NVIDIA CUDA device is available at runtime.
///
/// Never panics. Returns `false` when the CUDA driver cannot be initialized
/// (for example on non-NVIDIA platforms such as macOS) or when no device is
/// present. Call this before [`cuda_normal_pdf_batch`] to decide whether the
/// CUDA path is usable on the current host.
pub fn cuda_is_available() -> bool {
    oxicuda_driver::init().is_ok()
        && oxicuda_driver::device::Device::count()
            .map(|c| c > 0)
            .unwrap_or(false)
}

/// Generate the PTX text for the `normal_pdf_batch` kernel.
///
/// This is a fully **GPU-free** code generator: it builds and returns the PTX
/// string without touching any CUDA device, so it can be inspected and tested
/// on any platform. The kernel takes `(out_ptr: u64, n: u32, in_ptr: u64)` and
/// computes `out_ptr[tid] = exp(-0.5 * x^2) * INV_SQRT_2PI` for `tid < n`,
/// where `x = in_ptr[tid]` is an `f64` at byte offset `tid*8`. The exponential
/// evaluation uses [`BodyBuilder::exp_f64`](oxicuda_ptx::builder::BodyBuilder::exp_f64)
/// and every op is double-precision (`f64`-native, one thread per element).
///
/// # Errors
///
/// Returns [`StatsError::ComputationError`] if `oxicuda-ptx` fails to emit the
/// module.
pub fn generate_normal_pdf_ptx() -> StatsResult<String> {
    KernelBuilder::new("normal_pdf_batch")
        .target(SmVersion::Sm80)
        .param("out_ptr", PtxType::U64)
        .param("n", PtxType::U32)
        .param("in_ptr", PtxType::U64)
        .body(move |b| {
            let tid = b.global_thread_id_x();
            let n_reg = b.load_param_u32("n");
            b.if_lt_u32(tid.clone(), n_reg, |b| {
                let in_ptr = b.load_param_u64("in_ptr");
                let out_ptr = b.load_param_u64("out_ptr");
                let eight = b.mov_imm_u32(8);
                let off = b.mul_wide_u32_to_u64(tid.clone(), eight);
                let in_addr = b.add_u64(in_ptr, off.clone());
                let x = b.load_global_f64(in_addr);
                // PDF(x) = exp(-0.5 * x^2) * INV_SQRT_2PI.
                // `mul_rn_f64`/`neg_f64` are private, so we multiply `a*b` via
                // `fma_f64(a, b, zero)` (a*b + 0 = a*b).
                let zero = b.mov_imm_f64(0.0);
                let neg_half = b.mov_imm_f64(-0.5);
                let x2 = b.fma_f64(x.clone(), x.clone(), zero.clone());
                let arg = b.fma_f64(neg_half, x2, zero.clone());
                let e = b.exp_f64(&arg);
                let inv_sqrt_2pi = b.mov_imm_f64(0.398_942_280_401_432_677_94_f64);
                let y = b.fma_f64(e, inv_sqrt_2pi, zero);
                let out_addr = b.add_u64(out_ptr, off);
                b.store_global_f64(out_addr, y);
            });
            b.ret();
        })
        .build()
        .map_err(|e| {
            StatsError::ComputationError(format!("normal_pdf_batch PTX generation failed: {e}"))
        })
}

/// Generate the PTX text for the `normal_cdf_batch` kernel.
///
/// This is a fully **GPU-free** code generator: it builds and returns the PTX
/// string without touching any CUDA device, so it can be inspected and tested
/// on any platform. The kernel takes `(out_ptr: u64, n: u32, in_ptr: u64)` and
/// computes `out_ptr[tid] = 0.5 * (1 + erf(x * INV_SQRT2))` for `tid < n`,
/// where `x = in_ptr[tid]` is an `f64` at byte offset `tid*8`. The `erf`
/// evaluation uses [`BodyBuilder::erf_f64`](oxicuda_ptx::builder::BodyBuilder::erf_f64)
/// (A&S 7.1.26) and every op is double-precision (`f64`-native, one thread per
/// element).
///
/// # Errors
///
/// Returns [`StatsError::ComputationError`] if `oxicuda-ptx` fails to emit the
/// module.
pub fn generate_normal_cdf_ptx() -> StatsResult<String> {
    KernelBuilder::new("normal_cdf_batch")
        .target(SmVersion::Sm80)
        .param("out_ptr", PtxType::U64)
        .param("n", PtxType::U32)
        .param("in_ptr", PtxType::U64)
        .body(move |b| {
            let tid = b.global_thread_id_x();
            let n_reg = b.load_param_u32("n");
            b.if_lt_u32(tid.clone(), n_reg, |b| {
                let in_ptr = b.load_param_u64("in_ptr");
                let out_ptr = b.load_param_u64("out_ptr");
                let eight = b.mov_imm_u32(8);
                let off = b.mul_wide_u32_to_u64(tid.clone(), eight);
                let in_addr = b.add_u64(in_ptr, off.clone());
                let x = b.load_global_f64(in_addr);
                // CDF(x) = 0.5 * (1 + erf(x * INV_SQRT2)).
                // `mul_rn_f64` is private, so we multiply `a*b` via
                // `fma_f64(a, b, zero)` (a*b + 0 = a*b).
                let zero = b.mov_imm_f64(0.0);
                let inv_sqrt2 = b.mov_imm_f64(0.707_106_781_186_547_524_4_f64);
                let scaled = b.fma_f64(x.clone(), inv_sqrt2, zero.clone());
                let er = b.erf_f64(&scaled);
                let one = b.mov_imm_f64(1.0);
                let sum = b.add_f64(er, one);
                let half = b.mov_imm_f64(0.5);
                let y = b.fma_f64(sum, half, zero);
                let out_addr = b.add_u64(out_ptr, off);
                b.store_global_f64(out_addr, y);
            });
            b.ret();
        })
        .build()
        .map_err(|e| {
            StatsError::ComputationError(format!("normal_cdf_batch PTX generation failed: {e}"))
        })
}

/// Initialize the CUDA driver and build a stream bound to device 0.
///
/// Returns the owning [`oxicuda_driver::Context`] (in an `Arc`) alongside a
/// [`oxicuda_driver::stream::Stream`]; the caller must keep the context alive
/// for the lifetime of any kernel launch. All failures map to
/// [`StatsError::ComputationError`].
fn build_handle() -> StatsResult<(Arc<oxicuda_driver::Context>, oxicuda_driver::stream::Stream)> {
    oxicuda_driver::init()
        .map_err(|e| StatsError::ComputationError(format!("CUDA unavailable: {e}")))?;
    let count = oxicuda_driver::device::Device::count()
        .map_err(|e| StatsError::ComputationError(format!("device count: {e}")))?;
    if count <= 0 {
        return Err(StatsError::ComputationError(
            "no NVIDIA CUDA device available".into(),
        ));
    }
    let dev = oxicuda_driver::device::Device::get(0)
        .map_err(|e| StatsError::ComputationError(format!("device get: {e}")))?;
    let ctx = Arc::new(
        oxicuda_driver::Context::new(&dev)
            .map_err(|e| StatsError::ComputationError(format!("context: {e}")))?,
    );
    let stream = oxicuda_driver::stream::Stream::new(&ctx)
        .map_err(|e| StatsError::ComputationError(format!("stream: {e}")))?;
    Ok((ctx, stream))
}

/// Map an `oxicuda` driver/memory transfer error into
/// [`StatsError::ComputationError`].
fn launch_err(e: oxicuda_driver::CudaError) -> StatsError {
    StatsError::ComputationError(format!("oxicuda: {e}"))
}

/// Evaluate the standard-normal PDF over a batch of `f64` inputs on a CUDA
/// device, keeping full `f64` precision.
///
/// Returns one `f64` result per input element. The flow is: generate the
/// bespoke `normal_pdf_batch` PTX kernel → build a device handle → upload the
/// input buffer → launch `normal_pdf_batch` → synchronize → copy results back.
///
/// # Errors
///
/// Returns [`StatsError::ComputationError`] on PTX generation, no available
/// device, module load, kernel lookup, launch, sync, or buffer transfer
/// failure.
///
/// An empty input yields an empty output (`Ok(Vec::new())`).
pub fn cuda_normal_pdf_batch(input: &[f64]) -> StatsResult<Vec<f64>> {
    if input.is_empty() {
        return Ok(Vec::new());
    }
    let n = input.len();
    let ptx = generate_normal_pdf_ptx()?;

    // Keep `_ctx` alive across the launch so device memory + the loaded module
    // remain valid.
    let (_ctx, stream) = build_handle()?;

    let d_in = DeviceBuffer::from_host(input).map_err(launch_err)?;
    let d_out = DeviceBuffer::<f64>::alloc(n).map_err(launch_err)?;

    let module = Arc::new(
        Module::from_ptx(&ptx)
            .map_err(|e| StatsError::ComputationError(format!("module from_ptx: {e}")))?,
    );
    let kernel = Kernel::from_module(module, "normal_pdf_batch")
        .map_err(|e| StatsError::ComputationError(format!("kernel from_module: {e}")))?;

    let block = 256u32;
    let grid = grid_size_for(n as u32, block);
    let params = LaunchParams::new(grid, block);

    let args = (d_out.as_device_ptr(), n as u32, d_in.as_device_ptr());

    kernel
        .launch(&params, &stream, &args)
        .map_err(|e| StatsError::ComputationError(format!("launch: {e}")))?;
    stream
        .synchronize()
        .map_err(|e| StatsError::ComputationError(format!("sync: {e}")))?;

    let mut host_out = vec![0.0f64; n];
    d_out.copy_to_host(&mut host_out).map_err(launch_err)?;
    Ok(host_out)
}

/// Evaluate the standard-normal CDF over a batch of `f64` inputs on a CUDA
/// device, keeping full `f64` precision.
///
/// Returns one `f64` result per input element. The flow is: generate the
/// bespoke `normal_cdf_batch` PTX kernel → build a device handle → upload the
/// input buffer → launch `normal_cdf_batch` → synchronize → copy results back.
///
/// # Errors
///
/// Returns [`StatsError::ComputationError`] on PTX generation, no available
/// device, module load, kernel lookup, launch, sync, or buffer transfer
/// failure.
///
/// An empty input yields an empty output (`Ok(Vec::new())`).
pub fn cuda_normal_cdf_batch(input: &[f64]) -> StatsResult<Vec<f64>> {
    if input.is_empty() {
        return Ok(Vec::new());
    }
    let n = input.len();
    let ptx = generate_normal_cdf_ptx()?;

    // Keep `_ctx` alive across the launch so device memory + the loaded module
    // remain valid.
    let (_ctx, stream) = build_handle()?;

    let d_in = DeviceBuffer::from_host(input).map_err(launch_err)?;
    let d_out = DeviceBuffer::<f64>::alloc(n).map_err(launch_err)?;

    let module = Arc::new(
        Module::from_ptx(&ptx)
            .map_err(|e| StatsError::ComputationError(format!("module from_ptx: {e}")))?,
    );
    let kernel = Kernel::from_module(module, "normal_cdf_batch")
        .map_err(|e| StatsError::ComputationError(format!("kernel from_module: {e}")))?;

    let block = 256u32;
    let grid = grid_size_for(n as u32, block);
    let params = LaunchParams::new(grid, block);

    let args = (d_out.as_device_ptr(), n as u32, d_in.as_device_ptr());

    kernel
        .launch(&params, &stream, &args)
        .map_err(|e| StatsError::ComputationError(format!("launch: {e}")))?;
    stream
        .synchronize()
        .map_err(|e| StatsError::ComputationError(format!("sync: {e}")))?;

    let mut host_out = vec![0.0f64; n];
    d_out.copy_to_host(&mut host_out).map_err(launch_err)?;
    Ok(host_out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pdf_cpu(x: f64) -> f64 {
        (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
    }

    fn erf_cpu(x: f64) -> f64 {
        if x < 0.0 {
            return -erf_cpu(-x);
        }
        let t = 1.0 / (1.0 + 0.3275911 * x);
        let poly = t
            * (0.254_829_592
                + t * (-0.284_496_736
                    + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
        1.0 - poly * (-x * x).exp()
    }

    fn cdf_cpu(x: f64) -> f64 {
        0.5 * (1.0 + erf_cpu(x / std::f64::consts::SQRT_2))
    }

    #[test]
    fn generate_normal_pdf_ptx_has_markers() {
        let ptx = generate_normal_pdf_ptx().expect("ptx");
        assert!(
            ptx.contains(".entry normal_pdf_batch"),
            "missing entry: {ptx}"
        );
        assert!(ptx.contains("sm_80"), "missing target: {ptx}");
        assert!(
            ptx.contains("ld.global.f64"),
            "missing ld.global.f64: {ptx}"
        );
        assert!(
            ptx.contains("st.global.f64"),
            "missing st.global.f64: {ptx}"
        );
        assert!(ptx.contains("fma.rn.f64"), "missing fma.rn.f64: {ptx}");
        assert!(ptx.contains("shl.b64"), "missing shl.b64: {ptx}");
        assert!(
            ptx.contains("cvt.rzi.s32.f64"),
            "missing cvt.rzi.s32.f64: {ptx}"
        );
        assert!(
            !ptx.contains(".approx.f64"),
            "honest f64 invariant violated: {ptx}"
        );
    }

    #[test]
    fn generate_normal_cdf_ptx_has_markers() {
        let ptx = generate_normal_cdf_ptx().expect("ptx");
        assert!(
            ptx.contains(".entry normal_cdf_batch"),
            "missing entry: {ptx}"
        );
        assert!(ptx.contains("sm_80"), "missing target: {ptx}");
        assert!(
            ptx.contains("ld.global.f64"),
            "missing ld.global.f64: {ptx}"
        );
        assert!(
            ptx.contains("st.global.f64"),
            "missing st.global.f64: {ptx}"
        );
        assert!(ptx.contains("fma.rn.f64"), "missing fma.rn.f64: {ptx}");
        assert!(ptx.contains("abs.f64"), "missing abs.f64: {ptx}");
        assert!(
            ptx.contains("cvt.rzi.s32.f64"),
            "missing cvt.rzi.s32.f64: {ptx}"
        );
        assert!(
            !ptx.contains(".approx.f64"),
            "honest f64 invariant violated: {ptx}"
        );
    }

    #[test]
    fn cuda_normal_pdf_batch_or_skip() {
        if !cuda_is_available() {
            eprintln!("skipping: no NVIDIA CUDA device");
            assert!(!cuda_is_available());
            return;
        }
        let input: Vec<f64> = vec![-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
        let out = cuda_normal_pdf_batch(&input).expect("cuda pdf");
        assert_eq!(out.len(), input.len());
        for (i, &x) in input.iter().enumerate() {
            let want = pdf_cpu(x);
            assert!(
                (out[i] - want).abs() < 1e-6,
                "pdf({x}): got {}, want {want}",
                out[i]
            );
        }
    }

    #[test]
    fn cuda_normal_cdf_batch_or_skip() {
        if !cuda_is_available() {
            eprintln!("skipping: no NVIDIA CUDA device");
            assert!(!cuda_is_available());
            return;
        }
        let input: Vec<f64> = vec![-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
        let out = cuda_normal_cdf_batch(&input).expect("cuda cdf");
        assert_eq!(out.len(), input.len());
        for (i, &x) in input.iter().enumerate() {
            let want = cdf_cpu(x);
            assert!(
                (out[i] - want).abs() < 1e-5,
                "cdf({x}): got {}, want {want}",
                out[i]
            );
        }
    }

    /// §3b PTX→JIT→launch coverage: normal PDF over a wide range including
    /// deep tails (|x| up to 6).
    ///
    /// Tolerance rationale: PDF = exp(-0.5*x²)*INV_SQRT_2PI uses PTX `exp_f64`
    /// which is accurate to ~1 ULP end-to-end (no approximating polynomial). Both
    /// GPU and CPU evaluate the same formula, so 1e-9 absolute is well within the
    /// rounding budget at all points in this range.
    #[test]
    fn cuda_normal_pdf_tail_or_skip() {
        if !cuda_is_available() {
            eprintln!("skipping: no NVIDIA CUDA device");
            assert!(!cuda_is_available());
            return;
        }
        let input: Vec<f64> = vec![
            -6.0, -5.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0,
            6.0,
        ];
        let out = cuda_normal_pdf_batch(&input).expect("cuda pdf tails");
        assert_eq!(out.len(), input.len());
        for (i, &x) in input.iter().enumerate() {
            let want = pdf_cpu(x);
            assert!(
                (out[i] - want).abs() < 1e-9_f64,
                "pdf({x}): gpu={}, cpu={want}, |diff|={}",
                out[i],
                (out[i] - want).abs()
            );
        }
    }

    /// §3b PTX→JIT→launch coverage: normal CDF over a wide range including
    /// deep tails (|x| up to 6) with explicit boundary assertions.
    ///
    /// Tolerance rationale: CDF = 0.5*(1 + erf(x/√2)) with erf via A&S 7.1.26
    /// (~1.5e-7 absolute). Tolerance 1e-6 gives a factor-of-7 margin for any
    /// fma-chain rounding on top of the polynomial approximation error.
    #[test]
    fn cuda_normal_cdf_tail_or_skip() {
        if !cuda_is_available() {
            eprintln!("skipping: no NVIDIA CUDA device");
            assert!(!cuda_is_available());
            return;
        }
        let input: Vec<f64> = vec![
            -6.0, -5.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0,
            6.0,
        ];
        let out = cuda_normal_cdf_batch(&input).expect("cuda cdf tails");
        assert_eq!(out.len(), input.len());
        let tol = 1e-6_f64;
        for (i, &x) in input.iter().enumerate() {
            let want = cdf_cpu(x);
            assert!(
                (out[i] - want).abs() < tol,
                "cdf({x}): gpu={}, cpu={want}, |diff|={}",
                out[i],
                (out[i] - want).abs()
            );
        }
        // Explicit tail saturation: CDF(−6) → 0 and CDF(+6) → 1 within 1e-6.
        let idx_neg = input.iter().position(|&x| x == -6.0).expect("idx -6");
        let idx_pos = input.iter().position(|&x| x == 6.0).expect("idx +6");
        assert!(
            out[idx_neg] < tol,
            "cdf(-6) should be near 0, got {}",
            out[idx_neg]
        );
        assert!(
            (out[idx_pos] - 1.0_f64).abs() < tol,
            "cdf(6) should be near 1, got {}",
            out[idx_pos]
        );
    }
}
