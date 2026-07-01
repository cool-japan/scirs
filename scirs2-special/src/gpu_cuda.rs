#![cfg(feature = "cuda")]
//! Optional, off-by-default, NVIDIA-only pure-Rust CUDA path for batched `erf`,
//! built directly on the `oxicuda` ecosystem.
//!
//! This module is an opt-in CUDA acceleration path for the error function over
//! a batch of `f64` inputs. It is built entirely on the pure-Rust `oxicuda`
//! crates (`oxicuda-ptx`, `oxicuda-launch`, `oxicuda-driver`, `oxicuda-memory`),
//! compiled only when the crate-local `cuda` feature is enabled, and it does
//! **not** route through `scirs2-core`. The existing CPU and `wgpu` `erf` paths
//! are left completely untouched — this is purely additive.
//!
//! ## Availability (runtime-probed, NVIDIA-only)
//!
//! `oxicuda-driver` loads `libcuda` at runtime. On a machine with no NVIDIA
//! driver (for example this development Mac), initialization fails and the
//! public functions return a [`SpecialError`] variant rather than panicking.
//! Call [`cuda_is_available`] first to probe for a usable device; it never
//! panics and returns `false` when CUDA is unavailable. Importantly, the crate
//! still *compiles* on macOS with `--features cuda` — only the runtime probe
//! returns `false`.
//!
//! ## The CUSTOM-KERNEL pattern
//!
//! This module **generates a bespoke `erf_batch` PTX kernel** using
//! `oxicuda-ptx`'s instruction-level DSL — specifically the
//! [`erf_f64`](oxicuda_ptx::builder::BodyBuilder::erf_f64) transcendental
//! (Abramowitz & Stegun 7.1.26, ~1.5e-7 absolute accuracy) — then dispatches it
//! with `oxicuda-launch`. The PTX is produced by [`generate_erf_ptx`], which is
//! a fully **GPU-free** code generator: it can be inspected and unit-tested on
//! any platform without a CUDA device present. Only the actual device launch
//! (inside [`cuda_erf_batch`]) requires NVIDIA hardware.
//!
//! ## f64-native — a genuine precision advantage over the wgpu path
//!
//! The crate's `wgpu` `erf` path downcasts host `f64` inputs to `f32`, because
//! WGSL's portable storage type is `f32`. This CUDA path is **`f64`-native end
//! to end**: inputs are uploaded as `f64`, every arithmetic instruction is the
//! double-precision PTX op, and the result is read back as `f64`. The A&S
//! polynomial caps mathematical accuracy at ~1.5e-7, but the arithmetic carries
//! full `f64` precision (no `f32` bridge) and no `.approx.f64` instruction is
//! emitted.
//!
//! ## Kernel design — one thread per element
//!
//! The generated `erf_batch(out_ptr, n, in_ptr)` kernel has each thread `tid`
//! load `in_ptr[tid]` (an `f64` at byte offset `tid*8`), apply `erf_f64`, and
//! store the result to `out_ptr[tid]`. A `tid < n` guard makes the launch safe
//! for any grid size.

use oxicuda_driver::Module;
use oxicuda_launch::{grid_size_for, Kernel, LaunchParams};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::builder::KernelBuilder;
use oxicuda_ptx::ir::PtxType;
use std::sync::Arc;

use crate::error::{SpecialError, SpecialResult};

/// Probe whether a usable NVIDIA CUDA device is available at runtime.
///
/// Never panics. Returns `false` when the CUDA driver cannot be initialized
/// (for example on non-NVIDIA platforms such as macOS) or when no device is
/// present. Call this before [`cuda_erf_batch`] to decide whether the CUDA path
/// is usable on the current host.
pub fn cuda_is_available() -> bool {
    oxicuda_driver::init().is_ok()
        && oxicuda_driver::device::Device::count()
            .map(|c| c > 0)
            .unwrap_or(false)
}

/// Generate the PTX text for the `erf_batch` kernel.
///
/// This is a fully **GPU-free** code generator: it builds and returns the PTX
/// string without touching any CUDA device, so it can be inspected and tested
/// on any platform. The kernel takes `(out_ptr: u64, n: u32, in_ptr: u64)` and
/// computes `out_ptr[tid] = erf(in_ptr[tid])` for `tid < n`, where each element
/// is an `f64` at byte offset `tid*8`. The `erf` evaluation uses
/// [`BodyBuilder::erf_f64`](oxicuda_ptx::builder::BodyBuilder::erf_f64) (A&S
/// 7.1.26).
///
/// # Errors
///
/// Returns [`SpecialError::ComputationError`] if `oxicuda-ptx` fails to emit the
/// module.
pub fn generate_erf_ptx() -> SpecialResult<String> {
    KernelBuilder::new("erf_batch")
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
                let y = b.erf_f64(&x);
                let out_addr = b.add_u64(out_ptr, off);
                b.store_global_f64(out_addr, y);
            });
            b.ret();
        })
        .build()
        .map_err(|e| {
            SpecialError::ComputationError(format!("erf_batch PTX generation failed: {e}"))
        })
}

/// Initialize the CUDA driver and build a stream bound to device 0.
///
/// Returns the owning [`oxicuda_driver::Context`] (in an `Arc`) alongside a
/// [`oxicuda_driver::stream::Stream`]; the caller must keep the context alive
/// for the lifetime of any kernel launch. All failures map to
/// [`SpecialError::GpuNotAvailable`].
fn build_handle() -> SpecialResult<(Arc<oxicuda_driver::Context>, oxicuda_driver::stream::Stream)> {
    oxicuda_driver::init()
        .map_err(|e| SpecialError::GpuNotAvailable(format!("CUDA unavailable: {e}")))?;
    let count = oxicuda_driver::device::Device::count()
        .map_err(|e| SpecialError::GpuNotAvailable(format!("device count: {e}")))?;
    if count <= 0 {
        return Err(SpecialError::GpuNotAvailable(
            "no NVIDIA CUDA device available".into(),
        ));
    }
    let dev = oxicuda_driver::device::Device::get(0)
        .map_err(|e| SpecialError::GpuNotAvailable(format!("device get: {e}")))?;
    let ctx = Arc::new(
        oxicuda_driver::Context::new(&dev)
            .map_err(|e| SpecialError::GpuNotAvailable(format!("context: {e}")))?,
    );
    let stream = oxicuda_driver::stream::Stream::new(&ctx)
        .map_err(|e| SpecialError::GpuNotAvailable(format!("stream: {e}")))?;
    Ok((ctx, stream))
}

/// Map an `oxicuda` driver/memory transfer error into
/// [`SpecialError::ComputationError`].
fn launch_err(e: oxicuda_driver::CudaError) -> SpecialError {
    SpecialError::ComputationError(format!("oxicuda: {e}"))
}

/// Evaluate `erf` over a batch of `f64` inputs on a CUDA device, keeping full
/// `f64` precision.
///
/// Returns one `f64` result per input element. The flow is: generate the
/// bespoke `erf_batch` PTX kernel → build a device handle → upload the input
/// buffer → launch `erf_batch` → synchronize → copy results back.
///
/// # Errors
///
/// - [`SpecialError::GpuNotAvailable`] when no NVIDIA CUDA device is available.
/// - [`SpecialError::ComputationError`] on PTX generation, module load, kernel
///   lookup, launch, sync, or buffer transfer failure.
///
/// An empty input yields an empty output (`Ok(Vec::new())`).
pub fn cuda_erf_batch(input: &[f64]) -> SpecialResult<Vec<f64>> {
    if input.is_empty() {
        return Ok(Vec::new());
    }
    let n = input.len();
    let ptx = generate_erf_ptx()?;

    // Keep `_ctx` alive across the launch so device memory + the loaded module
    // remain valid.
    let (_ctx, stream) = build_handle()?;

    let d_in = DeviceBuffer::from_host(input).map_err(launch_err)?;
    let d_out = DeviceBuffer::<f64>::alloc(n).map_err(launch_err)?;

    let module = Arc::new(
        Module::from_ptx(&ptx)
            .map_err(|e| SpecialError::ComputationError(format!("module from_ptx: {e}")))?,
    );
    let kernel = Kernel::from_module(module, "erf_batch")
        .map_err(|e| SpecialError::ComputationError(format!("kernel from_module: {e}")))?;

    let block = 256u32;
    let grid = grid_size_for(n as u32, block);
    let params = LaunchParams::new(grid, block);

    let args = (d_out.as_device_ptr(), n as u32, d_in.as_device_ptr());

    kernel
        .launch(&params, &stream, &args)
        .map_err(|e| SpecialError::ComputationError(format!("launch: {e}")))?;
    stream
        .synchronize()
        .map_err(|e| SpecialError::ComputationError(format!("sync: {e}")))?;

    let mut host_out = vec![0.0f64; n];
    d_out.copy_to_host(&mut host_out).map_err(launch_err)?;
    Ok(host_out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_erf_ptx_has_erf_markers() {
        let ptx = generate_erf_ptx().expect("ptx");
        assert!(ptx.contains(".entry erf_batch"), "missing entry: {ptx}");
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
    fn cuda_erf_batch_or_skip() {
        if !cuda_is_available() {
            eprintln!("skipping: no NVIDIA CUDA device");
            assert!(!cuda_is_available());
            return;
        }
        let input: Vec<f64> = vec![-1.5, -0.5, 0.0, 0.25, 0.5, 1.0, 2.0];
        let out = cuda_erf_batch(&input).expect("cuda erf");
        assert_eq!(out.len(), input.len());
        for (i, &x) in input.iter().enumerate() {
            let want = crate::erf(x);
            assert!(
                (out[i] - want).abs() < 1e-6,
                "erf({x}): got {}, want {want}",
                out[i]
            );
        }
    }

    /// §3b PTX→JIT→launch coverage: `erf` over the full f64 range including
    /// deep tails (|x| up to 8).
    ///
    /// Tolerance rationale: A&S 7.1.26 max absolute error is ~1.5e-7; using 1e-6
    /// gives comfortable headroom against any rounding in the PTX fma chain.
    /// At |x| ≥ 4 erf saturates to ±1 in f64 to within 1.5e-8, so the explicit
    /// ±1 assertions are strictly tighter than the per-point comparison.
    #[test]
    fn cuda_erf_tail_or_skip() {
        if !cuda_is_available() {
            eprintln!("skipping: no NVIDIA CUDA device");
            assert!(!cuda_is_available());
            return;
        }
        let input: Vec<f64> = vec![
            -8.0, -6.0, -4.0, -2.0, -1.0, -0.25, 0.0, 0.25, 1.0, 2.0, 4.0, 6.0, 8.0,
        ];
        let out = cuda_erf_batch(&input).expect("cuda erf tails");
        assert_eq!(out.len(), input.len());
        let tol = 1e-6_f64;
        for (i, &x) in input.iter().enumerate() {
            let want = crate::erf(x);
            assert!(
                (out[i] - want).abs() < tol,
                "erf({x}): gpu={}, cpu={want}, |diff|={}",
                out[i],
                (out[i] - want).abs()
            );
        }
        // Explicit saturation: erf(±8) must be indistinguishable from ±1 within 1e-6.
        let neg_tail = input.iter().position(|&x| x == -8.0).expect("idx -8");
        let pos_tail = input.iter().position(|&x| x == 8.0).expect("idx +8");
        assert!(
            (out[neg_tail] - (-1.0_f64)).abs() < tol,
            "erf(-8) should saturate to -1, got {}",
            out[neg_tail]
        );
        assert!(
            (out[pos_tail] - 1.0_f64).abs() < tol,
            "erf(8) should saturate to +1, got {}",
            out[pos_tail]
        );
    }
}
