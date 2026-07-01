//! Pure-Rust CUDA FFT dense path via the `oxicuda` ecosystem.
//!
//! This module provides an optional, **off-by-default** CUDA acceleration path for
//! 1-D complex FFTs, built entirely on the pure-Rust `oxicuda` crates
//! (`oxicuda-driver`, `oxicuda-memory`, `oxicuda-fft`). It is compiled only when the
//! `cuda` feature is enabled and it does **not** route through `scirs2-core`; the
//! existing CPU and `wgpu` FFT paths are untouched.
//!
//! ## Availability (runtime-probed, NVIDIA-only)
//!
//! `oxicuda-driver` loads `libcuda` at runtime. On a machine with no NVIDIA driver
//! (for example this development Mac), initialization fails and the public functions
//! return [`FFTError::BackendError`] rather than panicking. Call
//! [`cuda_is_available`] first to probe for a usable device; it never panics and
//! returns `false` when CUDA is unavailable.
//!
//! ## Normalization semantics (matches scirs2-fft CPU `fft`/`ifft`)
//!
//! `oxicuda`'s C2C transforms are **unnormalized** in both directions. To stay
//! consistent with scirs2-fft's CPU convention (SciPy-style: forward unnormalized,
//! inverse scaled by `1/N`), [`cuda_fft_1d`] applies no scaling while
//! [`cuda_ifft_1d`] divides every output element by `N`.
//!
//! ## Extensibility
//!
//! The flow (build a handle, upload host data to a [`oxicuda_memory::DeviceBuffer`],
//! create an [`oxicuda_fft::FftPlan`], execute, download) is intentionally explicit so
//! that 2-D / 3-D and R2C variants can be added by swapping the plan constructor and
//! the buffer element type without reworking the device-management boilerplate.

use crate::error::{FFTError, FFTResult};
use scirs2_core::numeric::Complex64;

/// Map an `oxicuda-fft` error into an [`FFTError`].
fn fft_err(e: oxicuda_fft::FftError) -> FFTError {
    FFTError::ComputationError(format!("oxicuda-fft: {e}"))
}

/// Map an `oxicuda-driver` CUDA error into an [`FFTError`].
fn cuda_err(e: oxicuda_driver::CudaError) -> FFTError {
    FFTError::BackendError(format!("oxicuda CUDA driver: {e}"))
}

/// Probe whether a usable NVIDIA CUDA device is available at runtime.
///
/// Never panics. Returns `false` when the CUDA driver cannot be initialized
/// (for example on non-NVIDIA platforms such as macOS) or when no device is present.
pub fn cuda_is_available() -> bool {
    oxicuda_driver::init().is_ok()
        && oxicuda_driver::device::Device::count()
            .map(|c| c > 0)
            .unwrap_or(false)
}

/// Initialize the CUDA driver and build an FFT handle bound to device 0.
///
/// Returns the owning [`oxicuda_driver::Context`] (wrapped in an `Arc`) alongside the
/// [`oxicuda_fft::FftHandle`]; the caller must keep the context alive for the lifetime
/// of any plan execution.
fn build_handle() -> FFTResult<(
    std::sync::Arc<oxicuda_driver::Context>,
    oxicuda_fft::FftHandle,
)> {
    oxicuda_driver::init().map_err(|e| FFTError::BackendError(format!("CUDA unavailable: {e}")))?;
    let count = oxicuda_driver::device::Device::count().map_err(cuda_err)?;
    if count <= 0 {
        return Err(FFTError::BackendError(
            "no NVIDIA CUDA device available".into(),
        ));
    }
    let dev = oxicuda_driver::device::Device::get(0).map_err(cuda_err)?;
    let ctx = std::sync::Arc::new(oxicuda_driver::Context::new(&dev).map_err(cuda_err)?);
    let handle = oxicuda_fft::FftHandle::new(&ctx).map_err(fft_err)?;
    Ok((ctx, handle))
}

/// Compute a 1-D forward complex FFT on a CUDA device (unnormalized, SciPy convention).
///
/// Mirrors scirs2-fft's CPU `fft`: the forward transform applies no normalization.
/// Returns [`FFTError::BackendError`] if no NVIDIA CUDA device is available at runtime.
pub fn cuda_fft_1d(input: &[Complex64]) -> FFTResult<Vec<Complex64>> {
    if input.is_empty() {
        return Ok(Vec::new());
    }
    let n = input.len();
    // Keep the context alive (`_ctx`) until the end of the function so device memory
    // and the FFT plan remain valid through `execute`/`copy_to_host`.
    let (_ctx, handle) = build_handle()?;

    let dev_in: Vec<oxicuda_fft::Complex<f64>> = input
        .iter()
        .map(|c| oxicuda_fft::Complex::<f64>::new(c.re, c.im))
        .collect();

    let plan = oxicuda_fft::FftPlan::new_1d(n, oxicuda_fft::FftType::C2C, 1)
        .map_err(fft_err)?
        .with_precision(oxicuda_fft::FftPrecision::Double);

    let d_in = oxicuda_memory::DeviceBuffer::from_host(&dev_in).map_err(cuda_err)?;
    let d_out =
        oxicuda_memory::DeviceBuffer::<oxicuda_fft::Complex<f64>>::alloc(n).map_err(cuda_err)?;

    handle
        .execute(
            &plan,
            d_in.as_device_ptr(),
            d_out.as_device_ptr(),
            oxicuda_fft::FftDirection::Forward,
        )
        .map_err(fft_err)?;

    let mut host_out = vec![oxicuda_fft::Complex::<f64>::new(0.0, 0.0); n];
    d_out.copy_to_host(&mut host_out).map_err(cuda_err)?;

    // Forward transform is unnormalized (matches scirs2-fft CPU `fft`): no scaling.
    let out: Vec<Complex64> = host_out
        .iter()
        .map(|c| Complex64::new(c.re, c.im))
        .collect();
    Ok(out)
}

/// Compute a 1-D inverse complex FFT on a CUDA device, normalized by `1/N`.
///
/// `oxicuda`'s inverse C2C transform is unnormalized, so this function divides every
/// output element by `N` to match scirs2-fft's CPU `ifft` (SciPy convention, which
/// applies the `1/N` factor on the inverse). Returns [`FFTError::BackendError`] if no
/// NVIDIA CUDA device is available at runtime.
pub fn cuda_ifft_1d(input: &[Complex64]) -> FFTResult<Vec<Complex64>> {
    if input.is_empty() {
        return Ok(Vec::new());
    }
    let n = input.len();
    // Keep the context alive (`_ctx`) until the end of the function so device memory
    // and the FFT plan remain valid through `execute`/`copy_to_host`.
    let (_ctx, handle) = build_handle()?;

    let dev_in: Vec<oxicuda_fft::Complex<f64>> = input
        .iter()
        .map(|c| oxicuda_fft::Complex::<f64>::new(c.re, c.im))
        .collect();

    let plan = oxicuda_fft::FftPlan::new_1d(n, oxicuda_fft::FftType::C2C, 1)
        .map_err(fft_err)?
        .with_precision(oxicuda_fft::FftPrecision::Double);

    let d_in = oxicuda_memory::DeviceBuffer::from_host(&dev_in).map_err(cuda_err)?;
    let d_out =
        oxicuda_memory::DeviceBuffer::<oxicuda_fft::Complex<f64>>::alloc(n).map_err(cuda_err)?;

    handle
        .execute(
            &plan,
            d_in.as_device_ptr(),
            d_out.as_device_ptr(),
            oxicuda_fft::FftDirection::Inverse,
        )
        .map_err(fft_err)?;

    let mut host_out = vec![oxicuda_fft::Complex::<f64>::new(0.0, 0.0); n];
    d_out.copy_to_host(&mut host_out).map_err(cuda_err)?;

    // oxicuda's inverse C2C is UNNORMALIZED; apply 1/N here so the result matches
    // scirs2-fft's CPU `ifft`, which follows the SciPy convention of scaling the
    // inverse transform by 1/N.
    let scale = n as f64;
    let out: Vec<Complex64> = host_out
        .iter()
        .map(|c| Complex64::new(c.re / scale, c.im / scale))
        .collect();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // CUDA driver APIs (cuCtxCreate, cuMemAlloc, cufftExec…) are serialised
    // internally by the driver, but our `build_handle()` creates a new primary-
    // context binding on each call.  When two test threads both call
    // `oxicuda_driver::Context::new` concurrently for the same device, the
    // driver can return the same underlying handle to both threads, leading to
    // use-after-free when the first thread drops its Arc<Context>.  Serialising
    // all tests in this module with a module-level Mutex eliminates the race
    // without requiring any changes to the production code paths.
    static CUDA_TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn cuda_fft_roundtrip_or_skip() {
        let _guard = CUDA_TEST_LOCK.lock().expect("cuda test lock");
        if !cuda_is_available() {
            eprintln!("skipping: no NVIDIA CUDA device");
            assert!(!cuda_is_available());
            return;
        }

        let input: Vec<Complex64> = (0..8).map(|i| Complex64::new(i as f64, 0.0)).collect();
        let spec = cuda_fft_1d(&input).expect("fwd");
        let back = cuda_ifft_1d(&spec).expect("inv");

        for (a, b) in input.iter().zip(back.iter()) {
            assert!((a.re - b.re).abs() < 1e-6);
            assert!((a.im - b.im).abs() < 1e-6);
        }
    }

    // ---- Non-power-of-two sizes + normalization convention check ----
    //
    // The existing roundtrip test uses n=8 (a power of two).  This test covers:
    //
    // (a) Forward+inverse roundtrip for n=6 and n=12 (both non-powers-of-two)
    //     with COMPLEX inputs that have nonzero imaginary parts, verifying that
    //     the device path handles mixed-radix plan sizes and that the combined
    //     forward/inverse normalization (forward unnormalized, inverse /N) is
    //     self-consistent — i.e. `cuda_ifft_1d(cuda_fft_1d(x)) ≈ x` within 1e-6.
    //
    // (b) Explicit normalization convention check for n=6:
    //     - FORWARD: `cuda_fft_1d` (unnormalized) output is compared against the
    //       scirs2-fft CPU `fft` for the SAME real-valued signal.  Both follow the
    //       SciPy convention (forward unnormalized), so outputs must agree to 1e-6.
    //     - INVERSE: `cuda_ifft_1d` (scales by 1/N) output is compared against the
    //       scirs2-fft CPU `ifft` for a known real-valued spectrum [6,0,0,0,0,0],
    //       whose analytically correct result is [1,1,1,1,1,1] (the inverse DFT of
    //       a single DC bin of amplitude N).  Both CPU and CUDA must agree with
    //       each other and with the analytic result to 1e-6.
    #[test]
    fn cuda_fft_non_power_of_two_or_skip() {
        let _guard = CUDA_TEST_LOCK.lock().expect("cuda test lock");
        if !cuda_is_available() {
            eprintln!("skipping: no NVIDIA CUDA device");
            assert!(!cuda_is_available());
            return;
        }

        // ── Part (a): roundtrip for n=6 with complex inputs (nonzero imaginary parts)
        let input6: Vec<Complex64> = vec![
            Complex64::new(1.0, 2.0),
            Complex64::new(3.0, 0.5),
            Complex64::new(0.5, 1.5),
            Complex64::new(2.0, 3.0),
            Complex64::new(1.5, 2.5),
            Complex64::new(0.7, 1.3),
        ];
        let spec6 = cuda_fft_1d(&input6).expect("fwd n=6");
        let back6 = cuda_ifft_1d(&spec6).expect("inv n=6");
        for (i, (orig, recon)) in input6.iter().zip(back6.iter()).enumerate() {
            let diff_re = (orig.re - recon.re).abs();
            let diff_im = (orig.im - recon.im).abs();
            assert!(
                diff_re < 1e-6,
                "n=6 roundtrip re[{i}]: orig={:.6} recon={:.6} diff={diff_re:.3e}",
                orig.re,
                recon.re
            );
            assert!(
                diff_im < 1e-6,
                "n=6 roundtrip im[{i}]: orig={:.6} recon={:.6} diff={diff_im:.3e}",
                orig.im,
                recon.im
            );
        }

        // ── Part (a): roundtrip for n=12 with complex inputs (nonzero imaginary parts)
        let input12: Vec<Complex64> = (0..12_usize)
            .map(|i| Complex64::new((i as f64 * 0.7_f64).sin(), (i as f64 * 0.3_f64).cos()))
            .collect();
        let spec12 = cuda_fft_1d(&input12).expect("fwd n=12");
        let back12 = cuda_ifft_1d(&spec12).expect("inv n=12");
        for (i, (orig, recon)) in input12.iter().zip(back12.iter()).enumerate() {
            let diff_re = (orig.re - recon.re).abs();
            let diff_im = (orig.im - recon.im).abs();
            assert!(
                diff_re < 1e-6,
                "n=12 roundtrip re[{i}]: orig={:.6} recon={:.6} diff={diff_re:.3e}",
                orig.re,
                recon.re
            );
            assert!(
                diff_im < 1e-6,
                "n=12 roundtrip im[{i}]: orig={:.6} recon={:.6} diff={diff_im:.3e}",
                orig.im,
                recon.im
            );
        }

        // ── Part (b): normalization convention — forward (unnormalized) ──────────
        // Use real-valued f64 inputs so we can compare against the CPU `fft`
        // function (which requires T: NumCast, satisfied by f64 but not Complex64).
        // Convert the same values to Complex64 (im=0) for the CUDA side.
        let signal6 = [1.0_f64, 2.0, 3.0, 2.0, 1.0, 0.5];
        let cpu_fwd6 = crate::fft::fft(&signal6, Some(6)).expect("cpu fft n=6");
        let cuda_in6: Vec<Complex64> = signal6.iter().map(|&v| Complex64::new(v, 0.0)).collect();
        let cuda_fwd6 = cuda_fft_1d(&cuda_in6).expect("cuda fft n=6 norm-check");

        assert_eq!(
            cpu_fwd6.len(),
            cuda_fwd6.len(),
            "forward output length mismatch"
        );
        for (k, (cpu, gpu)) in cpu_fwd6.iter().zip(cuda_fwd6.iter()).enumerate() {
            let diff_re = (cpu.re - gpu.re).abs();
            let diff_im = (cpu.im - gpu.im).abs();
            assert!(
                diff_re < 1e-6,
                "fwd norm re[{k}]: cpu={:.6} cuda={:.6} diff={diff_re:.3e}",
                cpu.re,
                gpu.re
            );
            assert!(
                diff_im < 1e-6,
                "fwd norm im[{k}]: cpu={:.6} cuda={:.6} diff={diff_im:.3e}",
                cpu.im,
                gpu.im
            );
        }

        // ── Part (b): normalization convention — inverse (scales by 1/N) ─────────
        // Use the known spectrum [6, 0, 0, 0, 0, 0] for n=6.  This is the DFT of
        // the all-ones signal, so ifft([6,0,…,0]) must return [1,1,1,1,1,1].
        // Pass as f64 to CPU ifft and as Complex64 to CUDA ifft; both must agree
        // with the analytic result and with each other to 1e-6.
        let dc_spectrum_f64 = [6.0_f64, 0.0, 0.0, 0.0, 0.0, 0.0];
        let cpu_inv6 = crate::fft::ifft(&dc_spectrum_f64, Some(6)).expect("cpu ifft n=6");
        let dc_spectrum_cx: Vec<Complex64> = dc_spectrum_f64
            .iter()
            .map(|&v| Complex64::new(v, 0.0))
            .collect();
        let cuda_inv6 = cuda_ifft_1d(&dc_spectrum_cx).expect("cuda ifft n=6 norm-check");

        assert_eq!(cpu_inv6.len(), 6, "cpu ifft output length");
        assert_eq!(cuda_inv6.len(), 6, "cuda ifft output length");
        for k in 0..6 {
            // Analytic: ifft([6,0,…,0])[n] = (1/6) * 6 * exp(0) = 1 for all n.
            let expected = 1.0_f64;
            let diff_cpu = (cpu_inv6[k].re - expected).abs();
            let diff_cuda = (cuda_inv6[k].re - expected).abs();
            let diff_cpu_im = cpu_inv6[k].im.abs();
            let diff_cuda_im = cuda_inv6[k].im.abs();
            let cpu_vs_cuda_re = (cpu_inv6[k].re - cuda_inv6[k].re).abs();
            let cpu_vs_cuda_im = (cpu_inv6[k].im - cuda_inv6[k].im).abs();
            assert!(
                diff_cpu < 1e-6,
                "inv norm cpu re[{k}]: got={:.6} expected={expected} diff={diff_cpu:.3e}",
                cpu_inv6[k].re
            );
            assert!(
                diff_cuda < 1e-6,
                "inv norm cuda re[{k}]: got={:.6} expected={expected} diff={diff_cuda:.3e}",
                cuda_inv6[k].re
            );
            assert!(
                diff_cpu_im < 1e-6,
                "inv norm cpu im[{k}] should be ~0: got={:.3e}",
                cpu_inv6[k].im
            );
            assert!(
                diff_cuda_im < 1e-6,
                "inv norm cuda im[{k}] should be ~0: got={:.3e}",
                cuda_inv6[k].im
            );
            assert!(
                cpu_vs_cuda_re < 1e-6,
                "cpu vs cuda inv re[{k}] diff={cpu_vs_cuda_re:.3e}"
            );
            assert!(
                cpu_vs_cuda_im < 1e-6,
                "cpu vs cuda inv im[{k}] diff={cpu_vs_cuda_im:.3e}"
            );
        }
    }
}
