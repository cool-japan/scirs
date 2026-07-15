//! Runtime GPU availability detection.
//!
//! This module provides honest, runtime answers to "does this machine have a
//! usable GPU?" for [`super::types::PlatformCapabilities`]. It only *detects*
//! hardware — GPU *compute* is provided elsewhere (CUDA kernels live in the
//! per-crate `oxicuda-*` backends; portable compute uses wgpu/Metal behind the
//! corresponding crate features).
//!
//! # CUDA detection
//!
//! On Linux and Windows the NVIDIA driver ships a user-space driver library
//! (`libcuda.so.1`/`libcuda.so` on Linux, `nvcuda.dll` on Windows). We probe
//! it at runtime via `dlopen`/`LoadLibrary` (through the pure-Rust
//! `libloading` wrapper) and report `true` only when `cuInit(0)` succeeds
//! and `cuDeviceGetCount` enumerates at least one device. No CUDA SDK, crate
//! feature, or build-time dependency is required, and machines without the
//! driver simply get `false` — never an error or a panic. macOS and other
//! platforms have no CUDA driver, so the probe is compiled out and always
//! reports `false` there.
//!
//! # Metal detection
//!
//! On macOS, when the `metal` feature is enabled we ask the Metal framework
//! for the system default device. Without the feature we fall back to a
//! documented platform heuristic: every Apple Silicon Mac has a Metal-capable
//! GPU, and every Intel Mac able to run a macOS version supported by this
//! crate's toolchain (10.14+) ships a Metal-capable GPU as well, so macOS
//! itself implies Metal availability. Non-macOS targets always report
//! `false`.
//!
//! All probes are cached in [`std::sync::OnceLock`]s, so repeated calls (e.g.
//! from `PlatformCapabilities::detect()` in hot paths) are a single atomic
//! load after the first invocation.

#[cfg(any(target_os = "linux", target_os = "windows", target_os = "macos"))]
use std::sync::OnceLock;

/// `CUresult` success code (`CUDA_SUCCESS`) from the CUDA Driver API.
#[cfg(any(target_os = "linux", target_os = "windows"))]
const CUDA_SUCCESS: i32 = 0;

/// Detect whether a working NVIDIA CUDA driver with at least one device is
/// present at runtime.
///
/// The probe dynamically loads the CUDA driver library and calls
/// `cuInit(0)` followed by `cuDeviceGetCount`; it returns `true` only when
/// both succeed and the device count is positive. The result is computed once
/// per process and cached. This is detection only — CUDA compute lives in the
/// per-crate `oxicuda-*` backends, not in `scirs2-core`.
#[cfg(any(target_os = "linux", target_os = "windows"))]
pub fn detect_cuda_runtime() -> bool {
    static CUDA_AVAILABLE: OnceLock<bool> = OnceLock::new();
    *CUDA_AVAILABLE.get_or_init(probe_cuda_driver)
}

/// Detect whether a working NVIDIA CUDA driver is present at runtime.
///
/// This platform (e.g. macOS, WASM, BSDs) has no NVIDIA CUDA driver, so the
/// answer is always `false`.
#[cfg(not(any(target_os = "linux", target_os = "windows")))]
pub fn detect_cuda_runtime() -> bool {
    false
}

/// Candidate CUDA driver library names, in preference order.
#[cfg(target_os = "linux")]
const CUDA_DRIVER_CANDIDATES: &[&str] = &["libcuda.so.1", "libcuda.so"];

/// Candidate CUDA driver library names, in preference order.
#[cfg(target_os = "windows")]
const CUDA_DRIVER_CANDIDATES: &[&str] = &["nvcuda.dll"];

/// Load the CUDA driver library and query it for usable devices.
///
/// Returns `false` (never panics) when the driver is absent, exports the
/// wrong symbols, fails to initialize, or reports zero devices.
#[cfg(any(target_os = "linux", target_os = "windows"))]
fn probe_cuda_driver() -> bool {
    for name in CUDA_DRIVER_CANDIDATES {
        // SAFETY: loading the vendor-provided CUDA driver library. Library
        // initializers of the NVIDIA driver are designed to run under
        // dlopen/LoadLibrary; a missing or broken library yields Err.
        let lib = match unsafe { libloading::Library::new(*name) } {
            Ok(lib) => lib,
            Err(_) => continue,
        };
        match probe_loaded_driver(&lib) {
            Some(available) => {
                // `cuInit` has been invoked: the driver may have spawned
                // internal threads and registered process-global state, so
                // unloading it via `dlclose` would be unsound. Keep the
                // handle resident for the process lifetime (the probe runs
                // at most once thanks to the `OnceLock` cache).
                std::mem::forget(lib);
                return available;
            }
            // Required symbols missing — `cuInit` was never called, so the
            // library can be dropped (unloaded) safely; try the next name.
            None => continue,
        }
    }
    false
}

/// Resolve `cuInit`/`cuDeviceGetCount` in an already-loaded driver library
/// and ask it for the device count.
///
/// Returns `None` when either symbol is missing (i.e. `cuInit` was never
/// called), and `Some(available)` after a completed probe.
#[cfg(any(target_os = "linux", target_os = "windows"))]
fn probe_loaded_driver(lib: &libloading::Library) -> Option<bool> {
    // `CUDAAPI` is `__stdcall` on 32-bit Windows and the C ABI elsewhere;
    // `extern "system"` selects the right one on every target.
    type CuInitFn = unsafe extern "system" fn(u32) -> i32;
    type CuDeviceGetCountFn = unsafe extern "system" fn(*mut i32) -> i32;

    // SAFETY: the symbol name and ABI match the CUDA Driver API contract
    // `CUresult cuInit(unsigned int Flags)`.
    let cu_init: libloading::Symbol<'_, CuInitFn> = unsafe { lib.get(b"cuInit\0") }.ok()?;
    // SAFETY: matches `CUresult cuDeviceGetCount(int *count)`.
    let cu_device_get_count: libloading::Symbol<'_, CuDeviceGetCountFn> =
        unsafe { lib.get(b"cuDeviceGetCount\0") }.ok()?;

    // SAFETY: `cuInit` accepts `Flags == 0` and is the documented entry point
    // that must precede any other driver call.
    let init_status = unsafe { cu_init(0) };
    if init_status != CUDA_SUCCESS {
        return Some(false);
    }

    let mut device_count: i32 = 0;
    // SAFETY: we pass a valid, writable pointer to an `i32`, as required by
    // `cuDeviceGetCount`.
    let count_status = unsafe { cu_device_get_count(&mut device_count) };
    Some(count_status == CUDA_SUCCESS && device_count > 0)
}

/// Detect whether a Metal-capable GPU is available at runtime.
///
/// With the `metal` feature enabled this queries the Metal framework for the
/// system default device. Without the feature it uses a documented platform
/// heuristic: all Apple Silicon Macs, and all Intel Macs capable of running
/// macOS 10.14+, ship Metal-capable GPUs, so running on macOS implies Metal
/// availability. The result is cached per process.
#[cfg(target_os = "macos")]
pub fn detect_metal_runtime() -> bool {
    static METAL_AVAILABLE: OnceLock<bool> = OnceLock::new();
    *METAL_AVAILABLE.get_or_init(probe_metal)
}

/// Detect whether a Metal-capable GPU is available at runtime.
///
/// Metal is Apple-only; on non-macOS targets the answer is always `false`.
#[cfg(not(target_os = "macos"))]
pub fn detect_metal_runtime() -> bool {
    false
}

/// Ask the Metal framework for the system default device.
#[cfg(all(target_os = "macos", feature = "metal"))]
fn probe_metal() -> bool {
    metal::Device::system_default().is_some()
}

/// Heuristic Metal probe used when the `metal` feature is disabled.
///
/// Every Apple Silicon Mac has an integrated Metal GPU, and every Intel Mac
/// supported by macOS 10.14+ (the oldest macOS this crate's toolchain
/// targets) has a Metal-capable GPU, so macOS itself implies Metal support.
#[cfg(all(target_os = "macos", not(feature = "metal")))]
fn probe_metal() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Detection must never panic and must be stable across calls (cached).
    #[test]
    fn detection_does_not_panic_and_is_stable() {
        let first_cuda = detect_cuda_runtime();
        let second_cuda = detect_cuda_runtime();
        assert_eq!(first_cuda, second_cuda);

        let first_metal = detect_metal_runtime();
        let second_metal = detect_metal_runtime();
        assert_eq!(first_metal, second_metal);
    }

    /// Apple Silicon Macs always have a Metal-capable GPU.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn metal_reported_on_apple_silicon() {
        assert!(detect_metal_runtime());
    }

    /// macOS has no NVIDIA CUDA driver, so CUDA must never be reported.
    #[cfg(target_os = "macos")]
    #[test]
    fn cuda_never_reported_on_macos() {
        assert!(!detect_cuda_runtime());
    }

    /// Metal is Apple-only and must never be reported elsewhere.
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn metal_never_reported_off_macos() {
        assert!(!detect_metal_runtime());
    }

    /// `PlatformCapabilities::detect()` must agree with the runtime probes.
    #[test]
    fn platform_capabilities_reflect_runtime_probes() {
        let caps = super::super::types::PlatformCapabilities::detect();
        assert_eq!(caps.cuda_available, detect_cuda_runtime());
        assert_eq!(caps.metal_available, detect_metal_runtime());
    }
}
