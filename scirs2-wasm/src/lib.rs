//! # SciRS2-WASM: WebAssembly Bindings for SciRS2
//!
//! High-performance scientific computing in the browser and Node.js.
//!
//! ## Features
//!
//! - **Pure Rust**: 100% safe Rust code compiled to WASM
//! - **SIMD Support**: Optional WASM SIMD (wasm32-simd128) acceleration
//! - **Async Operations**: Non-blocking computations with async/await
//! - **TypeScript Support**: Full TypeScript type definitions
//! - **Memory Efficient**: Optimized memory management for browser environments
//! - **Zero-copy**: Direct array buffer access when possible
//!
//! ## Modules
//!
//! - `array`: N-dimensional array operations
//! - `linalg`: Linear algebra (matrix operations, decompositions)
//! - `stats`: Statistical functions (distributions, tests, descriptive stats)
//! - `fft`: Fast Fourier Transform operations
//! - `signal`: Signal processing (filtering, convolution, wavelets)
//! - `integrate`: Numerical integration and ODE solvers
//! - `interpolate`: Interpolation (linear, spline, Lagrange, PCHIP, Akima)
//! - `optimize`: Optimization algorithms (minimize, curve fitting)
//! - `random`: Random number generation and distributions
//!
//! ## Example Usage (JavaScript)
//!
//! ```javascript
//! import * as scirs2 from 'scirs2-wasm';
//!
//! // Initialize the library
//! await scirs2.default();
//!
//! // Create arrays
//! const a = scirs2.array([1, 2, 3, 4]);
//! const b = scirs2.array([5, 6, 7, 8]);
//!
//! // Perform operations
//! const sum = scirs2.add(a, b);
//! const dot = scirs2.dot(a, b);
//!
//! // Statistical operations
//! const mean = scirs2.mean(a);
//! const std = scirs2.std(a);
//!
//! // Linear algebra
//! const matrix = scirs2.array2d([[1, 2], [3, 4]]);
//! const inv = scirs2.inv(matrix);
//! const det = scirs2.det(matrix);
//! ```

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

use std::alloc::{alloc, dealloc, Layout};

use wasm_bindgen::prelude::*;
use web_sys::console;

pub mod array;
pub mod error;
pub mod fft;
pub mod integrate;
pub mod interpolate;
pub mod linalg;
pub mod optimize;
pub mod random;
pub mod signal;
pub mod stats;
pub mod utils;

// Async streaming via wasm-bindgen-futures
pub mod async_streaming;
// SharedArrayBuffer / Atomics JS bindings
pub mod shared_memory;

// Incremental PCA for WASM
pub mod incremental_pca;
// MFCC feature extraction for WASM
pub mod mfcc;
// Parallel WASM (SharedArrayBuffer)
pub mod parallel;
// Streaming FFT for WASM
pub mod streaming_fft;
// Wavelet transforms for WASM
pub mod wavelets;
// WebGPU compute shaders
pub mod webgpu;

/// Initialize the WASM module with panic hooks and logging
#[wasm_bindgen(start)]
pub fn init() {
    // Set panic hook for better error messages in the console
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();

    // Log initialization message
    log("SciRS2-WASM initialized successfully");
}

/// Get the version of SciRS2-WASM
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Log a message to the browser console
#[wasm_bindgen]
pub fn log(message: &str) {
    console::log_1(&JsValue::from_str(message));
}

/// Check if WASM SIMD is supported in the current environment
#[wasm_bindgen]
pub fn has_simd_support() -> bool {
    #[cfg(target_feature = "simd128")]
    {
        true
    }
    #[cfg(not(target_feature = "simd128"))]
    {
        false
    }
}

/// Get system capabilities and features available in this build
#[wasm_bindgen]
pub fn capabilities() -> JsValue {
    let caps = serde_json::json!({
        "version": env!("CARGO_PKG_VERSION"),
        "simd": has_simd_support(),
        "features": {
            "array": true,
            "linalg": cfg!(feature = "linalg"),
            "stats": cfg!(feature = "stats"),
            "fft": cfg!(feature = "fft"),
            "signal": cfg!(feature = "signal"),
            "integrate": cfg!(feature = "integrate"),
            "optimize": cfg!(feature = "optimize"),
            "interpolate": cfg!(feature = "interpolate"),
        },
        "target": {
            "arch": std::env::consts::ARCH,
            "os": "wasm32",
            "family": std::env::consts::FAMILY,
        }
    });

    serde_wasm_bindgen::to_value(&caps).unwrap_or(JsValue::NULL)
}

/// Performance timing utilities for benchmarking WASM operations
#[wasm_bindgen]
pub struct PerformanceTimer {
    start: f64,
    label: String,
}

#[wasm_bindgen]
impl PerformanceTimer {
    /// Create a new performance timer with a label
    #[wasm_bindgen(constructor)]
    pub fn new(label: String) -> Result<PerformanceTimer, JsValue> {
        let start = web_sys::window()
            .ok_or_else(|| JsValue::from_str("No window object available"))?
            .performance()
            .ok_or_else(|| JsValue::from_str("No performance object available"))?
            .now();

        Ok(PerformanceTimer { start, label })
    }

    /// Get elapsed time in milliseconds
    pub fn elapsed(&self) -> Result<f64, JsValue> {
        let now = web_sys::window()
            .ok_or_else(|| JsValue::from_str("No window object available"))?
            .performance()
            .ok_or_else(|| JsValue::from_str("No performance object available"))?
            .now();

        Ok(now - self.start)
    }

    /// Log the elapsed time to console
    pub fn log_elapsed(&self) -> Result<(), JsValue> {
        let elapsed = self.elapsed()?;
        let message = format!("{}: {:.3}ms", self.label, elapsed);
        console::log_1(&JsValue::from_str(&message));
        Ok(())
    }
}

/// Memory usage information for the WASM module
#[wasm_bindgen]
pub fn memory_usage() -> JsValue {
    let info = serde_json::json!({
        "note": "WASM memory usage should be checked via JavaScript Memory API"
    });

    serde_wasm_bindgen::to_value(&info).unwrap_or(JsValue::NULL)
}

// ---------------------------------------------------------------------------
// WASM linear-memory allocation helpers
// ---------------------------------------------------------------------------

/// Allocate `len` f64 values in WASM linear memory.
///
/// Returns a pointer (as `usize`) to the start of the allocated buffer.
/// The caller is responsible for eventually freeing the buffer by passing
/// the same pointer and length to [`free_f64_array`].
///
/// Returns `0` when `len` is zero; panics on allocation failure.
#[wasm_bindgen]
pub fn alloc_f64_array(len: usize) -> usize {
    if len == 0 {
        return 0;
    }
    let layout = Layout::array::<f64>(len).expect("alloc_f64_array: layout overflow");
    // SAFETY: layout is non-zero (len > 0 checked above).
    let ptr = unsafe { alloc(layout) };
    assert!(!ptr.is_null(), "alloc_f64_array: allocation failed");
    ptr as usize
}

/// Free a buffer previously allocated with [`alloc_f64_array`].
///
/// Both `ptr` and `len` must match exactly the values used when the buffer
/// was originally allocated.  Passing mismatched values is undefined
/// behaviour.  A `ptr` of `0` or a `len` of `0` is silently ignored.
#[wasm_bindgen]
pub fn free_f64_array(ptr: usize, len: usize) {
    if ptr == 0 || len == 0 {
        return;
    }
    let layout = Layout::array::<f64>(len).expect("free_f64_array: layout overflow");
    // SAFETY: `ptr` was returned by `alloc_f64_array` with the same `len`,
    // and WASM is single-threaded so no concurrent access is possible.
    unsafe { dealloc(ptr as *mut u8, layout) };
}

/// Return a `Float64Array` view into WASM linear memory at `ptr` / `len`.
///
/// This exposes a zero-copy window from JavaScript into the WASM heap.
/// The returned typed array is only valid while the underlying allocation
/// remains live; callers must not use the view after calling
/// [`free_f64_array`] for the same pointer.
///
/// Available on the `wasm32` target only.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn view_f64_array(ptr: usize, len: usize) -> js_sys::Float64Array {
    // SAFETY: `ptr` points to a live, f64-aligned allocation of at least
    // `len * 8` bytes allocated by `alloc_f64_array`.  WASM is
    // single-threaded, so there are no concurrent mutations.
    let slice = unsafe { std::slice::from_raw_parts(ptr as *const f64, len) };
    // SAFETY: `slice` is valid for the duration of this call; the returned
    // Float64Array must not outlive the underlying allocation.
    unsafe { js_sys::Float64Array::view(slice) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        let v = version();
        assert!(!v.is_empty());
    }

    #[test]
    fn test_has_simd_support() {
        // Should not panic
        let _simd = has_simd_support();
    }

    // -----------------------------------------------------------------------
    // alloc / free round-trip tests (native; WASM targets use wasm-bindgen-test)
    // -----------------------------------------------------------------------

    /// Allocating zero elements must return the sentinel 0 without allocating.
    #[test]
    fn test_alloc_zero_returns_zero() {
        let ptr = alloc_f64_array(0);
        assert_eq!(ptr, 0);
        // free_f64_array(0, 0) must be a no-op.
        free_f64_array(0, 0);
    }

    /// Allocate, write, read back, and free a single f64.
    #[test]
    fn test_alloc_free_single_element() {
        let len = 1_usize;
        let ptr = alloc_f64_array(len);
        assert_ne!(ptr, 0);

        // SAFETY: ptr is a valid, non-null allocation of len * 8 bytes.
        unsafe {
            let p = ptr as *mut f64;
            p.write(std::f64::consts::PI);
            let read_back = p.read();
            assert!(
                (read_back - std::f64::consts::PI).abs() < f64::EPSILON,
                "expected PI, got {read_back}"
            );
        }

        free_f64_array(ptr, len);
    }

    /// Allocate a larger buffer, write a pattern, verify it, then free.
    #[test]
    fn test_alloc_free_round_trip() {
        let len = 256_usize;
        let ptr = alloc_f64_array(len);
        assert_ne!(ptr, 0);

        // SAFETY: ptr is a valid allocation of len * 8 bytes.
        unsafe {
            let slice = std::slice::from_raw_parts_mut(ptr as *mut f64, len);
            for (i, elem) in slice.iter_mut().enumerate() {
                *elem = i as f64 * 1.5;
            }
            for (i, &val) in slice.iter().enumerate() {
                let expected = i as f64 * 1.5;
                assert!(
                    (val - expected).abs() < f64::EPSILON,
                    "index {i}: expected {expected}, got {val}"
                );
            }
        }

        free_f64_array(ptr, len);
    }

    /// free_f64_array with ptr==0 or len==0 must be silent no-ops.
    #[test]
    fn test_free_noop_on_null_or_zero_len() {
        free_f64_array(0, 128);
        free_f64_array(0, 0);
        // Allocate then use len==0 to demonstrate no-op free path.
        let ptr = alloc_f64_array(4);
        free_f64_array(ptr, 0); // no-op — we still need to free properly
        free_f64_array(ptr, 4); // actual free
    }
}
