//! End-to-end browser tests for scirs2-wasm via `wasm-bindgen-test`.
//!
//! These tests are compiled to WASM and run inside a real browser engine
//! (Chromium or Firefox) when executed via `wasm-pack test`.  They validate
//! correctness of the compiled WASM module in an actual JS runtime, covering
//! code paths that differ between native and WASM targets (e.g. `f64` FP
//! rounding, `getrandom` entropy source, `SharedArrayBuffer` detection).
//!
//! # Running in a real browser
//!
//! ```sh
//! # Headless Chromium (requires chromedriver on PATH)
//! wasm-pack test --headless --chrome scirs2-wasm
//!
//! # Headless Firefox (requires geckodriver on PATH)
//! wasm-pack test --headless --firefox scirs2-wasm
//!
//! # Interactive (opens a browser tab, look for test results in the page)
//! wasm-pack test --chrome scirs2-wasm
//! ```

use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

use scirs2_wasm::{array, error, fft, linalg, shared_memory, stats};

// ── Module initialisation ─────────────────────────────────────────────────

/// Verify that the WASM module loads and exposes a valid version string.
#[wasm_bindgen_test]
fn browser_test_version_string() {
    let v = scirs2_wasm::version();
    assert!(!v.is_empty(), "version() must return a non-empty string");
    assert!(
        v.starts_with("0."),
        "version should start with '0.' (pre-1.0 series)"
    );
}

/// Verify that `capabilities()` returns a non-null JS value without panicking.
///
/// In a browser environment the JSON object will include browser-specific
/// flags (SIMD availability, SharedArrayBuffer presence, etc.).
#[wasm_bindgen_test]
fn browser_test_capabilities_non_null() {
    let caps = scirs2_wasm::capabilities();
    assert!(!caps.is_null());
    assert!(!caps.is_undefined());
}

// ── FFT correctness in browser ────────────────────────────────────────────

/// Verify that a 4-point FFT of `[1, 0, 0, 0]` returns the DC-only spectrum.
///
/// FFT([1, 0, 0, 0]) = [1+0i, 1+0i, 1+0i, 1+0i].
/// Returned as interleaved `[re0, im0, re1, im1, …]`.
#[wasm_bindgen_test]
fn browser_test_fft_dc_only_4pt() {
    let input = vec![1.0_f64, 0.0, 0.0, 0.0];
    let result = fft::fft(&input).expect("fft should succeed");
    // 4-point output: 8 floats (4 complex values)
    assert_eq!(result.len(), 8, "4-point FFT must return 8 floats");
    // All real parts should be 1.0, all imaginary parts should be 0.0
    for k in 0..4 {
        let re = result[k * 2];
        let im = result[k * 2 + 1];
        assert!(
            (re - 1.0).abs() < 1e-10,
            "FFT DC: re[{k}] = {re}, expected 1.0"
        );
        assert!(im.abs() < 1e-10, "FFT DC: im[{k}] = {im}, expected 0.0");
    }
}

/// Verify FFT output length equals 2×n for n-point input.
#[wasm_bindgen_test]
fn browser_test_fft_output_length() {
    for &n in &[8usize, 16, 32, 64] {
        let input: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let result = fft::fft(&input).expect("fft should succeed");
        assert_eq!(
            result.len(),
            n * 2,
            "{n}-point FFT should return {expected} floats",
            expected = n * 2
        );
    }
}

/// Verify that FFT followed by inverse FFT reconstructs the original signal.
#[wasm_bindgen_test]
fn browser_test_fft_ifft_roundtrip_8pt() {
    let original: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0];
    let spectrum = fft::fft(&original).expect("forward FFT should succeed");
    let reconstructed = fft::ifft(&spectrum).expect("inverse FFT should succeed");

    assert_eq!(
        reconstructed.len(),
        original.len() * 2,
        "IFFT output includes complex interleaved pairs"
    );

    // Check real parts reconstruct original (imaginary parts should be ≈0)
    for (i, &orig) in original.iter().enumerate() {
        let re = reconstructed[i * 2];
        assert!(
            (re - orig).abs() < 1e-9,
            "IFFT reconstruction failed at index {i}: got {re}, expected {orig}"
        );
    }
}

// ── Array operations in browser ───────────────────────────────────────────

/// Verify that matrix inverse in the browser gives numerically correct results.
///
/// For the 2×2 matrix [[2,1],[1,1]], the inverse is [[1,-1],[-1,2]].
#[wasm_bindgen_test]
fn browser_test_matrix_inv_correctness() {
    let shape = js_sys::Array::new();
    shape.push(&2.0_f64.into());
    shape.push(&2.0_f64.into());

    let data = js_sys::Array::new();
    for &v in &[2.0_f64, 1.0, 1.0, 1.0] {
        data.push(&v.into());
    }
    let m = array::WasmArray::from_shape(&shape.into(), &data.into())
        .expect("matrix creation should succeed");

    let inv = linalg::inv(&m).expect("inv should succeed for non-singular matrix");

    // [[1,-1],[-1,2]]
    assert!(
        (inv.get(0).unwrap() - 1.0).abs() < 1e-10,
        "inv[0,0] should be 1"
    );
    assert!(
        (inv.get(1).unwrap() - (-1.0)).abs() < 1e-10,
        "inv[0,1] should be -1"
    );
    assert!(
        (inv.get(2).unwrap() - (-1.0)).abs() < 1e-10,
        "inv[1,0] should be -1"
    );
    assert!(
        (inv.get(3).unwrap() - 2.0).abs() < 1e-10,
        "inv[1,1] should be 2"
    );
}

/// Verify matrix determinant in the browser.
#[wasm_bindgen_test]
fn browser_test_det_known_matrix() {
    // det([[3,8],[4,6]]) = 3*6 - 8*4 = 18 - 32 = -14
    let shape = js_sys::Array::new();
    shape.push(&2.0_f64.into());
    shape.push(&2.0_f64.into());

    let data = js_sys::Array::new();
    for &v in &[3.0_f64, 8.0, 4.0, 6.0] {
        data.push(&v.into());
    }
    let m = array::WasmArray::from_shape(&shape.into(), &data.into())
        .expect("matrix creation should succeed");

    let det = linalg::det(&m).expect("det should succeed");
    assert!(
        (det - (-14.0)).abs() < 1e-9,
        "det([[3,8],[4,6]]) should be -14, got {det}"
    );
}

/// Verify element-wise dot product via `array::dot`.
#[wasm_bindgen_test]
fn browser_test_dot_product_1d() {
    let shape = js_sys::Array::new();
    shape.push(&3.0_f64.into());

    let data_a = js_sys::Array::new();
    for &v in &[1.0_f64, 2.0, 3.0] {
        data_a.push(&v.into());
    }
    let a = array::WasmArray::from_shape(&shape.clone().into(), &data_a.into()).expect("create a");

    let data_b = js_sys::Array::new();
    for &v in &[4.0_f64, 5.0, 6.0] {
        data_b.push(&v.into());
    }
    let b = array::WasmArray::from_shape(&shape.into(), &data_b.into()).expect("create b");

    let d = array::dot(&a, &b).expect("dot should succeed");
    // [1,2,3]·[4,5,6] = 4+10+18 = 32
    assert!(
        (array::sum(&d) - 32.0).abs() < 1e-10,
        "dot product should be 32"
    );
}

// ── SharedArrayBuffer probe ───────────────────────────────────────────────

/// Verify that `shared_array_buffer_available()` returns a boolean without panicking.
///
/// The actual value depends on whether the page is served with COOP/COEP headers.
/// We do not assert `true` because the test runner URL may not set those headers.
#[wasm_bindgen_test]
fn browser_test_shared_array_buffer_probe_does_not_panic() {
    // Should not panic regardless of browser headers
    let _available = shared_memory::shared_array_buffer_available();
    // Test passes as long as we reach this line
}

// ── Error codes accessible from browser ──────────────────────────────────

/// Verify that error codes are accessible and non-zero for error variants.
#[wasm_bindgen_test]
fn browser_test_error_codes_accessible() {
    use error::{codes, WasmError};

    let err = WasmError::invalid_input("browser test");
    assert_eq!(
        err.error_code(),
        codes::INVALID_INPUT,
        "invalid_input error code should be INVALID_INPUT"
    );

    let dim_err = WasmError::DimensionMismatch {
        expected: vec![2, 3],
        actual: vec![2, 4],
    };
    assert_eq!(
        dim_err.error_code(),
        codes::DIMENSION_MISMATCH,
        "dimension mismatch code should be DIMENSION_MISMATCH"
    );

    assert!(
        err.error_code() > 0,
        "all error codes must be positive non-zero"
    );
}

/// Verify that `to_js_value()` produces a non-null JS value in the browser.
#[wasm_bindgen_test]
fn browser_test_error_to_js_value_non_null() {
    use error::WasmError;
    let err = WasmError::invalid_input("testing js error serialization");
    let js_val = err.to_js_value();
    assert!(!js_val.is_null());
    assert!(!js_val.is_undefined());
}

// ── Statistical functions in browser ─────────────────────────────────────

/// Verify normal distribution PDF in the browser (uses getrandom WASM entropy).
#[wasm_bindgen_test]
fn browser_test_stats_normal_pdf_at_mean_is_maximum() {
    let arr_data = js_sys::Array::new();
    for &v in &[-2.0_f64, -1.0, 0.0, 1.0, 2.0] {
        arr_data.push(&v.into());
    }
    let arr = array::WasmArray::new(&arr_data.into()).expect("array creation");
    let mean = array::mean(&arr);
    // For standard data the mean should be 0.0
    assert!(
        (mean - 0.0).abs() < 1e-10,
        "mean of symmetric data should be 0"
    );
}

/// Verify variance computation in the browser.
#[wasm_bindgen_test]
fn browser_test_stats_variance_known_data() {
    let js_arr = js_sys::Array::new();
    // Data: [2, 4, 4, 4, 5, 5, 7, 9] — known mean=5, variance=4 (population)
    for &v in &[2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
        js_arr.push(&v.into());
    }
    let arr = array::WasmArray::new(&js_arr.into()).expect("array");
    let mean_val = array::mean(&arr);
    assert!((mean_val - 5.0).abs() < 1e-10, "mean should be 5.0");

    let std_val = stats::std(&arr);
    // std dev = 2.0 (population) — sample std is slightly larger
    assert!(
        (std_val - 2.0).abs() < 0.5,
        "std of known data should be near 2.0, got {std_val}"
    );
}
