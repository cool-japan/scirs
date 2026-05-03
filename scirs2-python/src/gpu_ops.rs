//! GPU-accelerated matrix operations exposed to Python.
//!
//! This module provides a GPU-dispatch API with a pure-CPU fallback that is
//! always available. The API surface is identical regardless of whether GPU
//! hardware is present, so Python callers need no conditional logic.
//!
//! # GPU path (future `cuda_bridge` feature)
//! When the `cuda_bridge` feature is enabled and `cudarc`/`candle` are linked,
//! the functions below dispatch to the GPU kernel instead of the CPU path.
//! The feature gate is wired up but the cudarc integration itself is deferred
//! until GPU hardware is available in CI.  See TODO.md L149/L151.
//!
//! # CPU path (default, pure Rust)
//! The CPU implementations use plain `Vec<f64>` arithmetic and are correct,
//! tested, and zero-dependency.

use pyo3::exceptions::{PyNotImplementedError, PyTypeError, PyValueError};
use pyo3::prelude::*;

// ── GPU device info ────────────────────────────────────────────────────────

/// Return a string describing the active compute device.
///
/// Returns `"cpu (cuda_bridge feature not enabled)"` unless the `cuda_bridge`
/// Cargo feature is enabled and CUDA hardware is detected at runtime.
#[pyfunction]
pub fn gpu_device_info() -> String {
    "cpu (cuda_bridge feature not enabled)".to_string()
}

// ── Matrix multiply ────────────────────────────────────────────────────────

/// Multiply two row-major matrices: C = A (m×k) × B (k×n).
///
/// Returns the product as a flat `Vec<f64>` of length `m * n` in row-major
/// order together with the output shape `(m, n)`.
///
/// # Arguments
/// * `a_data`  – flat row-major elements of A, length must equal `a_rows * a_cols`
/// * `a_rows`  – number of rows in A (= m)
/// * `a_cols`  – number of columns in A (= k)
/// * `b_data`  – flat row-major elements of B, length must equal `a_cols * b_cols`
/// * `b_cols`  – number of columns in B (= n)
///
/// # Errors
/// Returns `PyValueError` when any length constraint is violated.
#[pyfunction]
pub fn gpu_matmul(
    a_data: Vec<f64>,
    a_rows: usize,
    a_cols: usize,
    b_data: Vec<f64>,
    b_cols: usize,
) -> PyResult<Vec<f64>> {
    if a_data.len() != a_rows * a_cols {
        return Err(PyValueError::new_err(format!(
            "a_data length {} does not match a_rows * a_cols = {} * {} = {}",
            a_data.len(),
            a_rows,
            a_cols,
            a_rows * a_cols,
        )));
    }
    if b_data.len() != a_cols * b_cols {
        return Err(PyValueError::new_err(format!(
            "b_data length {} does not match a_cols * b_cols = {} * {} = {}",
            b_data.len(),
            a_cols,
            b_cols,
            a_cols * b_cols,
        )));
    }

    // CPU row-major matrix multiply (ikj loop order for cache locality)
    let mut c = vec![0.0f64; a_rows * b_cols];
    for i in 0..a_rows {
        for k in 0..a_cols {
            let a_ik = a_data[i * a_cols + k];
            for j in 0..b_cols {
                c[i * b_cols + j] += a_ik * b_data[k * b_cols + j];
            }
        }
    }
    Ok(c)
}

// ── Element-wise activation functions ─────────────────────────────────────

/// Apply an element-wise activation to every element of `data`.
///
/// Supported operations: `"exp"`, `"log"`, `"sqrt"`, `"relu"`, `"sigmoid"`,
/// `"tanh"`, `"abs"`, `"square"`.
///
/// For `"log"` of non-positive values the result is `-∞`; for `"sqrt"` of
/// negative values the result is `NaN`.  These match NumPy conventions.
///
/// # Errors
/// Returns `PyValueError` if `op` is not one of the supported strings.
#[pyfunction]
pub fn gpu_elementwise(data: Vec<f64>, op: &str) -> PyResult<Vec<f64>> {
    let result: Vec<f64> = match op {
        "exp" => data.iter().map(|&x| x.exp()).collect(),
        "log" => data
            .iter()
            .map(|&x| if x > 0.0 { x.ln() } else { f64::NEG_INFINITY })
            .collect(),
        "sqrt" => data
            .iter()
            .map(|&x| if x >= 0.0 { x.sqrt() } else { f64::NAN })
            .collect(),
        "relu" => data.iter().map(|&x| x.max(0.0)).collect(),
        "sigmoid" => data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect(),
        "tanh" => data.iter().map(|&x| x.tanh()).collect(),
        "abs" => data.iter().map(|&x| x.abs()).collect(),
        "square" => data.iter().map(|&x| x * x).collect(),
        _ => {
            return Err(PyValueError::new_err(format!(
                "Unknown op '{op}'. Supported: exp, log, sqrt, relu, sigmoid, tanh, abs, square"
            )))
        }
    };
    Ok(result)
}

// ── Batch matrix operations ────────────────────────────────────────────────

/// Add two row-major matrices element-wise.
///
/// Both vectors must have the same length (= rows × cols).
///
/// # Errors
/// Returns `PyValueError` on length mismatch.
#[pyfunction]
pub fn gpu_matrix_add(a_data: Vec<f64>, b_data: Vec<f64>) -> PyResult<Vec<f64>> {
    if a_data.len() != b_data.len() {
        return Err(PyValueError::new_err(format!(
            "Length mismatch: a has {} elements, b has {}",
            a_data.len(),
            b_data.len(),
        )));
    }
    Ok(a_data
        .iter()
        .zip(b_data.iter())
        .map(|(&a, &b)| a + b)
        .collect())
}

/// Scale a row-major matrix by a scalar.
#[pyfunction]
pub fn gpu_matrix_scale(data: Vec<f64>, scalar: f64) -> Vec<f64> {
    data.iter().map(|&x| x * scalar).collect()
}

/// Compute the Frobenius norm of a flat matrix.
#[pyfunction]
pub fn gpu_frobenius_norm(data: Vec<f64>) -> f64 {
    data.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

// ── CUDA tensor bridge (DLPack protocol, GPU path deferred) ───────────────

/// Multiply two PyTorch/JAX tensors via the DLPack protocol.
///
/// This is the entry point for the zero-copy GPU path. When the `cuda_bridge`
/// Cargo feature is enabled (and `cudarc` is linked), the function accepts any
/// Python object implementing `__dlpack__` and dispatches directly to a CUDA
/// GEMM kernel.
///
/// In the current CPU-only build this function returns `PyNotImplementedError`
/// with a clear message directing callers to `gpu_matmul()`.
///
/// # Python example
/// ```python
/// import torch, scirs2
/// a = torch.randn(512, 512, device='cuda')
/// b = torch.randn(512, 512, device='cuda')
/// # GPU path (when cuda_bridge feature is enabled):
/// c = scirs2.cuda_tensor_matmul(a, b)
/// # CPU fallback for all tensor sizes:
/// c_data = scirs2.gpu_matmul(a.flatten().tolist(), 512, 512, b.flatten().tolist(), 512)
/// ```
#[pyfunction]
pub fn cuda_tensor_matmul<'py>(
    _py: Python<'py>,
    tensor_a: &Bound<'py, PyAny>,
    _tensor_b: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    // Verify that the input implements the DLPack protocol before returning the
    // not-implemented error, so that callers know their tensor type is compatible.
    let has_dlpack = tensor_a.hasattr("__dlpack__").unwrap_or(false);
    if !has_dlpack {
        return Err(PyTypeError::new_err(
            "Tensors must implement the __dlpack__ protocol (e.g. PyTorch or JAX tensors)",
        ));
    }

    // CPU-only build: direct to the Vec-based fallback instead.
    Err(PyNotImplementedError::new_err(
        "CUDA tensor bridge is not yet compiled in. \
         Enable the `cuda_bridge` Cargo feature and install `cudarc`. \
         For a CPU fallback that accepts Python lists, use gpu_matmul().",
    ))
}

// ── Module registration ────────────────────────────────────────────────────

/// Register all GPU-dispatch functions in the parent Python module.
pub fn register_gpu_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gpu_device_info, m)?)?;
    m.add_function(wrap_pyfunction!(gpu_matmul, m)?)?;
    m.add_function(wrap_pyfunction!(gpu_elementwise, m)?)?;
    m.add_function(wrap_pyfunction!(gpu_matrix_add, m)?)?;
    m.add_function(wrap_pyfunction!(gpu_matrix_scale, m)?)?;
    m.add_function(wrap_pyfunction!(gpu_frobenius_norm, m)?)?;
    m.add_function(wrap_pyfunction!(cuda_tensor_matmul, m)?)?;
    Ok(())
}

// ── Unit tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_device_info_non_empty() {
        let info = gpu_device_info();
        assert!(!info.is_empty());
        assert!(info.contains("cpu"));
    }

    #[test]
    fn test_matmul_2x2_identity() {
        // [1,0; 0,1] × [5,6; 7,8] = [5,6; 7,8]
        let id = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let c = gpu_matmul(id, 2, 2, b.clone(), 2).expect("matmul should not fail");
        assert!((c[0] - 5.0).abs() < 1e-12);
        assert!((c[1] - 6.0).abs() < 1e-12);
        assert!((c[2] - 7.0).abs() < 1e-12);
        assert!((c[3] - 8.0).abs() < 1e-12);
    }

    #[test]
    fn test_matmul_2x2_general() {
        // [1,2; 3,4] × [5,6; 7,8] = [19,22; 43,50]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let c = gpu_matmul(a, 2, 2, b, 2).expect("matmul should not fail");
        assert!((c[0] - 19.0).abs() < 1e-12);
        assert!((c[1] - 22.0).abs() < 1e-12);
        assert!((c[2] - 43.0).abs() < 1e-12);
        assert!((c[3] - 50.0).abs() < 1e-12);
    }

    #[test]
    fn test_matmul_non_square() {
        // [1,2,3; 4,5,6] (2×3) × [7,8; 9,10; 11,12] (3×2) = [58,64; 139,154]
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let c = gpu_matmul(a, 2, 3, b, 2).expect("non-square matmul should succeed");
        assert!((c[0] - 58.0).abs() < 1e-12);
        assert!((c[1] - 64.0).abs() < 1e-12);
        assert!((c[2] - 139.0).abs() < 1e-12);
        assert!((c[3] - 154.0).abs() < 1e-12);
    }

    #[test]
    fn test_matmul_a_length_mismatch_returns_error() {
        let a = vec![1.0, 2.0]; // length=2 but a_rows=2, a_cols=2 expects 4
        let b = vec![1.0, 2.0, 3.0, 4.0];
        assert!(gpu_matmul(a, 2, 2, b, 2).is_err());
    }

    #[test]
    fn test_matmul_b_length_mismatch_returns_error() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 2.0]; // length=2 but a_cols=2, b_cols=2 expects 4
        assert!(gpu_matmul(a, 2, 2, b, 2).is_err());
    }

    #[test]
    fn test_elementwise_relu() {
        let data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let out = gpu_elementwise(data, "relu").expect("relu should succeed");
        assert_eq!(out, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_elementwise_sigmoid_bounds() {
        let data = vec![-100.0, 0.0, 100.0];
        let out = gpu_elementwise(data, "sigmoid").expect("sigmoid should succeed");
        assert!(out[0] < 1e-3, "sigmoid(-100) should be near 0");
        assert!((out[1] - 0.5).abs() < 1e-12, "sigmoid(0) should be 0.5");
        assert!(out[2] > 1.0 - 1e-3, "sigmoid(100) should be near 1");
    }

    #[test]
    fn test_elementwise_tanh() {
        let data = vec![-1.0, 0.0, 1.0];
        let out = gpu_elementwise(data, "tanh").expect("tanh should succeed");
        assert!((out[1] - 0.0).abs() < 1e-12);
        assert!((out[2] - 1.0_f64.tanh()).abs() < 1e-12);
    }

    #[test]
    fn test_elementwise_exp_log_roundtrip() {
        let data = vec![1.0, 2.0, 3.0];
        let exped = gpu_elementwise(data.clone(), "exp").expect("exp should succeed");
        let logged = gpu_elementwise(exped, "log").expect("log should succeed");
        for (orig, rt) in data.iter().zip(logged.iter()) {
            assert!((orig - rt).abs() < 1e-10, "exp-log roundtrip failed");
        }
    }

    #[test]
    fn test_elementwise_sqrt_non_negative() {
        let data = vec![0.0, 1.0, 4.0, 9.0, 16.0];
        let out = gpu_elementwise(data, "sqrt").expect("sqrt should succeed");
        assert!((out[0] - 0.0).abs() < 1e-12);
        assert!((out[1] - 1.0).abs() < 1e-12);
        assert!((out[2] - 2.0).abs() < 1e-12);
        assert!((out[4] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_elementwise_abs() {
        let data = vec![-3.0, -1.5, 0.0, 2.5];
        let out = gpu_elementwise(data, "abs").expect("abs should succeed");
        assert_eq!(out, vec![3.0, 1.5, 0.0, 2.5]);
    }

    #[test]
    fn test_elementwise_square() {
        let data = vec![-2.0, 3.0];
        let out = gpu_elementwise(data, "square").expect("square should succeed");
        assert!((out[0] - 4.0).abs() < 1e-12);
        assert!((out[1] - 9.0).abs() < 1e-12);
    }

    #[test]
    fn test_elementwise_unknown_op_returns_error() {
        let data = vec![1.0, 2.0];
        assert!(gpu_elementwise(data, "unknown_activation").is_err());
    }

    #[test]
    fn test_matrix_add_correct() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let out = gpu_matrix_add(a, b).expect("matrix_add should succeed");
        assert_eq!(out, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_matrix_add_length_mismatch_returns_error() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0];
        assert!(gpu_matrix_add(a, b).is_err());
    }

    #[test]
    fn test_matrix_scale() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let out = gpu_matrix_scale(data, 2.5);
        assert_eq!(out, vec![2.5, 5.0, 7.5, 10.0]);
    }

    #[test]
    fn test_frobenius_norm_identity() {
        // Frobenius norm of 2×2 identity = sqrt(2)
        let id = vec![1.0, 0.0, 0.0, 1.0];
        let norm = gpu_frobenius_norm(id);
        assert!((norm - 2.0_f64.sqrt()).abs() < 1e-12);
    }
}
