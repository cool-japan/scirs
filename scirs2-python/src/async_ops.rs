//! Async operations for Python
//!
//! This module provides async versions of long-running operations that can be awaited in Python.
//!
//! # Example (Python)
//! ```python
//! import asyncio
//! import scirs2
//! import numpy as np
//!
//! async def main():
//!     # Async FFT for large arrays
//!     data = np.random.randn(1_000_000)
//!     result = await scirs2.fft_async(data)
//!
//!     # Async matrix decomposition
//!     matrix = np.random.randn(1000, 1000)
//!     svd = await scirs2.svd_async(matrix)
//!
//! asyncio.run(main())
//! ```

use crate::error::SciRS2Error;
use pyo3::prelude::*;
use pyo3_async_runtimes;
use scirs2_numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};

/// Async FFT operation for large arrays
///
/// This function runs FFT in a background thread and returns a Python awaitable.
/// Useful for large arrays (>100k elements) to avoid blocking the event loop.
#[pyfunction]
pub fn fft_async<'py>(
    py: Python<'py>,
    data: &Bound<'_, PyArray1<f64>>,
) -> PyResult<Bound<'py, PyAny>> {
    let data_vec: Vec<f64> = {
        let binding = data.readonly();
        let arr = binding.as_array();
        arr.iter().cloned().collect()
    };

    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        // Run FFT in blocking task
        // FFT returns Vec<Complex64> - collect real and imag parts
        let (real_part, imag_part): (Vec<f64>, Vec<f64>) = tokio::task::spawn_blocking(move || {
            use scirs2_core::Complex64;
            use scirs2_fft::fft;

            let result: Vec<Complex64> = fft(data_vec.as_slice(), None)
                .map_err(|e| SciRS2Error::ComputationError(format!("FFT failed: {}", e)))?;

            let real: Vec<f64> = result.iter().map(|c| c.re).collect();
            let imag: Vec<f64> = result.iter().map(|c| c.im).collect();
            Ok::<(Vec<f64>, Vec<f64>), SciRS2Error>((real, imag))
        })
        .await
        .map_err(|e| SciRS2Error::RuntimeError(format!("Task join error: {}", e)))??;

        // Return as Py<PyAny> which is Send
        let py_result: Py<PyAny> = Python::attach(|py| {
            use pyo3::types::PyDict;
            use scirs2_core::Array1;
            // Return dict with real and imag arrays
            let dict = PyDict::new(py);
            let real_arr: Array1<f64> = Array1::from_vec(real_part);
            let imag_arr: Array1<f64> = Array1::from_vec(imag_part);
            dict.set_item("real", real_arr.into_pyarray(py))?;
            dict.set_item("imag", imag_arr.into_pyarray(py))?;
            Ok::<Py<PyAny>, PyErr>(dict.into_any().unbind())
        })?;

        Ok(py_result)
    })
}

/// Async SVD operation for large matrices
///
/// This function runs SVD in a background thread and returns a Python awaitable.
/// Useful for large matrices (>500x500) to avoid blocking the event loop.
#[pyfunction]
pub fn svd_async<'py>(
    py: Python<'py>,
    matrix: &Bound<'_, PyArray2<f64>>,
    full_matrices: Option<bool>,
) -> PyResult<Bound<'py, PyAny>> {
    let matrix_shape = matrix.shape().to_vec();
    let matrix_vec: Vec<f64> = {
        let binding = matrix.readonly();
        let arr = binding.as_array();
        arr.iter().cloned().collect()
    };
    let full_matrices = full_matrices.unwrap_or(true);

    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        // Run SVD in blocking task
        let result = tokio::task::spawn_blocking(move || {
            use scirs2_core::Array2;
            use scirs2_linalg::svd_f64_lapack;

            let arr = Array2::from_shape_vec((matrix_shape[0], matrix_shape[1]), matrix_vec)
                .map_err(|e| SciRS2Error::ArrayError(format!("Array reshape failed: {}", e)))?;

            svd_f64_lapack(&arr.view(), full_matrices)
                .map_err(|e| SciRS2Error::ComputationError(format!("SVD failed: {}", e)))
        })
        .await
        .map_err(|e| SciRS2Error::RuntimeError(format!("Task join error: {}", e)))??;

        // Convert result to Python dict; return Py<PyAny> which is Send
        let py_result: Py<PyAny> = Python::attach(|py| {
            use pyo3::types::PyDict;
            let dict = PyDict::new(py);
            dict.set_item("U", result.0.into_pyarray(py))?;
            dict.set_item("S", result.1.into_pyarray(py))?;
            dict.set_item("Vt", result.2.into_pyarray(py))?;
            Ok::<Py<PyAny>, PyErr>(dict.into_any().unbind())
        })?;

        Ok(py_result)
    })
}

/// Async QR decomposition for large matrices
#[pyfunction]
pub fn qr_async<'py>(
    py: Python<'py>,
    matrix: &Bound<'_, PyArray2<f64>>,
) -> PyResult<Bound<'py, PyAny>> {
    let matrix_shape = matrix.shape().to_vec();
    let matrix_vec: Vec<f64> = {
        let binding = matrix.readonly();
        let arr = binding.as_array();
        arr.iter().cloned().collect()
    };

    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let result = tokio::task::spawn_blocking(move || {
            use scirs2_core::Array2;
            use scirs2_linalg::qr_f64_lapack;

            let arr = Array2::from_shape_vec((matrix_shape[0], matrix_shape[1]), matrix_vec)
                .map_err(|e| SciRS2Error::ArrayError(format!("Array reshape failed: {}", e)))?;

            qr_f64_lapack(&arr.view())
                .map_err(|e| SciRS2Error::ComputationError(format!("QR failed: {}", e)))
        })
        .await
        .map_err(|e| SciRS2Error::RuntimeError(format!("Task join error: {}", e)))??;

        let py_result: Py<PyAny> = Python::attach(|py| {
            use pyo3::types::PyDict;
            let dict = PyDict::new(py);
            dict.set_item("Q", result.0.into_pyarray(py))?;
            dict.set_item("R", result.1.into_pyarray(py))?;
            Ok::<Py<PyAny>, PyErr>(dict.into_any().unbind())
        })?;

        Ok(py_result)
    })
}

/// Async numerical integration for expensive integrands
#[pyfunction]
pub fn quad_async<'py>(
    py: Python<'py>,
    func: Py<PyAny>,
    a: f64,
    b: f64,
    epsabs: Option<f64>,
    epsrel: Option<f64>,
) -> PyResult<Bound<'py, PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let result: (f64, f64) = tokio::task::spawn_blocking(move || {
            Python::attach(|py| {
                use scirs2_integrate::quad::{quad, QuadOptions};

                let abs_tol = epsabs.unwrap_or(1e-8);
                let rel_tol = epsrel.unwrap_or(1e-8);

                // Create Rust closure that calls Python function
                let integrand = |x: f64| -> f64 {
                    func.call1(py, (x,))
                        .and_then(|result| result.extract::<f64>(py))
                        .unwrap_or(f64::NAN)
                };

                let options = QuadOptions {
                    abs_tol,
                    rel_tol,
                    ..Default::default()
                };

                let result = quad(integrand, a, b, Some(options)).map_err(|e| {
                    PyErr::from(SciRS2Error::ComputationError(format!(
                        "Integration failed: {}",
                        e
                    )))
                })?;

                Ok::<(f64, f64), PyErr>((result.value, result.abs_error))
            })
        })
        .await
        .map_err(|e| SciRS2Error::RuntimeError(format!("Task join error: {}", e)))??;

        let py_result: Py<PyAny> = Python::attach(|py| {
            use pyo3::types::PyDict;
            let dict = PyDict::new(py);
            dict.set_item("value", result.0)?;
            dict.set_item("error", result.1)?;
            Ok::<Py<PyAny>, PyErr>(dict.into_any().unbind())
        })?;

        Ok(py_result)
    })
}

/// Async optimization for expensive objective functions
#[pyfunction]
pub fn minimize_async<'py>(
    py: Python<'py>,
    func: Py<PyAny>,
    x0: &Bound<'_, PyArray1<f64>>,
    method: Option<String>,
    maxiter: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let x0_vec: Vec<f64> = {
        let binding = x0.readonly();
        let arr = binding.as_array();
        arr.iter().cloned().collect()
    };

    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let result: (Vec<f64>, f64, usize) = tokio::task::spawn_blocking(move || {
            Python::attach(|py| {
                use scirs2_core::ndarray::ArrayView1;
                use scirs2_optimize::unconstrained::{minimize, Method};

                // Create Rust closure that calls Python function
                let objective = |x: &ArrayView1<f64>| -> f64 {
                    let x_slice = x.as_slice().unwrap_or(&[]);
                    let x_py = match pyo3::types::PyList::new(py, x_slice) {
                        Ok(list) => list,
                        Err(_) => return f64::NAN,
                    };
                    func.call1(py, (x_py,))
                        .and_then(|r| r.extract::<f64>(py))
                        .unwrap_or(f64::NAN)
                };

                let opt_method = match method.as_deref() {
                    Some("BFGS") => Method::BFGS,
                    Some("Newton") | Some("NewtonCG") => Method::NewtonCG,
                    Some("GradientDescent") | Some("CG") => Method::CG,
                    Some("NelderMead") => Method::NelderMead,
                    Some("LBFGS") => Method::LBFGS,
                    _ => Method::BFGS,
                };

                use scirs2_optimize::unconstrained::Options;
                let options = Options {
                    max_iter: maxiter.unwrap_or(1000),
                    ..Default::default()
                };

                let result =
                    minimize(objective, &x0_vec, opt_method, Some(options)).map_err(|e| {
                        PyErr::from(SciRS2Error::ComputationError(format!(
                            "Optimization failed: {}",
                            e
                        )))
                    })?;

                let x_vec = result.x.to_vec();
                let fun_val: f64 = result.fun;
                let nit = result.nit;
                Ok::<(Vec<f64>, f64, usize), PyErr>((x_vec, fun_val, nit))
            })
        })
        .await
        .map_err(|e| SciRS2Error::RuntimeError(format!("Task join error: {}", e)))??;

        let py_result: Py<PyAny> = Python::attach(|py| {
            use pyo3::types::PyDict;
            use scirs2_core::Array1;

            let dict = PyDict::new(py);
            let x = Array1::from_vec(result.0);
            dict.set_item("x", x.into_pyarray(py))?;
            dict.set_item("fun", result.1)?;
            dict.set_item("nit", result.2)?;
            Ok::<Py<PyAny>, PyErr>(dict.into_any().unbind())
        })?;

        Ok(py_result)
    })
}

/// Register async operations with Python module
pub fn register_async_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fft_async, m)?)?;
    m.add_function(wrap_pyfunction!(svd_async, m)?)?;
    m.add_function(wrap_pyfunction!(qr_async, m)?)?;
    m.add_function(wrap_pyfunction!(quad_async, m)?)?;
    m.add_function(wrap_pyfunction!(minimize_async, m)?)?;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use pyo3::prelude::*;
    use scirs2_core::Array1;
    use scirs2_core::Array2;
    use scirs2_numpy::{IntoPyArray, PyArray1, PyArray2};

    /// Boot the embedded interpreter and register a fresh asyncio event loop as the
    /// current thread's "running" loop, so that `pyo3_async_runtimes::tokio::future_into_py`
    /// (which internally calls `asyncio.get_running_loop()`) succeeds in plain Rust tests.
    fn install_running_loop(py: Python<'_>) {
        let asyncio = py.import("asyncio").expect("import asyncio");
        let loop_ = asyncio
            .call_method0("new_event_loop")
            .expect("new_event_loop");
        let events = py.import("asyncio.events").expect("import asyncio.events");
        events
            .call_method1("_set_running_loop", (loop_,))
            .expect("_set_running_loop");
    }

    /// Verify `fft_async` produces an awaitable (Bound<PyAny>) without panicking.
    ///
    /// We cannot `.await` the coroutine from plain Rust tests without a running
    /// asyncio event loop, but we can confirm that:
    ///   1. The function succeeds and returns a Python object.
    ///   2. The returned object claims to be a coroutine / awaitable.
    #[test]
    fn fft_async_returns_awaitable() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            install_running_loop(py);
            let data: Array1<f64> = Array1::from_vec(vec![1.0, 0.0, -1.0, 0.0]);
            let py_arr: Bound<'_, PyArray1<f64>> = data.into_pyarray(py);

            let result = super::fft_async(py, &py_arr);
            assert!(result.is_ok(), "fft_async returned Err: {:?}", result.err());
            // The returned object should have a `__await__` method (i.e. be a coroutine).
            let obj = result.expect("fft_async should succeed");
            assert!(
                obj.hasattr("__await__").unwrap_or(false)
                    || obj.hasattr("send").unwrap_or(false)
                    || obj.hasattr("__next__").unwrap_or(false),
                "returned object is not awaitable"
            );
        });
    }

    /// Verify `svd_async` produces an awaitable for a small square matrix.
    #[test]
    fn svd_async_returns_awaitable() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            install_running_loop(py);
            // 2×2 identity matrix
            let data: Array2<f64> =
                Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).expect("shape ok");
            let py_arr: Bound<'_, PyArray2<f64>> = data.into_pyarray(py);

            let result = super::svd_async(py, &py_arr, Some(false));
            assert!(result.is_ok(), "svd_async returned Err: {:?}", result.err());
            let obj = result.expect("svd_async should succeed");
            assert!(
                obj.hasattr("__await__").unwrap_or(false)
                    || obj.hasattr("send").unwrap_or(false)
                    || obj.hasattr("__next__").unwrap_or(false),
                "returned object is not awaitable"
            );
        });
    }

    /// Verify `qr_async` produces an awaitable for a small matrix.
    #[test]
    fn qr_async_returns_awaitable() {
        pyo3::Python::initialize();
        Python::attach(|py| {
            install_running_loop(py);
            let data: Array2<f64> =
                Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).expect("shape ok");
            let py_arr: Bound<'_, PyArray2<f64>> = data.into_pyarray(py);

            let result = super::qr_async(py, &py_arr);
            assert!(result.is_ok(), "qr_async returned Err: {:?}", result.err());
            let obj = result.expect("qr_async should succeed");
            assert!(
                obj.hasattr("__await__").unwrap_or(false)
                    || obj.hasattr("send").unwrap_or(false)
                    || obj.hasattr("__next__").unwrap_or(false),
                "returned object is not awaitable"
            );
        });
    }
}
